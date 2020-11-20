"""
Lightning-model for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

import os
import sys
import glob
import cv2
import numpy as np

from moving_mnist.moving_mnist_segmentation import make_moving_mnist
from pytorch_stream_loader.utils import grab_images_and_videos, normalize

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from torchvision.utils import make_grid

from kornia.utils import one_hot, mean_iou
from kornia.losses import DiceLoss, dice_loss

import skvideo.io


def normalize(im):
    low, high = im.min(), im.max()
    return (im - low) / (1e-5 + high - low)
    
def filter_outliers(input_val, num_std=3):
    val_range = num_std * input_val.std()
    img_min = input_val.mean() - val_range
    img_max = input_val.mean() + val_range
    normed = input_val.clamp_(img_min, img_max)
    return normed    

def search_latest_checkpoint(log_dir):
    """looks for latest checkpoint in latest sub-directory"""
    vdir = os.path.join(log_dir, 'checkpoints')
    ckpt = sorted(glob.glob(vdir + '/*.ckpt'))[-1]
    return ckpt


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, out, y):
        return self.xe(out,y) + self.dice(out,y)

        
class SegmentationLightningModel(pl.LightningModule) :
  
    def __init__(self, hparams):     
        super().__init__()

        self.model = model

        self.criterion = SegLoss()
        self.save_hyperparameters('batch_size', 'max_workers', 'tbins', 'max_frames_per_video', 'height', 'width', 'parallel', 'train_dir', 'val_dir', 'K', 'max_frames_per_epoch', 'max_objects')

    def _inference(self, batch, batch_nb):
        x, y, reset_mask = batch["histos"], batch["labels"], batch["reset"] # x : T,B,2,H,W // y : T,B,H,W
        
        T,B,C,H,W = x.shape
        x = x.float() # int -> float

        out = self.model.forward(x) # T,B,13,H,W
        
        out = out.reshape(T*B,self.hparams.K,H,W)
        y = y.reshape(T*B,H,W).long()
        return out, y

    def training_step(self,batch,batch_nb) :
        self.train()
        out, y = self._inference(batch,batch_nb)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return {'loss': loss}
        
    def training_epoch_end(self,training_step_outputs) :
        loss = torch.mean(torch.tensor([elt['loss'] for elt in training_step_outputs]))
        logs = {'train_loss' : loss}
        self.log('train_loss', loss)

        dataloader = self.val_dataloader()
        self.vizu(dataloader, self.current_epoch)
        return
            
    def validation_step(self, batch, batch_nb):
        out, y = self._inference(batch,batch_nb) # T*B,K,H,W / T*B,H,W
        loss = self.criterion(out, y)
        acc = self.pixels_acc(out,y)
        #acc = self.iou_acc(out,y) # T*B,K
        acc = torch.mean(acc,axis=0) # K // acc for every class
        self.log('val_loss', loss)
        self.log('acc', acc)
        return {'acc':acc, 'val_loss': loss.item()} 

    def validation_epoch_end(self, validation_step_outputs) :
        avg_loss = torch.mean(torch.tensor([elt['val_loss'] for elt in validation_step_outputs]))
        avg_acc = torch.mean(torch.tensor([elt['acc'] for elt in validation_step_outputs]))
        self.log('val_acc', avg_acc)
        self.log('avg_val_loss', avg_loss)

    def train_dataloader(self) :
        train_dataloader, _ = make_moving_mnist(train=True,max_objects=self.hparams.max_objects,height=self.hparams.height,width=self.hparams.width,tbins=self.hparams.tbins, max_frames_per_video=self.hparams.max_frames_per_video, max_frames_per_epoch=self.hparams.max_frames_per_epoch)
        return train_dataloader   
        
    def val_dataloader(self) :
        dataloader, _ = make_moving_mnist(train=False,max_objects=self.hparams.max_objects,height=self.hparams.height,width=self.hparams.width,tbins=self.hparams.tbins, max_frames_per_video=self.hparams.max_frames_per_video, max_frames_per_epoch=10000)
        return dataloader 
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
       
    def iou_acc(self,out,y) :
        y = y.long().cpu()
        out = torch.argmax(out,dim=1).long().cpu() # T*B,H,W
        return mean_iou(out,y,num_classes=self.hparams.K).cuda() # T*B,K

    def pixels_acc(self,out,y) :
        # out : T*B,13,H,W
        # y : T*B,H,W
        out = torch.argmax(out,dim=1) # T*B,H,W
        N,H,W = out.shape
        #return torch.sum(out==y).float()/(N*H*W)
        mask = (out==y)[y!=0]
        return torch.sum(mask).float()/torch.numel(mask)

    def demo_video(self):
        dataloader = self.val_dataloader()
        self.vizu(dataloader, -1)
        return

    def vizu(self,dataloader,epoch) :
        self.eval()
        
        cv2.namedWindow("histos",cv2.WINDOW_NORMAL)
        
        nrows = 2 ** ((self.hparams.batch_size.bit_length() - 1) // 2)
        
        count=0
        total_acc = None


        out_name = os.path.join(self.hparams.train_dir, "videos",  "test#"+str(epoch)+".mp4")

        dirname = os.path.dirname(out_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        out_video = skvideo.io.FFmpegWriter(out_name, outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec 'libx264'
        #'-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'slow'   #the slower the better compression, in princple, try 
                     #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        }) 
        
        with torch.no_grad() :
            for batch_idx, batch in enumerate(dataloader) :
                x, y, reset_mask = batch["histos"], batch["labels"], batch["reset"] # x.shape : T,B,2,H,W // y.shape : T,B,H,W
                T,B,C,H,W = x.shape
                x = x.float() # int -> float
                
                x = x.cuda()
                reset_mask = reset_mask.cuda()
                if hasattr(self.model , 'reset'):
                    self.model.reset(reset_mask)
                
                out = self.model.forward(x) # T,B,13,H,W

                out = torch.argmax(out,dim=2) # T,B,H,W
                out = out[:,:,None] # T,B,1,H,W
                
                x = 255*x
                
                tbins, batchsize = x.shape[:2]
                color = (0, 0, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                for t in range(len(x)):
                    #input
                    gridx = make_grid(x[t], nrow=nrows).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
                    #labels
                    gridt = make_grid(y[t][:,None], nrow=nrows).permute(1, 2, 0).cpu().numpy().copy()
                    grid_rgb = cv2.applyColorMap((gridt*30)%255, cv2.COLORMAP_RAINBOW)
                    grid_rgb[gridt==0] = 0

                    #prediction
                    grid2 = make_grid(out[t], nrow=nrows).permute(1,2,0).cpu().numpy().copy()
                    mask_background = np.max(grid2,axis=2)==0 # H,W
                    grid3 = cv2.applyColorMap(grid2.astype(np.uint8)*30, cv2.COLORMAP_RAINBOW)
                    grid3[mask_background] = 0
                    final_grid = grid3.astype(np.uint8)                    

                    frame = np.concatenate([gridx, final_grid, grid_rgb], axis=0)
                    cv2.imshow("histos",frame)

                    out_video.writeFrame(frame[...,::-1])
                    
                    key = cv2.waitKey(5)
                    if key == 27:
                        break  
                           
            cv2.destroyWindow("histos")                 
            out_video.close()
    
    
def train_moving_mnist_segmentation(train_dir, height=128, width=128, max_epochs=100, num_tbins=10, batch_size=64, num_workers=1, max_frames_per_video=100,
                                    max_frames_per_epoch=10000, max_objects=1, resume=False, just_demo=False):

    model = SequenceWise(SegVit2D(3,11))

    unet = LightningSegmentationModel()

    if resume:
        ckpt = search_latest_checkpoint(train_dir)
    else:
        ckpt = None
    
    tmpdir = os.path.join(train_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, period=2)

    logger = TestTubeLogger(
        save_dir=os.path.join(train_dir, 'logs'),
        version=1)

    if just_demo:
        checkpoint = torch.load(ckpt)
        unet.cuda()
        unet.load_state_dict(checkpoint['state_dict'])
        unet.demo_video()
    else:
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, gpus=1,precision=32,resume_from_checkpoint=ckpt)
        trainer.fit(unet)
  
  
if __name__ == "__main__" :
    import fire
    fire.Fire(train_moving_mnist_segmentation)
