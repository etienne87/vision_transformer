"""
Lightning-model for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

import os
import argparse
import cv2
import numpy as np
import tqdm
import skvideo.io

from torchvision.utils import make_grid
from types import SimpleNamespace
from kornia.utils import one_hot, mean_iou
from kornia.losses import DiceLoss, dice_loss
from core.temporal import time_to_batch



class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, out, y):
        return self.xe(out,y) + self.dice(out,y)


class SegmentationModel(pl.LightningModule) :

    def __init__(self, model, hparams: argparse.Namespace):
        super().__init__()

        self.model = model
        self.hparams = hparams
        self.criterion = SegLoss()

    def _inference(self, batch, batch_nb):
        x, y, reset_mask = batch["inputs"], batch["labels"], batch["mask_keep_memory"]
        if hasattr(self.model, "reset"):
            self.model.reset(reset_mask)
        out = self.model.forward(x)
        out = time_to_batch(out)[0]
        y = time_to_batch(y)[0].long()
        return out, y

    def training_step(self,batch,batch_nb) :
        out, y = self._inference(batch,batch_nb)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        out, y = self._inference(batch,batch_nb)
        loss = self.criterion(out, y)
        self.log('val_loss', loss)
        return {'val_loss': loss.item()}

    def validation_epoch_end(self, validation_step_outputs) :
        avg_loss = torch.mean(torch.tensor([elt['val_loss'] for elt in validation_step_outputs]))
        self.log('avg_val_loss', avg_loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def demo_video(self, dataloader, epoch=-1) :
        self.eval()

        cv2.namedWindow("histos",cv2.WINDOW_NORMAL)

        nrows = 2 ** ((self.hparams.batch_size.bit_length() - 1) // 2)

        count=0
        total_acc = None


        out_name = os.path.join(self.hparams.train_dir, "videos",  "test#"+str(epoch)+".mp4")

        dirname = os.path.dirname(out_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        out_video = skvideo.io.FFmpegWriter(out_name, outputdict={'-vcodec': 'libx264', '-preset':'slow'})

        with torch.no_grad() :
            for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
                x, y, reset_mask = batch["inputs"], batch["labels"], batch["mask_keep_memory"]

                x = x.to(self.device)
                reset_mask = reset_mask.cuda()
                if hasattr(self.model , 'reset'):
                    self.model.reset(reset_mask)

                out = self.model.forward(x)
                out = torch.argmax(out,dim=2)
                out = out[:,:,None]

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

                    frame = np.concatenate([gridx, final_grid, grid_rgb], axis=1)
                    cv2.imshow("histos",frame)

                    out_video.writeFrame(frame[...,::-1])

                    key = cv2.waitKey(5)
                    if key == 27:
                        break

            cv2.destroyWindow("histos")
            out_video.close()


