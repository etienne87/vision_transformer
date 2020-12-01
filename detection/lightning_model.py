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
import skvideo.io

from moving_mnist.moving_mnist_detection import make_moving_mnist
from core.temporal import time_to_batch
from detection.hungarian_loss import HungarianMatcher, SetCriterion
from detection.utils import normalize, filter_outliers
from detection.box_ops import box_xyxy_to_cxcywh
from detection.post_process import PostProcess
from data import box_api

from types import SimpleNamespace
from itertools import chain, islice
from torchvision.utils import make_grid


        
class DetectionModel(pl.LightningModule) :
  
    def __init__(self, model, hparams: argparse.Namespace):     
        super().__init__()

        self.model = model
        self.hparams = hparams
        self.num_classes = 10
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': hparams.bbox_loss_coef, 'loss_giou': hparams.giou_loss_coef}
        matcher = HungarianMatcher(hparams.cost_class, hparams.cost_bbox, hparams.cost_giou)
        self.criterion = SetCriterion(self.num_classes, matcher, hparams.eos_coef, losses = ['labels', 'boxes', 'cardinality'])
        self.post_process = PostProcess()

    def _inference(self, batch):
        x, reset_mask = batch["inputs"], batch["mask_keep_memory"] 
        if hasattr(self.model, "reset"):
            self.model.reset(reset_mask)
        out = self.model.forward(x) 
        out = time_to_batch(out)[0]
        out = {'pred_logits': out[...,4:], 'pred_boxes': out[..., :4].sigmoid()} 
        return out
    
    @torch.no_grad()
    def get_boxes(self, batch, score_thresh):
        t, b, _, h, w = batch['inputs'].shape
        out = self._inference(batch)
        batch_size = len(out['pred_logits'])
        target_sizes = torch.FloatTensor([h, w]*batch_size).reshape(batch_size, 2).type_as(batch['inputs'])
        boxes = self.post_process(out, target_sizes, score_thresh) 
        boxes_txn = [boxes[i*b:(i+1)*b] for i in range(t)]
        # boxes_txn = [[None]*b]*t
        # for i in range(len(boxes)):
        #     tbin = i//t
        #     num = i%b
        #     boxes_txn[tbin][num] = boxes[i]
        return boxes_txn

    def get_loss(self, batch, batch_nb):
        h, w = batch['inputs'].shape[-2:]
        scale_factor = torch.tensor([1./w, 1./h, 1./w, 1./h]).type_as(batch['inputs'])
        targets = batch['labels']
        targets = list(chain.from_iterable(targets))
        targets = [{'labels': bbox[:,4].long(), 'boxes': box_xyxy_to_cxcywh(bbox[:,:4])*scale_factor} for bbox in targets]
        out = self._inference(batch)
        loss_dict = self.criterion(out, targets)
        loss = sum([loss_dict[key]*weight for key, weight in self.weight_dict.items()]) 
        return out, loss, loss_dict

    def training_step(self,batch, batch_nb) :
        _, loss, loss_dict = self.get_loss(batch, batch_nb)
        for key in self.weight_dict.keys():
            self.log('train_loss_'+key, loss_dict[key])
        self.log('train_loss', loss.item())
        return {'loss': loss}
        
    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            out, loss, loss_dict = self.get_loss(batch, batch_nb)
        for key in self.weight_dict.keys():
            self.log('train_loss_'+key, loss_dict[key])
        self.log('val_loss', loss.item())
        #TODO
        #self.accumulate_results(out)

    def validation_epoch_end(self, validation_step_outputs) :
        avg_loss = torch.mean(torch.tensor([elt['val_loss'] for elt in validation_step_outputs]))
        self.log('avg_val_loss', avg_loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]
  
    @torch.no_grad()
    def demo_video(self, dataloader, num_batches=10000, show_video=False):
        """
        This runs our detector on several videos of the testing dataset
        """
        hparams = self.hparams

        height, width = hparams.height, hparams.width
        batch_size = hparams.batch_size
        nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
        ncols = int(np.ceil(hparams.batch_size / nrows))

        grid = np.zeros((nrows * hparams.height, ncols * hparams.width, 3), dtype=np.uint8)
        video_name = os.path.join(hparams.train_dir, 'videos', f'video#-1.mp4')

        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)

        out_video = skvideo.io.FFmpegWriter(video_name, outputdict={'-vcodec': 'libx264', '-preset':'slow'}) 

        self.eval()

        if show_video:
            window_name = 'detection'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for batch_nb, batch in enumerate(islice(dataloader, num_batches)):
            images = batch["inputs"].cpu().clone().data.numpy()

            batch["inputs"] = batch["inputs"].to(self.device)
            batch["mask_keep_memory"] = batch["mask_keep_memory"].to(self.device)

            predictions = self.get_boxes(batch, score_thresh=0.5)

            for t in range(len(images)):
                for i in range(len(images[0])):
                    frame = dataloader.get_vis_func()(images[t][i])
                    pred = predictions[t][i]
                    target = batch["labels"][t][i]

                    if isinstance(target, torch.Tensor):
                        target = target.cpu().numpy()
                    if target.dtype.isbuiltin:
                        target = box_api.box_vectors_to_bboxes(target[:,:4], target[:,4])

                    if len(pred['scores']):
                        boxes = pred['boxes'].cpu().data.numpy()
                        labels = pred['labels'].cpu().data.numpy()
                        scores = pred['scores'].cpu().data.numpy()
                        bboxes = box_api.box_vectors_to_bboxes(boxes, labels, scores)
                        frame = box_api.draw_box_events(frame, bboxes, dataloader.label_map, draw_score=True, thickness=2)

                    frame = box_api.draw_box_events(frame, target, dataloader.label_map, force_color=[255,255,255], draw_score=False, thickness=1)

                    y = i // ncols
                    x = i % ncols
                    y1, y2 = y*height, (y+1)*height
                    x1, x2 = x*width, (x+1)*width
                    grid[y1:y2,x1:x2] = frame

                if show_video:
                    cv2.imshow(window_name, grid)
                    cv2.waitKey(5)


                out_video.writeFrame(grid[...,::-1])

        if show_video:
            cv2.destroyWindow(window_name)
        out_video.close()
