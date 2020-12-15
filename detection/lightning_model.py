"""
Lightning-model for detection 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

import os
import argparse
import random
import cv2
import numpy as np
import skvideo.io

from moving_mnist.moving_mnist_detection import make_moving_mnist
from core.temporal import time_to_batch
from detection.hungarian_loss import HungarianMatcher, SetCriterion
from detection.utils import normalize, filter_outliers
from detection.box_ops import box_xyxy_to_cxcywh
from detection.post_process import PostProcess
from detection.coco_eval import coco_evaluation
from data import box_api

from types import SimpleNamespace
from itertools import chain, islice
from torchvision.utils import make_grid
from collections import defaultdict

from detection.utils import cuda_time

        
class DetectionModel(pl.LightningModule) :
  
    def __init__(self, model, hparams: argparse.Namespace):     
        super().__init__()

        self.model = model
        self.hparams = hparams
        self.num_classes = 10
        self.label_map = [str(i) for i in range(self.num_classes)]
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': hparams.bbox_loss_coef, 'loss_giou': hparams.giou_loss_coef}
        matcher = HungarianMatcher(hparams.cost_class, hparams.cost_bbox, hparams.cost_giou)
        self.criterion = SetCriterion(self.num_classes, matcher, hparams.eos_coef, losses = ['labels', 'boxes', 'cardinality'])
        self.post_process = PostProcess()

    def on_epoch_start(self):
        np.random.seed(self.current_epoch)
        random.seed(self.current_epoch)

    #@cuda_time
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
        out = self._inference(batch)
        return self.get_boxes_from_network_output(out, batch['inputs'].shape, score_thresh)

    @torch.no_grad()
    def get_boxes_from_network_output(self, out, in_shape, score_thresh):
        t,b,_,h,w = in_shape
        batch_size = len(out['pred_logits'])
        target_sizes = torch.FloatTensor([h, w]*batch_size).reshape(batch_size, 2).type_as(out['pred_logits'])
        boxes = self.post_process(out, target_sizes, score_thresh) 
        # we fold the flatten list
        boxes_txn = [boxes[i*b:(i+1)*b] for i in range(t)]
        return boxes_txn

    #@cuda_time
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
        for key, val in loss_dict.items():
            self.log('train_loss_'+key, loss_dict[key])
        self.log('train_loss', loss.item())
        return {'loss': loss}
        
    def inference_step(self, batch, batch_nb):
        with torch.no_grad():
            out, loss, loss_dict = self.get_loss(batch, batch_nb)
        for key in self.weight_dict.keys():
            self.log('val_loss_'+key, loss_dict[key])
        self.log('val_loss', loss.item())

        # accumulate results for meanAP measurement
        preds = self.get_boxes_from_network_output(out, batch['inputs'].shape, score_thresh=0.05)
        dt_dic, gt_dic = self.accumulate_predictions(
            preds,
            batch["labels"],
            batch["video_infos"],
            batch["frame_is_labelled"])
        return {'loss': loss.item(), 'dt': dt_dic, 'gt': gt_dic}

    def validation_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def validation_epoch_end(self, outputs):
        print('\n-- validation epoch end --')
        avg_loss = torch.mean(torch.tensor([elt['loss'] for elt in outputs]))
        self.log('avg_val_loss', avg_loss)
        return self.inference_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.tensor([elt['loss'] for elt in outputs]))
        self.log('avg_test_loss', avg_loss)
        return self.inference_epoch_end(outputs, 'test')

    def inference_epoch_end(self, outputs) :
        avg_loss = torch.mean(torch.tensor([elt['loss'] for elt in outputs]))
        self.log('avg_val_loss', avg_loss)
        self.inference_epoch_end(validation_step_outputs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return opt
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        #return [opt], [sch]

    def inference_epoch_end(self, outputs, mode='val'):
        """
        Runs Metrics

        Args:
            outputs: accumulated outputs
            mode: 'val' or 'test'
        """
        print('==> Start evaluation')
        # merge all dictionaries
        dt_detections = defaultdict(list)
        gt_detections = defaultdict(list)
        for item in outputs:
            for k, v in item['gt'].items():
                gt_detections[k].extend(v)
            for k, v in item['dt'].items():
                dt_detections[k].extend(v)

        dt_dec = list(chain.from_iterable([dt_detections[v] for v in dt_detections]))
        gt_dec = list(chain.from_iterable([gt_detections[v] for v in gt_detections]))
        coco_kpi = coco_evaluation(gt_dec, dt_dec, self.hparams.height, self.hparams.width, self.label_map)

        for k, v in coco_kpi.items():
            print(k, ': ', v)
            self.log('coco_metrics/{}'.format(k), v)

        self.log(mode + '_acc', coco_kpi['mean_ap'])

    def accumulate_predictions(self, preds, targets, video_infos, frame_is_labelled):
        """
        Accumulates prediction to run coco-metrics on the full videos
        """
        dt_detections = defaultdict(list)
        gt_detections = defaultdict(list)
        for t in range(len(targets)):
            for i in range(len(targets[t])):
                gt_boxes = targets[t][i]
                pred = preds[t][i]

                video_info, tbin_start, _ = video_infos[i]

                # skipping when padding or the frame is not labelled
                if video_info.padding or frame_is_labelled[t, i] == False:
                    continue

                name = video_info.path
                assert video_info.start_ts == 0
                ts = tbin_start + t * video_info.delta_t

                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu().numpy()
                if gt_boxes.dtype.isbuiltin:
                    gt_boxes = box_api.box_vectors_to_bboxes(gt_boxes[:,:4], gt_boxes[:,4], ts=ts)


                # Targets are timed
                # Targets timestamped before 0.5s are skipped
                # Labels are in range(1, C) (0 is background) (not in 0, C-1, where 0 would be first class)
                if len(pred['scores']):
                    boxes = pred['boxes'].cpu().data.numpy()
                    labels = pred['labels'].cpu().data.numpy()
                    scores = pred['scores'].cpu().data.numpy()
                    dt_boxes = box_api.box_vectors_to_bboxes(boxes, labels, scores, ts=ts)
                    dt_detections[name].append(dt_boxes)
                else:
                    dt_detections[name].append(np.zeros((0), dtype=box_api.EventBbox))

                if len(gt_boxes):
                    gt_boxes["t"] = ts
                    gt_detections[name].append(gt_boxes)
                else:
                    gt_detections[name].append(np.zeros((0), dtype=box_api.EventBbox))


        return dt_detections, gt_detections
  
    @torch.no_grad()
    def demo_video(self, dataloader, num_batches=10000, show_video=False):
        """
        This runs our detector on several videos of the testing dataset
        """
        hparams = self.hparams
        height, width = hparams.height, hparams.width
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

            batch_size = images.shape[1]
            nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
            ncols = int(np.ceil(batch_size / nrows))
            grid = np.zeros((nrows * hparams.height, ncols * hparams.width, 3), dtype=np.uint8)

            predictions = self.get_boxes(batch, score_thresh=0.1)

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
                    key = cv2.waitKey(5)

                out_video.writeFrame(grid[...,::-1])

                if key == 27:
                    break
                

        if show_video:
            cv2.destroyWindow(window_name)
        out_video.close()
