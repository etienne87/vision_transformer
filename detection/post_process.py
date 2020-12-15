import torch
import torch.nn as nn
import torch.nn.functional as F
from detection.box_ops import box_cxcywh_to_xyxy

from torchvision.ops.boxes import batched_nms


def nms(boxes, scores, labels, nms_thresh=0.5):
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    return boxes[keep], scores[keep], labels[keep]


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, score_thresh=0.7, nms_thresh=0.7):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        masks = [s>= score_thresh for s in scores]
        # In theory: just filter
        if nms_thresh == 0:
            results = [{'scores': s[m], 'labels': l[m], 'boxes': b[m]} for s, l, b, m in zip(scores, labels, boxes, masks)]
        else:
            # In practice: do nms...SAD :/ 
            results = []
            for s, l, b, m in zip(scores, labels, boxes, masks):
                s = s[m]
                l = l[m]
                b = b[m]
                b,s,l = nms(b,s,l)
                results.append({'scores': s, 'labels': l, 'boxes': b})

        return results
