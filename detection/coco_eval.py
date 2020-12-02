"""
Utility to compute coco metrics with a time tolerance
"""
import os
import json
import numpy as np

from six import string_types
from numba import jit

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    print("PyCOCO tool not installed, please install from https://github.com/cocodataset/cocoapi")


def summarize(coco_eval):
    '''
    Computes and displays summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        return stats

    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    if not coco_eval.eval:
        raise Exception('Please run accumulate() first')
    iouType = coco_eval.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    stats = summarize()
    return stats


def coco_evaluation(gts, detections, height, width, labelmap=("car", "pedestrian")):
    """Simple helper function wrapping around COCO's Python API
    
    Args:
        gts: iterable of numpy boxes for the ground truth
        detections: iterable of numpy boxes for the detections
        height (int): frame height 
        width (int): frame width
        labelmap (list): iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results) if len(results) else COCO()

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.stats = summarize(coco_eval)

    stats = {
        "mean_ap": coco_eval.stats[0],
        "mean_ap50": coco_eval.stats[1],
        "mean_ap75": coco_eval.stats[2],
        "mean_ap_small": coco_eval.stats[3],
        "mean_ap_medium": coco_eval.stats[4],
        "mean_ap_big": coco_eval.stats[5],
        "mean_ar": coco_eval.stats[8],
        "mean_ar_small": coco_eval.stats[9],
        "mean_ar_medium": coco_eval.stats[10],
        "mean_ar_big": coco_eval.stats[11]
    }
    return stats


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """Utilitary function producing our data in a COCO usable format

    Args:
        gts: ground-truth boxes
        detections: detection boxes
        categories: class for pycoco api
        height: frame height
        width: frame width
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:
            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1,
                'score': float(bbox['class_confidence']),
                'bbox': [bbox['x'], bbox['y'], bbox['w'], bbox['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}

    return dataset, results
