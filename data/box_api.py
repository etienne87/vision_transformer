"""
Collections of functions to bounding box 
"""
import numpy as np
import json
import cv2

EventBbox = np.dtype({'names':['t','x','y','w','h','class_id','track_id','class_confidence'], 'formats':['<i8','<f4','<f4','<f4','<f4','<u4','<u4','<f4'], 'offsets':[0,8,12,16,20,24,28,32], 'itemsize':40})

COLORS = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)[:, 0]


def bboxes_to_box_vectors(bbox):
    out = np.zeros((len(bbox), 6), dtype=np.float32)
    bbox = {key:np.float32(bbox[key].copy()) for key in bbox.dtype.names}
    out[:,0] = bbox['x']
    out[:,1] = bbox['y']
    out[:,2] = bbox['x'] + bbox['w']
    out[:,3] = bbox['y'] + bbox['h']
    out[:,4] = bbox['class_id']
    out[:,5] = bbox['track_id']
    return out

def box_vectors_to_bboxes(boxes, labels, scores=None, track_ids=None, ts=0):
    box_events = np.zeros((len(boxes),), dtype=EventBbox)
    if scores is None:
        scores = np.zeros((len(boxes),), dtype=np.float32)
    if track_ids is None:
        track_ids = np.arange(len(boxes), dtype=np.uint32)

    box_events['t'] = ts
    box_events['x'] = boxes[:, 0]
    box_events['y'] = boxes[:, 1]
    box_events['w'] = boxes[:, 2] - boxes[:, 0]
    box_events['h'] = boxes[:, 3] - boxes[:, 1]
    box_events['class_confidence'] = scores
    box_events['class_id'] = labels
    box_events['track_id'] = track_ids
    return box_events


def _choose_color(box_events, color_field, force_color=None):
    if force_color is not None:
        assert len(force_color) == 3
        return np.array([force_color for _ in box_events], dtype=np.uint8)
    else:
        assert np.issubdtype(box_events[color_field].dtype, np.integer), 'color_field {:s} should be integer'.format(
            color_field)
        assert color_field in box_events.dtype.names, 'color_field should be a field of box_events dtype'
        return COLORS[box_events[color_field] * 60 % len(COLORS)]


def draw_box_events(frame, box_events, label_map, force_color=None, draw_score=True, thickness=1,
                color_from="class_id", confidence_field="class_confidence"):
    height, width = frame.shape[:2]
    if len(box_events) == 0:
        return frame

    assert confidence_field in box_events.dtype.names, 'wrong confidence field in dtype: {}'.format(
        confidence_field)

    topleft_x = np.clip(box_events["x"], 0, width - 1).astype('int')
    topleft_y = np.clip(box_events["y"], 0, height - 1).astype('int')
    botright_x = np.clip(box_events["x"] + box_events["w"], 0, width - 1).astype('int')
    botright_y = np.clip(box_events["y"] + box_events["h"], 0, height - 1).astype('int')

    colors = _choose_color(box_events, color_from, force_color=force_color)

    for i, (tlx, tly, brx, bry) in enumerate(zip(topleft_x, topleft_y, botright_x, botright_y)):
        color = colors[i].tolist()
        cv2.rectangle(frame, (tlx, tly), (brx, bry), color, thickness)
        text = label_map[box_events[i]["class_id"]]
        if draw_score:
            text += " {:.2f}".format(box_events[i][confidence_field])
        cv2.putText(frame, text, (int(tlx + 0.05 * (brx - tlx)), int(tly + 0.94 * (-tly + bry))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)

    return frame
