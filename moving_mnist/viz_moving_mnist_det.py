"""
Toy Problem visualization
"""
from __future__ import absolute_import
import sys
import time
import numpy as np
import random
import cv2

from utils import box_api
from moving_mnist.moving_mnist_detection import MovingMNISTDetDataset


def draw_mask(mask_set, class_ids):
    mask_set = mask_set.numpy()
    class_ids = class_ids.numpy()
    n,h,w = mask_set.shape
    assert len(class_ids) == n
    mask = np.zeros((h,w,3), dtype=np.float32)
    for i, class_id in enumerate(class_ids):
        label = mask_set[i] * class_id
        label = np.uint8(label)
        labelrgb = cv2.applyColorMap((label*30)%255, cv2.COLORMAP_RAINBOW)
        labelrgb[label==0] = 0
        mask += labelrgb / n
    return mask.astype(np.uint8)




def show_mnist_det(tbins=10, num_workers=2, batch_size=8, height=128, width=128, min_objects=1, max_objects=2, max_frames_per_video=10, max_frames_per_epoch=10000, delay=5, epochs=1):
    show_batchsize = batch_size
    dataloader = MovingMNISTDetDataset(
        tbins,
        num_workers,
        batch_size,
        height,
        width,
        max_frames_per_video,
        10000,
        True)

    label_map = [str(i) for i in range(10)]

    start = 0
    nrows = 2 ** ((show_batchsize.bit_length() - 1) // 2)
    ncols = show_batchsize // nrows
    grid = np.zeros((nrows, ncols, height, width, 3), dtype=np.uint8)
    grid_masks = np.zeros((nrows, ncols, height, width, 3), dtype=np.uint8)

    for e in range(epochs):
        print('EPOCH#', e)
        images = []
        random.seed(e)
        np.random.seed(e)
        for i, data in enumerate(dataloader):
            batch, targets = data['inputs'], data['labels']
            height, width = batch.shape[-2], batch.shape[-1]
            runtime = time.time() - start
            for t in range(tbins):
                grid[...] = 0
                grid_masks[...] = 0
                for n in range(batch_size):
                    img = (batch[t,n].permute(1, 2, 0).cpu().numpy()*255).copy()
                    boxes = targets[t][n]
                    boxes = box_api.box_vectors_to_bboxes(boxes[:,:4], boxes[:,4])
                    img = box_api.draw_box_events(img, boxes, label_map, draw_score=False)
                    grid[n//ncols, n%ncols] = img

                    #mask
                    mask_set = data['masks'][t][n]
                    class_ids = targets[t][n][:,4]
                    mask = draw_mask(mask_set, class_ids)
                    grid_masks[n//ncols, n%ncols] = mask

                im = grid.swapaxes(1, 2).reshape(nrows * height, ncols * width, 3)
                masks = grid_masks.swapaxes(1, 2).reshape(nrows*height, ncols*width, 3)
                cat = np.concatenate((im, masks), axis=1)
                cv2.imshow('dataset', cat)
                key = cv2.waitKey(delay)
                if key == 27:
                    break


            sys.stdout.write('\rtime: %f' % (runtime))
            sys.stdout.flush()
            start = time.time()

        print('END EPOCH')

def debug_seeds():
    for i in range(5):
        print('ITER#', i)
        show_mnist_det(1, 1, 1, 128, 128, 1, 1, 1, 3, 0, 4)
        print('END ITER')




if __name__ == '__main__':
    import fire
    fire.Fire(show_mnist_det)



