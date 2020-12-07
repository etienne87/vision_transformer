"""
Toy Problem visualization
"""
import sys
import time
import numpy as np
import cv2

from data import box_api
from moving_mnist_detection import make_moving_mnist



def show_mnist_det(tbins=10, num_workers=1, batch_size=8, height=128, width=128, min_objects=1, max_objects=2, max_frames_per_video=10, max_frames_per_epoch=10000, delay=5):
    dataloader, label_map = make_moving_mnist(tbins, num_workers, batch_size, height, width, max_frames_per_video, max_frames_per_epoch, min_objects=min_objects, max_objects=max_objects, train=False)
    show_batchsize = batch_size 

    start = 0
    nrows = 2 ** ((show_batchsize.bit_length() - 1) // 2)
    ncols = show_batchsize // nrows
    grid = np.zeros((nrows, ncols, height, width, 3), dtype=np.uint8)

    for i, data in enumerate(dataloader):
        batch, targets = data['inputs'], data['labels']
        height, width = batch.shape[-2], batch.shape[-1]
        runtime = time.time() - start
        for t in range(tbins):
            grid[...] = 0
            for n in range(batch_size):
                img = (batch[t,n].permute(1, 2, 0).cpu().numpy()*255).copy()
                boxes = targets[t][n] 
                boxes = box_api.box_vectors_to_bboxes(boxes[:,:4], boxes[:,4])
                img = box_api.draw_box_events(img, boxes, label_map, draw_score=False)
                grid[n//ncols, n%ncols] = img
            im = grid.swapaxes(1, 2).reshape(nrows * height, ncols * width, 3)
            cv2.imshow('dataset', im)
            key = cv2.waitKey(delay)
            if key == 27:
                break
        
        sys.stdout.write('\rtime: %f' % (runtime))
        sys.stdout.flush()
        start = time.time()

def debug_seeds():
    for i in range(5):
        print('ITER#', i)
        show_mnist_det(1, 1, 1, 128, 128, 1, 1, 10, 10, 0)
        print('END')




if __name__ == '__main__':
    import fire
    fire.Fire(debug_seeds)
   


