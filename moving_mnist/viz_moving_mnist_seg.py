# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located at docs.prophesee.ai/licensing and in the "LICENSE" file accompanying
# this file.
"""
Toy Problem visualization
"""
import sys
import time
import numpy as np
import cv2

from moving_mnist_segmentation import make_moving_mnist



def show_mnist(tbins=10, num_workers=1, batch_size=8, height=128, width=128, max_frames_per_epoch=10000, max_frames_per_video=10):
    dataloader, label_map = make_moving_mnist(tbins, num_workers, batch_size, height=height, width=width, max_frames_per_epoch=max_frames_per_epoch, max_frames_per_video=max_frames_per_video)
    show_batchsize = batch_size 

    start = 0
    nrows = 2 ** ((show_batchsize.bit_length() - 1) // 2)
    ncols = show_batchsize // nrows
    grid = np.zeros((nrows, ncols, height, width, 3), dtype=np.uint8)
    grid_label = np.zeros_like(grid)

    for i, data in enumerate(dataloader):
        batch, targets = data['inputs'], data['labels']
        height, width = batch.shape[-2], batch.shape[-1]
        runtime = time.time() - start
        for t in range(len(batch)):
            grid[...] = 0
            grid_label[...] = 0
            for n in range(batch_size):
                img = (batch[t,n].permute(1, 2, 0).cpu().numpy()*255).copy()
                grid[n//ncols, n%ncols] = img

                label = (targets[t,n].cpu().numpy()).copy()
                labelrgb = cv2.applyColorMap((label*30)%255, cv2.COLORMAP_RAINBOW)
                labelrgb[label==0] = 0
                grid_label[n//ncols, n%ncols] = labelrgb

            im = grid.swapaxes(1, 2).reshape(nrows * height, ncols * width, 3)
            im2 = grid_label.swapaxes(1, 2).reshape(nrows * height, ncols * width, 3)
            im3 = np.concatenate((im,im2), axis=1)

            cv2.imshow('dataset', im3)

            key = cv2.waitKey(0)
            if key == 27:
                break
        
        sys.stdout.write('\rtime: %f' % (runtime))
        sys.stdout.flush()
        start = time.time()



if __name__ == '__main__':
    import fire
    fire.Fire(show_mnist)
   



