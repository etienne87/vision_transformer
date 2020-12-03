import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from detection.utils import search_latest_checkpoint



class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, idx=0, num=200):
        super().__init__()
        self.idx = idx
        self.num = num

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield torch.randn(20)


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.enc = nn.Linear(20, 10)
        self.dec = nn.Linear(10, 20)

    def forward(self, x):
        x = self.enc(x)
        x = F.relu(x)
        x = self.dec(x)
        return x

    def training_step(self, batch, batchIdx):
        x = self.forward(batch)
        loss = torch.mean(x)
        return loss

    def validation_step(self, batch, batchIdx):
        x = self.forward(batch)
        loss = torch.mean(x)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        print('val epoch end')
        return {'val_loss': torch.mean(torch.stack([x['val_loss'] for x in outputs]))}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


def train(train_dir, resume=False, max_epochs=10):     
    params = argparse.Namespace(**locals())
    model = Model(params)

    #train_dl = MyDataLoader([i for i in range(10)], Dataset, 4, 2) 
    #val_dl = MyDataLoader([i for i in range(10)], Dataset, 4, 2) 
    train_dl = torch.utils.data.DataLoader(Dataset(0), batch_size=32, num_workers=2)
    val_dl = torch.utils.data.DataLoader(Dataset(0), batch_size=32, num_workers=2)

    if resume:
        ckpt = search_latest_checkpoint(train_dir)
    else:
        ckpt = None

    tmpdir = os.path.join(train_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='toto#{epoch}', save_top_k=-1, period=1) 

    logger = TestTubeLogger(
        save_dir=os.path.join(train_dir, 'logs'),
        version=1)

    import pdb;pdb.set_trace()
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, gpus=1, precision=32, resume_from_checkpoint=ckpt, max_epochs=max_epochs)
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

  
  
if __name__ == "__main__" :
    import fire
    fire.Fire(train)
