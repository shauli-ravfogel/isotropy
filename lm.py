import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb

train = pd.read_csv("train.tsv", sep = "\t")
dev = pd.read_csv("dev.tsv", sep = "\t")
train = train[~train['word'].isnull()]
dev = dev[~dev['word'].isnull()]

with open("w2i.pickle", "rb") as f:
    w2i = pickle.load(f)
wandb.init(project='embds')



class LMDataset(Dataset):

    def __init__(self, df, w2i):
        self.data = df
        self.x = self.data["example"].tolist()
        self.y = self.data["word"].tolist()
        self.w2i = w2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        x = eval(x)
        x = torch.tensor([self.w2i[w] for w in x])
        y = self.w2i[y]
        
        with torch.no_grad():
        
            return x,y
            


class LanguageModel(pl.LightningModule):

    def __init__(self, w2i, in_dim, out_dim, seq_len = 11, tied = False):
        super().__init__()
        self.w2i = w2i
        self.embds_in = nn.Embedding(len(w2i), in_dim)
        self.embds_out = nn.Embedding(len(w2i), out_dim)
        self.seq_len = seq_len
        self.loss_fn = nn.CrossEntropyLoss()
        
        if tied:
            self.embds_out = self.embds_in
        
        self.net = nn.Sequential(*[torch.nn.Linear(in_dim * seq_len, 1024),torch.nn.BatchNorm1d(1024), torch.nn.ELU(),
        torch.nn.Linear(1024, 1024),torch.nn.BatchNorm1d(1024), torch.nn.ELU(), torch.nn.Linear(1024, out_dim),
        torch.nn.BatchNorm1d(out_dim),
        torch.nn.ELU()])

    def forward(self, x):
        mask_ind = self.seq_len // 2
        embds = self.embds_in(x)
        embds = embds.reshape(embds.shape[0], embds.shape[1] * embds.shape[2])
        
        h = self.net(embds)
        logits = h@self.embds_out.weight.T
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        mask_ind = self.seq_len // 2
        embds = self.embds_in(x)
        embds = embds.reshape(embds.shape[0], embds.shape[1] * embds.shape[2])       
        h = self.net(embds)
        logits = h@self.embds_out.weight.T
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds==y).float().mean()
        
        self.log('train_loss', loss)
        if batch_idx % 100 == 0:
            wandb.log({"train_loss": loss})
            wandb.log({"train_acc": acc})
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        #optimizer = torch.optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9)
        return optimizer
        
        
            
train_dataset = LMDataset(train, w2i)
dev_dataset = LMDataset(dev, w2i)

model = LanguageModel(w2i, 128, 512, tied = False)
wandb.watch(model)
trainer = pl.Trainer()
trainer.fit(model, DataLoader(train_dataset, batch_size = 128, num_workers = 8), DataLoader(dev_dataset, batch_size = 64))
