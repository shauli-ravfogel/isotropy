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
import sgd

train = pd.read_csv("train.window_size=3.tsv", sep = "\t")
dev = pd.read_csv("dev.window_size=3.tsv", sep = "\t")
train = train[~train['word'].isnull()]
dev = dev[~dev['word'].isnull()]

with open("w2i.pickle", "rb") as f:
    w2i = pickle.load(f)
#run = wandb.init(project='embds')
#trained_artifact = wandb.Artifact("baseline-model",type ="model",description="baseline model, 256 dim embds, tied")
#trained_artifact.add_dir("model")


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

    def __init__(self, w2i, in_dim, out_dim, seq_len = 7, tied = False):
        super().__init__()
        self.w2i = w2i
        self.embds_in = nn.Embedding(len(w2i), in_dim)
        self.embds_out = nn.Embedding(len(w2i), out_dim)
        self.embds_out = nn.Linear(out_dim, len(w2i))
        self.seq_len = seq_len
        self.loss_fn = nn.CrossEntropyLoss()
        
        if tied:
            self.embds_out.weight = self.embds_in.weight
        
        self.net = nn.Sequential(*[torch.nn.Linear(in_dim * seq_len, 2048), torch.nn.BatchNorm1d(2048), torch.nn.ELU(),
        torch.nn.Linear(2048, out_dim), torch.nn.BatchNorm1d(out_dim), torch.nn.ELU()])

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
        logits = self.embds_out(h)#@self.embds_out.weight.T + self.embds_out.bias
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds==y).float().mean()
        
        self.log('train_loss', loss)
        #if batch_idx % 25 == 0:
        #    wandb.log({"train_loss": loss})
        #    wandb.log({"train_acc": acc})
        
        if batch_idx % 100 == 0:
            #trained_artifact = wandb.Artifact("baseline-model",type ="model",description="baseline model, 256 dim embds, tied")
            torch.save(self.state_dict(), 'model/model_ns.pth')
            #run.log_artifact(trained_artifact)
            
        return loss

    def configure_optimizers(self):
    
        group1 = list(self.embds_out.parameters())
        group2 = list(self.embds_in.parameters()) + list(self.net.parameters())
        
        #optimizer = sgd.SGD({'params': group1, 'lr': 1e-3, "name": "last"}, {'params': group2, 'lr': 1e-3, "name": "rest"})
        optimizer = sgd.SGD([{'params': group1, "name": "last"}, {'params': group2, "name": "rest"}],
        lr = 1e-3, momentum = 0.9)
        #optimizer = torch.optim.Adam(self.parameters(), weight_decay = 1e-5)
        #optimizer = torch.optim.SGD(self.parameters(), lr = 0.5*1e-2, momentum = 0.9)
        return optimizer
        
        
            
train_dataset = LMDataset(train, w2i)
dev_dataset = LMDataset(dev, w2i)

model = LanguageModel(w2i, 512, 768, tied = False)
#wandb.watch(model)
trainer = pl.Trainer(max_epochs=2, min_epochs=2)
trainer.fit(model, DataLoader(train_dataset, batch_size = 64, num_workers = 8, drop_last = True), DataLoader(dev_dataset, batch_size = 64, drop_last = True))
