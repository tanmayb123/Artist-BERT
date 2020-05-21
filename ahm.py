from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle

class SelfAttention(nn.Module):
    def __init__(self, input_size, query_key_size, value_size):
        super(SelfAttention, self).__init__()
        self.iq = nn.Linear(input_size, query_key_size)
        self.ik = nn.Linear(input_size, query_key_size)
        self.iv = nn.Linear(input_size, value_size)

    def forward(self, X):
        keys = torch.relu(self.ik(X))
        values = torch.relu(self.iv(X))
        outputs = []
        for ts in range(X.shape[1]):
            kq = []
            query = torch.relu(self.iq(X[:, ts]))
            for i in range(query.shape[0]):
                kq.append(torch.matmul(keys[i], query[i][:, None]))
            kq = torch.stack(kq)
            kq = torch.softmax(kq, 1)
            outputs.append(torch.sum(values * kq, dim=1))
        return torch.transpose(torch.stack(outputs), 0, 1)

class AttentionBlock(nn.Module):
    def __init__(self, size, attention_size):
        super(AttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(size)
        self.a = SelfAttention(size, attention_size, size)
        self.ln2 = nn.LayerNorm(size)
        self.d = nn.Linear(size, size)

    def forward(self, X):
        X = self.ln1(X + self.a(X))
        X = self.ln2(X + self.d(X))
        return X

class ArtistBERT(nn.Module):
    def __init__(self):
        super(ArtistBERT, self).__init__()
        self.embed = nn.Linear(28, 128)
        self.pos_embed = nn.Linear(16, 128)
        self.seg_embed = nn.Linear(4, 128)
        self.a1 = AttentionBlock(128, 128)
        self.a2 = AttentionBlock(128, 128)
        self.a3 = AttentionBlock(128, 128)
        self.a4 = AttentionBlock(128, 128)
        self.d1 = nn.Linear(128, 26)

    def forward(self, X, seg_in):
        pos_in = torch.tensor(np.eye(16)).float().type_as(X)
        X = self.embed(X) + self.pos_embed(pos_in) + self.seg_embed(seg_in)
        X = self.a1(X)
        X = self.a2(X)
        X = self.a3(X)
        X = self.a4(X)
        return self.d1(X)

def get_relevant_output(y_hat, mask_idxs):
    o = []
    for i in range(y_hat.shape[0]):
        for j in mask_idxs[i]:
            o.append(y_hat[i, j])
    return torch.stack(o)

batchsize = 128 * 4

names, mask_idxs, mask_y, seg = pickle.load(open("artist_data.pkl", "rb"))
batches = int(100000 / batchsize)

d = "cuda:0"

m = nn.DataParallel(ArtistBERT()).to(d)
print(m.module)
opt = torch.optim.Adam(m.parameters())

loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    losses = []
    for bidx in tqdm(list(range(batches))):
        opt.zero_grad()
        bstart = bidx * batchsize
        bend = bstart + batchsize
        bnames = torch.tensor(names[bstart:bend]).float().to(d)
        bseg = torch.tensor(seg[bstart:bend]).float().to(d)
        bmask_idxs = mask_idxs[bstart:bend]
        bmask_ys = []
        for i in mask_y[bstart:bend]:
            for j in i:
                bmask_ys.append(j)
        bmask_ys = torch.tensor(bmask_ys).to(d)
        yhat = m(bnames, bseg)
        yhat_mask = get_relevant_output(yhat, bmask_idxs)
        loss = loss_fn(yhat_mask, bmask_ys)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    l = sum(losses) / len(losses)
    print(l)
    torch.save(m.module.state_dict(), "model_{}_{}.pt".format(epoch, l))
