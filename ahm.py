from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle

"""class SelfAttention(nn.Module):
    def __init__(self, input_size, query_key_size, value_size):
        super(SelfAttention, self).__init__()
        self.iq = nn.Linear(input_size, query_key_size)
        self.ik = nn.Linear(input_size, query_key_size)
        self.iv = nn.Linear(input_size, value_size)

    def forward(self, X, mask):
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
        return torch.transpose(torch.stack(outputs), 0, 1) * mask"""

class SelfAttention(nn.Module):
    def __init__(self, input_size, query_key_size, value_size):
        super(SelfAttention, self).__init__()
        self.iq = nn.Linear(input_size, query_key_size)
        self.ik = nn.Linear(input_size, query_key_size)
        self.iv = nn.Linear(input_size, value_size)

    def forward(self, X, mask):
        queries = torch.relu(self.iq(X))
        keys = torch.relu(self.ik(X))
        values = torch.relu(self.iv(X))
        scores = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)), dim=-1)
        return torch.matmul(scores, values) * mask

class AttentionBlock(nn.Module):
    def __init__(self, size, attention_size):
        super(AttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(size)
        self.a = SelfAttention(size, attention_size, size)
        self.ln2 = nn.LayerNorm(size)
        self.d = nn.Linear(size, size)

    def forward(self, X, mask):
        X = self.ln1(X + self.a(X, mask))
        X = self.ln2(X + self.d(X))
        return X

class ArtistBERT(nn.Module):
    def __init__(self):
        super(ArtistBERT, self).__init__()
        self.embed = nn.Linear(29, 128)
        self.pos_embed = nn.Linear(17, 128)
        self.seg_embed = nn.Linear(4, 128)
        self.a1 = AttentionBlock(128, 128)
        self.a2 = AttentionBlock(128, 128)
        self.a3 = AttentionBlock(128, 128)
        self.a4 = AttentionBlock(128, 128)
        self.a5 = AttentionBlock(128, 128)
        self.a6 = AttentionBlock(128, 128)
        self.d1 = nn.Linear(128, 29)

    def forward(self, X, mask_in, seg_in):
        pos_in = torch.tensor(np.eye(17)).float().type_as(X)
        X = self.embed(X) + self.pos_embed(pos_in) + self.seg_embed(seg_in)
        X = self.a1(X, mask_in)
        X = self.a2(X, mask_in)
        X = self.a3(X, mask_in)
        X = self.a4(X, mask_in)
        X = self.a5(X, mask_in)
        X = self.a6(X, mask_in)
        return self.d1(X)

def get_relevant_output(y_hat, mask_idxs):
    o = []
    for i in range(y_hat.shape[0]):
        for j in mask_idxs[i]:
            o.append(y_hat[i, j])
    return torch.stack(o)

batchsize = 4096 * 4

names, name_masks, loss_masks, y, seg = pickle.load(open("artist_data.pkl", "rb"))
batches = int(len(names) / batchsize)

d = "cuda:0"

m = nn.DataParallel(ArtistBERT()).to(d)
print(m.module)
opt = torch.optim.Adam(m.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss(reduction="none")

for epoch in range(100):
    losses = []

    for bidx in tqdm(list(range(batches))):
        opt.zero_grad()

        bstart = bidx * batchsize
        bend = bstart + batchsize

        bnames = torch.tensor(names[bstart:bend]).float().to(d)
        bname_masks = torch.tensor(name_masks[bstart:bend]).float().to(d)
        bloss_masks = torch.tensor(loss_masks[bstart:bend]).float().to(d)
        by = torch.tensor(y[bstart:bend]).long().to(d).reshape(-1)
        bseg = torch.tensor(seg[bstart:bend]).float().to(d)

        yhat = m(bnames, bname_masks, bseg).reshape(-1, 29)
        loss = torch.sum(loss_fn(yhat, by).reshape(-1, 17) * bloss_masks) / torch.sum(bloss_masks.reshape(-1))

        loss.backward()
        opt.step()

        losses.append(loss.item())

    l = sum(losses) / len(losses)
    print(l)
    torch.save(m.module.state_dict(), "model_{}_{}.pt".format(epoch, l))
