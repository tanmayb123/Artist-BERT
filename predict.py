from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle
import sys

class SelfAttention(nn.Module):
    def __init__(self, input_size, query_key_size, value_size):
        super(SelfAttention, self).__init__()
        self.iq = nn.Linear(input_size, query_key_size)
        self.ik = nn.Linear(input_size, query_key_size)
        self.iv = nn.Linear(input_size, value_size)

    def forward(self, X):
        queries = torch.relu(self.iq(X))
        keys = torch.relu(self.ik(X))
        values = torch.relu(self.iv(X))
        scores = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)), dim=-1)
        return torch.matmul(scores, values)

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

    def forward(self, X, seg_in):
        pos_in = torch.tensor(np.eye(17)).float().type_as(X)
        X = self.embed(X) + self.pos_embed(pos_in) + self.seg_embed(seg_in)
        X = self.a1(X)
        X = self.a2(X)
        X = self.a3(X)
        X = self.a4(X)
        X = self.a5(X)
        X = self.a6(X)
        return self.d1(X)

def get_relevant_output(y_hat, mask_idxs):
    o = []
    for i in range(y_hat.shape[0]):
        for j in mask_idxs[i]:
            o.append(y_hat[i, j])
    return torch.stack(o)

letterset = "abcdefghijklmnopqrstuvwxyz"
charset = letterset + " ME"
maxlen = 17

def segment_in(name):
    segment = 0
    segments = []
    for i in name:
        segments.append(segment)
        if i == " ":
            segment += 1
    segments = np.array(segments)
    y = np.zeros((len(segments), 4))
    y[np.arange(len(segments)), segments] = 1
    return y

def onehot(x):
    y = np.zeros((len(x), len(charset)))
    y[np.arange(len(x)), x] = 1
    return y

def pad(x, f=len(charset)):
    return np.concatenate([x, np.zeros((maxlen - len(x), f))], axis=0)

def process_name(name):
    return pad(onehot([charset.index(x) for x in name]))

def mask_name(name):
    name = list(name)
    oname = name
    idxs = [x for x in list(range(len(name))) if name[x] != " "]
    maskamt = int(len(idxs) * 0.35)
    mask_idxs = random.sample(idxs, maskamt)
    labels = []
    for i in mask_idxs:
        labels.append(letterset.index(name[i]))
        name[i] = "M"
    return "".join(name), [x + 1 for x in mask_idxs], labels

m = ArtistBERT()
print(m)
m.load_state_dict(torch.load("model_70_1.6991746102349232.pt", map_location=torch.device('cpu')))

n = sys.argv[1]
x = process_name(n)
s = pad(segment_in(n), 4)
print(x.shape)
print(s.shape)

x = torch.tensor(x)[None, ...].float()
s = torch.tensor(s)[None, ...].float()

p = m(x, s)[0].detach().numpy()[n.index("M")]
print([(charset[x], p[x]) for x in np.argsort(p)])
