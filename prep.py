import random
import pickle
from collections import OrderedDict
from tqdm import tqdm
from unidecode import unidecode
import numpy as np
import pandas as pd

letterset = "abcdefghijklmnopqrstuvwxyz"
charset = letterset + " M"
maxlen = 16

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
    padding = np.zeros((maxlen - len(x), f))
    padded = np.concatenate([x, padding], axis=0)
    mask = np.concatenate([np.ones((x.shape[0], 1)), np.zeros((maxlen - x.shape[0], 1))], axis=0)
    return padded, mask

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
    return "".join(name), [x for x in mask_idxs], labels

artists = pd.read_csv("artists.csv").sort_values(by=["listeners_lastfm"], ascending=False)["artist_mb"]
print(artists.head())

artists = list(OrderedDict.fromkeys([unidecode(x).lower() for x in artists if isinstance(x, str)]))
artists = [x for x in artists if len(x) <= maxlen and x.count(" ") <= 3 and x.replace(" ", "").isalpha() and len(x.replace(" ", "")) > 3]
artists = artists[:400000]
segments = [x.count(" ") for x in artists]

print(max(segments))
print(len(artists))

names = []
masks = []
idxs = []
y = []
s = []

for i in tqdm(list(zip(artists, segments))):
    masked_name, mask_idxs, mask_y = mask_name(i[0])
    name, mask = process_name(masked_name)
    names.append(name)
    masks.append(mask)
    idxs.append(mask_idxs)
    y.append(mask_y)
    s.append(pad(segment_in(masked_name), 4)[0])

names = np.array(names)
masks = np.array(masks)
idxs = np.array(idxs)
y = np.array(y)
s = np.array(s)

print(names.shape)
print(masks.shape)
print(idxs.shape)
print(y.shape)
print(s.shape)

pickle.dump([names, masks, idxs, y, s], open("artist_data.pkl", "wb"))
