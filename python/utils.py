import numpy as np
import torch
import editdistance


def cer(labels, preds):
    dists = np.array(list(map(lambda x: editdistance.eval(x[0], x[1]), list(zip(labels, preds)))))
    lens = np.array(list(map(len, labels)))
    dists = dists[lens!=0]
    lens = lens[lens!=0]
    return (dists/lens).mean()

def accuracy(labels, preds):
    matches = list(map(lambda x: 1*(x[0]==x[1]), list(zip(labels, preds))))
    return sum(matches)/len(labels)