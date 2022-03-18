import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def get_metric(yt, yp):
    #yt = (yt[:, 0]+1.)/2.
    #yp = (yp > 0).astype('int32')[:, 0]
    yp = (yp > 0.5).astype('int32')[:, 0]
    return f1_score(yt, yp), accuracy_score(yt, yp),precision_score(yt,yp), recall_score(yt,yp)

def metric_hist(yt, Lr):
    res = []
    for yp in Lr:
        res.append(list(get_metric(yt, yp)))
    return np.array(res)

def get_best(rsm, acc_T):
    ss = 0
    for i in range(rsm.shape[1]):
        if rsm[0, i] > acc_T:
            ss = i
            break
    rsm = rsm[:, ss:]

    ii = np.argmax(rsm[1, :][::-1])
    ii = rsm.shape[1]-ii-1
    return rsm[2, ii], rsm[3, ii], rsm[4, ii], rsm[5, ii], ii 
  # return rsm[2, ii], rsm[3, ii]

