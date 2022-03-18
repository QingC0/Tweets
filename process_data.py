import numpy as np
import pickle, gzip

def prepare_data0(fn):
    #fn = '/home/qchen/' + sys.argv[1]
    f = gzip.open(fn, 'rb')
    train, valid, test, dic, w_emb = pickle.load(f, encoding='latin')
    f.close()

    #fan_in, fan_out = w_emb.shape
    #s = np.sqrt(6. / (fan_in+fan_out))
    w_emb = w_emb/np.max(np.abs(w_emb))

    X_train, y_train_s = train
    y_train = np.array(y_train_s)[:, None]
    y_train = 2.*(y_train-0.5)

    X_valid, y_valid_s = valid
    y_valid = np.array(y_valid_s)[:, None]
    y_valid = 2.*(y_valid-0.5)

    X_test, y_test_s = test
    y_test = np.array(y_test_s)[:, None]
    y_test = 2.*(y_test-0.5)

    X0 = np.concatenate((X_train, X_valid), axis=0)
    y0 = np.concatenate((y_train, y_valid), axis=0)

    N = X0.shape[0]
    ixs = np.random.permutation(N)
    ixv = ixs[:200]
    ixn = ixs[200:]
    
    X_train = X0[ixn]
    y_train = y0[ixn]
    X_valid = X0[ixv]
    y_valid = y0[ixv]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test), w_emb


def prepare_data(fne, fnd):

    w_emb = pickle.load(open(fne, 'rb'))
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(open(fnd, 'rb'))


    def clean_normalize(W):
        ds = np.std(W, axis=0)
        ixs = np.nonzero(ds<0.1)[0]
        W = W[:, ixs]
        N = W.shape[0]
        return (W - np.tile(np.mean(W, axis=0), [N, 1]))

    w_emb = clean_normalize(w_emb[1:])
    fan_in, fan_out = w_emb.shape
    s = np.sqrt(6. / (fan_in+fan_out))
    w_emb = w_emb*s
    w_emb = np.vstack([np.zeros((1, w_emb.shape[1])), w_emb])

    return X_train, y_train, X_valid, y_valid, X_test, y_test, w_emb
