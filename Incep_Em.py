
import sys
import numpy as np
import pickle, gzip
import tensorflow as tf
import keras.backend as K
sc = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0'))
session = tf.Session(config=sc)
session.as_default()
K.set_session(session)

#f = gzip.open('/home/qchen/Data/Ethnicity_CV4.pkl.gz', 'rb')
fn = '/home/qchen/Data/' + sys.argv[1]
f_name = sys.argv[1][:6]
f = gzip.open(fn, 'rb')

train, valid, test, dic, w_emb = pickle.load(f,encoding='latin')
f.close()

X_train, y_train = train
y_train = np.array(y_train)[:, None]
y_train = 2.*(y_train-0.5)

X_valid, y_valid = valid
y_valid = np.array(y_valid)[:, None]
y_valid = 2.*(y_valid-0.5)

X_test, y_test = test
y_test = np.array(y_test)[:, None]
y_test = 2.*(y_test-0.5)


n_tws, maxlen = X_train.shape[1:]
n_words, dim_emb = w_emb.shape

n_cnn = int(sys.argv[2])  #40
sz_w = 3
mmC = float(sys.argv[3])  #0.1


from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn import svm

test_res = {}
val_res = {}

train_pre = []
val_pre = []
test_pre = []

class Hinge_AF(Callback):
    def on_epoch_end(self, epoch, logs={}):
        yp_t = self.model.predict(X_train, batch_size=20, verbose=0)
        yp_t = (yp_t >= 0).astype('int32')[:, 0]
        train_pre.append(yp_t)

        yp = self.model.predict(X_test, batch_size=20, verbose=0)
        yp = (yp >= 0).astype('int32')[:, 0]
        test_pre.append(yp)
        yt = y_test_s
        f1, acc, precision, recall = f1_score(yt, yp), accuracy_score(yt, yp), precision_score(yt,yp), recall_score(yt,yp)

        yp_v = self.model.predict(X_valid, batch_size=20, verbose=0)
        yp_v = (yp_v >= 0).astype('int32')[:, 0]
        val_pre.append(yp_v)
        yt_v = y_valid_s
        f1_v, acc_v, precision_v, recall_v = f1_score(yt_v, yp_v), accuracy_score(yt_v, yp_v), precision_score(yt_v,yp_v), recall_score(yt_v,yp_v)

        print("epoch:", epoch, "test_acc:", acc, "test_f1:", f1, "test_precision:", precision, "test_recall:", recall, "val_acc:", acc_v, "val_f1:", f1_v)
        if epoch == 0:
            test_res['hinge_acc'] = [acc]
            test_res['hinge_f1'] = [f1]
            test_res['hinge_pre'] = [precision]
            test_res['hinge_rec'] = [recall]
            val_res['hinge_acc_v'] = [acc_v]
            val_res['hinge_f1_v'] = [f1_v]
            val_res['hinge_pre_v'] = [precision_v]
            val_res['hinge_rec_v'] = [recall_v]
        else:
            test_res['hinge_acc'].append(acc)
            test_res['hinge_f1'].append(f1)
            test_res['hinge_pre'].append(precision)
            test_res['hinge_rec'].append(recall)
            val_res['hinge_acc_v'].append(acc_v)
            val_res['hinge_f1_v'].append(f1_v)
            val_res['hinge_pre_v'].append(precision_v)
            val_res['hinge_rec_v'].append(recall_v)


bk = Hinge_AF()


from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Conv1D, TimeDistributed, Lambda, Embedding
from keras.optimizers import SGD
import keras.backend as K
from keras.engine.topology import Layer
from keras.regularizers import l2
import tensorflow as tf
from nnlib import c_net, hinge_acc


class Embn0(Layer):
    def __init__(self, w_emb, **kwargs):
        self.dim_emb = w_emb.shape[1]
        self.W0 = tf.constant(np.zeros((1, self.dim_emb), dtype='float32'))
        self.W = K.variable(w_emb[1:])
        self.trainable_weights = [self.W]
        kwargs['input_dtype'] = 'int32'
        super(Embn0, self).__init__(**kwargs)

    def call(self, x, mask=None):
        W = K.concatenate([self.W0, self.W], 0)
        return K.gather(W, x)

    def get_output_shape_for(self, input_shape):
        return input_shape+(self.dim_emb,)


cnns = []
for i in range(6):
    cnns.append(Conv1D(n_cnn, sz_w, activation='relu', border_mode='same'))
emb = Embn0(w_emb, input_shape=(maxlen,))

def inception(x0):
    x = emb(x0)
    x1 = cnns[0](x)

    x2 = cnns[1](x)
    x2 = cnns[2](x2)

    x3 = cnns[3](x)
    x3 = cnns[4](x3)
    x3 = cnns[5](x3)

    return Lambda(lambda x: K.mean(K.concatenate(x), 1),
                output_shape=(3*n_cnn,))([x1, x2, x3])


input_shape = (n_tws, maxlen)
txt = Input(shape=input_shape, dtype='int32')

ff = Lambda(lambda x: K.reshape(x, (-1, maxlen)), output_shape=(maxlen,))(txt)
ff = inception(ff)
ff = Lambda(lambda x: K.mean(K.reshape(x, (-1, n_tws, 3*n_cnn)), 1),
            output_shape=(3*n_cnn,))(ff)
prob = Dense(1, W_regularizer=l2(mmC))(ff)

model = Model(input=txt, output=prob)
model.compile(loss='hinge', optimizer='adam', metrics=[hinge_acc])


print(model.summary())

h = model.fit(X_train, y_train, batch_size=20, nb_epoch=15, callbacks=[bk], verbose=2)

tl = np.array(h.history['hinge_acc'])

vl = np.array(test_res['hinge_acc'])
vl2 = np.array(test_res['hinge_f1'])
vl_pre = np.array(test_res['hinge_pre'])
vl2_rec = np.array(test_res['hinge_rec'])

vl_v = np.array(val_res['hinge_acc_v'])
vl2_v = np.array(val_res['hinge_f1_v'])


print("train_acc:",np.max(tl),"val_acc:", np.max(vl_v), "epoch:", np.argmax(vl_v), "val_f1:", np.max(vl2_v), "epoch:", np.argmax(vl2_v))
    

print("test_acc:",vl[np.argmax(vl_v)],"test_f1:", vl2[np.argmax(vl_v)],"test_pre:",vl_pre[np.argmax(vl_v)], "test_rec:",vl2_rec[np.argmax(vl_v)], "at epoch where val_acc is best'")
print("n_cnn:",n_cnn,"mmC:",mmC)   

import io,os

with io.FileIO("Incep_EM_"+f_name+"_test.out","w") as f:
    for k in range(len(test_pre)):
        for j in test_pre[k]:
            f.write(str(j)+' ')
        f.write('\n')

with io.FileIO("Incep_EM_"+f_name+"_train.out","w") as f:
    for k in range(len(train_pre)):
        for j in train_pre[k]:
            f.write(str(j)+' ')
        f.write('\n')

with io.FileIO("Incep_EM_"+f_name+"_val.out","w") as f:
    for k in range(len(val_pre)):
        for j in val_pre[k]:
            f.write(str(j)+' ')
        f.write('\n')
