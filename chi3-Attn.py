import sys
expt = sys.argv[1][:3]
from config import *
conf = conf[expt]
import numpy as np


import pickle, gzip
fn = '/home/qchen/Tweets/Data/' + sys.argv[1]
f_name = sys.argv[1][:7]
f = gzip.open(fn, 'rb')

train, valid, test, dic, w_emb = pickle.load(f, encoding='latin')
f.close()

X_train, y_train = train
y_train = np.array(y_train)[:, None]

if conf.get('log', None) is None:
    y_train = 2.*(y_train-0.5)

X_valid, y_valid = valid
y_valid = np.array(y_valid)[:, None]
if conf.get('log', None) is None:
    y_valid = 2.*(y_valid-0.5)

X_test, y_test = test
y_test = np.array(y_test)[:, None]
if conf.get('log', None) is None:
    y_test = 2.*(y_test-0.5)

import random as rn
import os
#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(42)
#rn.seed(12345)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


import tensorflow as tf
import keras.backend as K

sc = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0'))   #,intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.Session(config=sc)
session.as_default()
K.set_session(session)


from model import Hinge_AF, make_model, get_activations
bk = Hinge_AF(X_train, X_valid, X_test)
model, train = make_model(conf, X_train.shape[1:], w_emb)

#initial = model.get_weights()
#print('initial:\n',initial)

#model.save_weights(f_name+'_weights.h5')
#model.load_weights(f_name+'_weights.h5')

#print(model.summary())

h = model.fit(X_train, y_train, batch_size=20, epochs=conf['n_ep'], callbacks=[bk], verbose=0)



from process_results import metric_hist, get_best
res_tn = metric_hist(y_train, bk.pred['train'])
res_val = metric_hist(y_valid, bk.pred['val'])
res_tt = metric_hist(y_test, bk.pred['test'])

for i in range(len(res_tn)):
    print('==============================================================================')
    print('trai_acc',res_tn[i][1],'trai_f1',res_tn[i][0],'trai_pre',res_tn[i][2],'trai_rec',res_tn[i][3])
    print('test_acc',res_tt[i][1],'test_f1',res_tt[i][0],'test_pre',res_tt[i][2],'test_rec',res_tt[i][3])
    print('vali_acc',res_val[i][1],'vali_f1',res_val[i][0],'vali_pre',res_val[i][2],'vali_rec',res_val[i][3])

rm = np.stack([res_tn[:, 1], res_val[:, 1], res_tt[:, 1], res_tt[:, 0], res_tt[:, 2], res_tt[:, 3]], axis=0)
#rm = np.stack([res_tn[:, 1], res_val[:, 1], res_tt[:, 1], res_tt[:, 0]], axis=0)
acc, f1, pre, recall, ep = get_best(rm, conf['nacc_T'])

print('acc, f1, pre, recall at epoch:',acc,f1,pre,recall,ep)

res = '%1.3f, %1.3f, %1.3f, %1.3f, %d \n' % (acc, f1, pre, recall, ep)
f = open('new_results_%s.csv' % expt, 'a+')
f.write(res)
f.close()

import csv
with open('Train_results_%s.csv' % f_name, 'w+') as my_csv:
    csvWriter = csv.writer(my_csv) #,delimiter=','
    csvWriter.writerows(bk.pred['train'])

with open('Test_results_%s.csv' % f_name, 'w+') as my_csv1:
    csvWriter1 = csv.writer(my_csv1) #,delimiter=','
    csvWriter1.writerows(bk.pred['test'])

with open('Val_results_%s.csv' % f_name, 'w+') as my_csv2:
    csvWriter2 = csv.writer(my_csv2) #,delimiter=','
    csvWriter2.writerows(bk.pred['val'])

session.close()
