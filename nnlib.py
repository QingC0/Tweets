from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM, Conv1D, TimeDistributed, Lambda
from keras.optimizers import SGD
import keras.backend as K
from keras.engine.topology import Layer
from keras.regularizers import l2

import tensorflow as tf

    
def c_net(n, input_dim):
    net = Sequential()
    net.add(Dense(n, activation='tanh', input_shape=(input_dim,)))
    net.add(Dense(1, activation='sigmoid'))
    return net

def hinge_acc(y_true, y_pred):
    p = tf.to_float(tf.greater_equal(y_pred, 0))
    t = (y_true+1.)/2.
    return 1.-K.mean(K.abs(p-t))


def acc_mean(y_true, y_pred):
    p = tf.to_float(tf.greater_equal(y_pred, 0))
    t = (y_true+1.)/2.

    return K.mean(K.equal(y_true, p))

def acc_count(y_true, y_pred):
    p = tf.to_float(tf.greater_equal(y_pred, 0))
    t = (y_true+1.)/2.
    
    # Count true positives, true negatives, false positives and false negatives.
    tp = tf.count_nonzero(p * t)
    tn = tf.count_nonzero((p - 1) * (t - 1))
    fp = tf.count_nonzero(p * (t - 1))
    fn = tf.count_nonzero((p - 1) * t)

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)

    return accuracy

