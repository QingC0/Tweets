import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, Lambda, Activation

from keras.engine.topology import Layer
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
import tensorflow as tf

#from mxnet import nd, autograd, gluon
from keras import losses

from keras.callbacks import Callback
class Hinge_AF(Callback):
    def __init__(self, tn, val, tt):
        self.X_train = tn
        self.X_valid = val
        self.X_test = tt
        self.pred = {}
        self.pred['train'] = []
        self.pred['val'] = []
        self.pred['test'] = []
        super(Hinge_AF, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        ypn = self.model.predict(self.X_train, batch_size=20, verbose=0)
        ypv = self.model.predict(self.X_valid, batch_size=20, verbose=0)
        ypt = self.model.predict(self.X_test, batch_size=20, verbose=0)

        self.pred['train'].append(ypn)
        self.pred['val'].append(ypv)
        self.pred['test'].append(ypt)


class Embn0(Layer):
    def __init__(self, w_emb, **kwargs):
        self.w_emb = w_emb
        self.dim_emb = w_emb.shape[1]
        self.W0 = tf.constant(np.zeros((1, self.dim_emb), dtype='float32'))
        kwargs['input_dtype'] = 'int32'
        super(Embn0, self).__init__(**kwargs)

    def build(self, input_shape):
        def my_init(shape, dtype=None):
            return K.variable(self.w_emb[1:], dtype='float32')

        emb_shape = self.w_emb.shape
        self.W = self.add_weight(name='emb_mat', shape=(emb_shape[0]-1, emb_shape[1]),
                                 initializer=my_init, trainable=True)
        super(Embn0, self).build(input_shape)

    def call(self, x, mask=None):
        W = K.concatenate([self.W0, self.W], 0)
        return K.gather(W, x)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.dim_emb,)


def make_model(conf, input_shape, w_emb):

    # loss func
    def hinge(y_true, y_pred):
        return K.mean(K.maximum(conf['hinge_t'] - y_true * y_pred, 0.))

    # negative log likelyhood
    '''
    def logistic(z):
        return 1. / (1. + nd.exp(-z))

    def log_loss(output, y):
        yhat = logistic(output)
        return  - nd.nansum(  y * nd.log(yhat) + (1-y) * nd.log(1-yhat))
    '''

    ## NN layers
    nns = {}
    nns['att0'] = Dense(10, activation='relu', kernel_regularizer=l2(1e-3), name='att1')
    nns['att1'] = Dense(1, kernel_regularizer=l2(1e-3), name='att2')

    ## Computation
    txt = Input(shape=input_shape, dtype='int32')
    ff = Lambda(lambda x: K.reshape(x, (-1, input_shape[1])),
                output_shape=(input_shape[1],))(txt)

    #mask_zero = Lambda(get_non_zero)(ff) 
 
    x = Embn0(w_emb, input_shape=(input_shape[1],), name='emb')(ff)
    x1 = Conv1D(conf['n_cnn'], conf['sz_w'], activation='relu', padding='same', name='c1')(x)
    #x1 = Lambda(a_mask)([x1,mask_zero])

    x2 = Conv1D(conf['n_cnn'], conf['sz_w'], activation='relu', padding='same', name='c21')(x)
    #x2 = Lambda(a_mask)([x2,mask_zero])

    if conf.get('bnorm', None) is not None:
        x2 = Conv1D(conf['n_cnn'], conf['sz_w'], padding='same', name='c22')(x2)
        x2 = BatchNormalization(scale=False)(x2)
        x2 = Activation('relu')(x2)
    else:
        x2 = Conv1D(conf['n_cnn'], conf['sz_w'], activation='relu', padding='same', name='c22')(x2)
    #x2 = Lambda(a_mask)([x2,mask_zero])

    x3 = Conv1D(conf['n_cnn'], conf['sz_w'], activation='relu', padding='same', name='c31')(x)
    #x3 = Lambda(a_mask)([x3,mask_zero])
    x3 = Conv1D(conf['n_cnn'], conf['sz_w'], activation='relu', padding='same', name='c32')(x3)
    #x3 = Lambda(a_mask)([x3,mask_zero])
    if conf.get('bnorm', None) is not None:
        x3 = Conv1D(conf['n_cnn'], conf['sz_w'], padding='same', name='c33')(x3)
        x3 = BatchNormalization(scale=False)(x3)
        x3 = Activation('relu')(x3)
    else:
        x3 = Conv1D(conf['n_cnn'], conf['sz_w'], activation='relu', padding='same', name='c33')(x3)
    #x3 = Lambda(a_mask)([x3,mask_zero])

    tcnn = 3*conf['n_cnn']
    ff = Lambda(lambda x: K.mean(K.concatenate(x), 1),
                output_shape=(tcnn,), name='word_avg')([x1, x2, x3])
    ff = Lambda(lambda x: K.reshape(x, (-1, input_shape[0], tcnn)),
                output_shape=(input_shape[0], tcnn))(ff)
    #ff = Dropout(0.5)(ff)

    attn = nns['att0'](ff)  
    attn = nns['att1'](attn)

    attn = Lambda(lambda x: tf.squeeze(x), output_shape=(input_shape[0],))(attn)
    attn = Lambda(lambda x: tf.tile(tf.expand_dims(tf.nn.softmax(x), -1), [1, 1, tcnn]),
                  output_shape=(input_shape[0], tcnn))(attn)

    if conf.get('attn', None) is not None:
        fft = Lambda(lambda x: K.sum(x[0]*x[1], 1), output_shape=(tcnn,), name='att_avg')([ff, attn])
    else:
        fft = Lambda(lambda x: K.mean(x,1), output_shape=(tcnn,), name='ff_avg')(ff) 
    
    prob = Dense(1, kernel_regularizer=l2(conf['mmC']), name='out')(fft)

    model = Model(inputs=txt, outputs=prob)

    if conf.get('log', None) is not None:
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=conf['lr']))
    else:
        model.compile(loss=hinge, optimizer=Adam(lr=conf['lr']))

    ytrue = tf.placeholder(tf.float32, (None, 1))

    if conf.get('log', None) is not None:
        loss = losses.binary_crossentropy(ytrue, prob)
    else:
        loss = hinge(ytrue, prob)
    
    vars_n = model.trainable_weights[1:]
    vars_e = [model.trainable_weights[0]]

    oz1 = tf.train.AdamOptimizer(learning_rate=2e-5)
    opt1 = oz1.minimize(loss, var_list=vars_n)
    oz2 = tf.train.AdamOptimizer(learning_rate=1e-4)
    opt2 = oz2.minimize(loss, var_list=vars_e)

    def train(sess, data, yt):
        _1, _2, ll = sess.run([opt1, opt2, loss], feed_dict={txt:data, ytrue:yt})
        return ll
    return model, train


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations


def get_non_zero(x): 
    x = K.cast(x, 'int32')
    mask = tf.greater(x, tf.constant(0,'int32'))
    mask = K.cast(mask, 'int32')
    return mask

def a_mask(input):
    x = input[0]
    mask = input[1]
    mask = K.cast(mask,'float32')

    mask = K.repeat(mask, x.shape[2])
    mask = K.permute_dimensions(mask, (0, 2, 1))

   # mask = tf.tile(mask,[1,2])
   # mask = tf.reshape(mask,[mask.shape[0],4,2])   
    return x * mask






 
