# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
# from keras.datasets import mnist
from keras.datasets import cifar10
# from keras.models import Sequential, Model
# from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, Activation
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential, Model
from keras.layers import Input, merge, LeakyReLU
# from keras.layers.core import Activation, Lambda
from keras.layers.convolutional import _Conv
from keras.legacy import interfaces
from keras.engine import InputSpec
from keras import backend as K
from keras.engine.topology import Layer

import matplotlib.pyplot as plt
from pylab import savefig

from keras import backend as K
from keras.engine.topology import Layer
import  tensorflow as tf


num_classes = 10
# test 
#epochs = 2
# train epoches
epochs = 200
class SNConv2D(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.Ip = 1
        self.u = self.add_weight(
            name='W_u',
            shape=(1,filters),
            initializer='random_uniform',
            trainable=False
        )

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.W_bar(),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(SNConv2D, self).get_config()
        config.pop('rank')
        return config

    def W_bar(self):
        # Spectrally Normalized Weight
        W_mat = K.permute_dimensions(self.kernel, (3, 2, 0, 1)) # (h, w, i, o) => (o, i, h, w)
        W_mat = K.reshape(W_mat,[K.shape(W_mat)[0], -1]) # (o, i * h * w)

        if not self.Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = _l2normalize(K.dot(_u, W_mat))
            _u = _l2normalize(K.dot(_v, K.transpose(W_mat)))

        sigma = K.sum(K.dot(_u,W_mat)*_v)

        K.update(self.u,K.in_train_phase(_u, self.u))
        return self.kernel / sigma

def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())

# attention class


class Attention(Layer):

    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        # 最后的是连接起来，而不是真正的相加
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        # 这个是函数的嵌套
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['ch'] = 256 # say self. _localization_net  if you store the argument in __init__
        # config[''] =  # say self. _output_size  if you store the argument in __init__
        return config

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # 这个 index 是针对全部的图像的编号，x 是所有图像的array
            inc = random.randrange(1, num_classes)
            # 使用的index ==0 作为 positive pair，然后其他的作为 negetive pair
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)



def create_base_network3(input_shape, embedding_size=128):
    seq = Sequential()
    # CONV => RELU => POOL
    seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    seq.add(Conv2D(50, kernel_size=5, padding="same"))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten => RELU
    seq.add(Flatten())
    # original Dense(500) ,
    seq.add(Dense(embedding_size))

    return seq

def create_base_network_sn(input_shape, embedding_size =128):
    seq =Sequential()
    seq.add(SNConv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    seq.add(LeakyReLU(0.2))

    seq.add(SNConv2D(128,(3,3),strides=(2,2),padding="same"))
    seq.add(LeakyReLU(0.2))

    seq.add(SNConv2D(256,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    seq.add(LeakyReLU(0.2))

    seq.add(SNConv2D(512,(3,3),strides=(2,2),padding="same"))
    seq.add(LeakyReLU(0.2))

    seq.add(Flatten())
    seq.add(Dense(embedding_size))

    return seq

def create_base_network_attention(input_shape, embedding_size =128):
    seq =Sequential()
    seq.add(SNConv2D(64,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    seq.add(LeakyReLU(0.2))

    seq.add(SNConv2D(128,(3,3),strides=(2,2),padding="same"))
    seq.add(LeakyReLU(0.2))

    seq.add(SNConv2D(256,(3,3),strides=(2,2),padding="same",input_shape=input_shape))
    seq.add(LeakyReLU(0.2))
    # 是不是这里少了 maxpooling() 之类的东西，需要降一下维度， 下面这样的
    # seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # add attention layer
    seq.add(Attention(256))

    seq.add(SNConv2D(512,(3,3),strides=(2,2),padding="same"))
    seq.add(LeakyReLU(0.2))

    seq.add(Flatten())
    seq.add(Dense(embedding_size))

    return seq

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)


#embedding_size = 128
embedding_size =1024
#base_network = create_base_network3(input_shape, embedding_size)

# try spectrual normalization
base_network =create_base_network_attention(input_shape, embedding_size)

# 这个才是真正的模型的数据的输入，真正的数据的流入
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

history =model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.subplot(211)
plt.title('model accuracy')
plt.plot(history.history['accuracy'], color='r', label ='train')
plt.plot(history.history['val_accuracy'], color='b', label='val')
plt.legend(loc='best')
#plt.show()
# summarize history for loss
plt.subplot(212)
plt.title('model loss')
plt.plot(history.history['loss'], color='r', label='train')
plt.plot(history.history['val_loss'], color='b', label ='val')
plt.legend(loc='best')
plt.tight_layout()
#plt.show()
plt.savefig('loss-accuracy_attention.png', bbox_inches='tight')

# save and reload model
# so easy hhhhh
model.save('mymodel_attention.h5')
del model

# load images from dir
import os
from scipy.misc import imread
import numpy as np
from keras.models import load_model


def load_images_from_dir(path):
    img_paths =os.listdir(path)
    images =[]
    for img_path in img_paths:
        img =os.path.join(path, img_path)
        image =imread(img)
        images.append(image)
    images =np.array(images)
    images =images.astype('float32')
    images /= 255
    return images



# 就现在而言，这个path1 and path2 是相同的哦
path1 ="/home/jijeng/projects/datasets/cifar10-specific/1/dog"
path2 ="/home/jijeng/projects/datasets/cifar10-specific/1/dog"
n_iters =1
result_path ="result_attention.txt"

# for test
images1= load_images_from_dir(path1)[:10]
images2 =load_images_from_dir(path2)[:10]
print("shape of images1:", images1.shape)
print("shape of images2:", images2.shape)

n1 =len(images1)
n2 =len(images2)
n_pairs =min(n1,n2) -1

model =load_model('mymodel_attention.h5', custom_objects={'contrastive_loss':contrastive_loss,
                                                          'SNConv2D':SNConv2D,
                                                          'Attention':Attention})
print("load pretrained model succeed!!!")
result_txt =open(result_path, 'a')
for i in range(n_iters):
    result_txt.write(str(i)+"\n")
    #seed =random.randint(1,1000000)
    #random.seed(seed)
    isource =np.random.randint(0, n1,n_pairs)
    itarget =np.random.randint(0, n2, n_pairs)
    real=[images1[i,:,:] for i in isource]
    generated =[images2[i,:,:] for i in itarget]

    real_arr =np.array(real).reshape((n_pairs,32,32,3))
    generated_arr =np.array(generated).reshape((n_pairs,32,32,3))
    preds =model.predict([real_arr, generated_arr])
    result =str(np.mean(preds))+'\t'+str(np.std(preds))
    #print('result:', result)
    result_txt.write(result+'\n')
    if i% 10 ==0:
        print("epoch:", i)
        print("preds shape:",preds.shape)
        print("result:", result_txt)

print("success")
result_txt.close()


