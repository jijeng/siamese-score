from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from scipy.misc import imread
import numpy as np
from keras.models import load_model
import sys
import tensorflow as tf
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


def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

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
if __name__ =="__main__":

    path1 =sys.argv[1]
    path2 =sys.argv[2]
    result_path =sys.argv[3]
    sample_num =int(sys.argv[4])
    n_iters =int(sys.argv[5])
    #path1 ="dog"
    #path2 ="dog"
    #n_iters =10
    #result_path ="result.txt"

    # for test
    # shape (nums, 32,32,3)
    images1= load_images_from_dir(path1)
    #images2 =load_images_from_dir(path2)
    print("shape of images1:", images1.shape)
    #print("shape of images2:", images2.shape)

    #n1 =len(images1)
    #n2 =len(images2)
    #n_pairs =min(n1,n2) -1
    model =load_model('mymodel_overfit3.h5', custom_objects={'contrastive_loss':contrastive_loss,
                                                              'SNConv2D':SNConv2D,
                                                              'Attention':Attention})
    print("load pretrained model succeed!!!")
    result_txt =open(result_path,"a")

    n_counts =sample_num
    total_size =5000 # real training data
    #total_size =1000 # test data
    len_img =len(images1)
    for i in range(n_iters):
        # with replacement
        #isource =np.random.randint(0,len_img, n_counts)
        isource =np.random.choice(range(len_img), n_counts, replace=False)
        #import ipdb
        #ipdb.set_trace()

        img_tmp =[images1[i,:,:,:] for i in isource]

        times =int(total_size/n_counts)
        img1s =[]

        for i in range(times):
            img1s += img_tmp
        import copy
        img2s =copy.deepcopy(img1s)

        np.random.shuffle(img1s)
        np.random.shuffle(img2s)
        #import ipdb
        #ipdb.set_trace()
        total_size =len(img1s)

        real_arr =np.array(img1s).reshape((total_size,32,32,3))
        generated_arr =np.array(img2s).reshape((total_size,32,32,3))

        import time
        start_time =time.time()

        preds =model.predict([real_arr, generated_arr])

        # test whether the same as img1 and img2

        elapsed_time =time.time()-start_time
        results =str(np.mean(preds))+"\t"+str(np.std(preds))+"\t"+str(total_size)+"\t"+str(elapsed_time)
        result_txt.write(results+"\n")


    result_txt.close()

    """
    # 这个是处理数据的东西
    def get_images(filename):
        filename = os.path.join(dir_data, filename)
        return scipy.misc.imread(filename)


    images = random.sample(os.listdir(dir_data), sample_num)
    # filenames = glob.glob(os.path.join(dir_data, '*.*'))
    images = [get_images(image) for image in images]
    # copy multiple times and shuffle
    times = int(total_num / sample_num)
    imgs = []
    for i in range(times):
        imgs += images
    # import ipdb
    # ipdb.set_trace()

    np.random.shuffle(imgs)

    print("the length of imgs: ",len(imgs))
    """

    '''
    result_txt =open(result_path, 'a')
    for i in range(n_iters):
        #seed =random.randint(1,1000000)
        #random.seed(seed)
        isource =np.random.randint(0, n1,n_pairs)
        itarget =np.random.randint(0, n2, n_pairs)
        real=[images1[i,:,:] for i in isource]
        generated =[images2[i,:,:] for i in itarget]
        import ipdb
        ipdb.set_trace()

        real_arr =np.array(real).reshape((n_pairs,32,32,3))
        generated_arr =np.array(generated).reshape((n_pairs,32,32,3))
        # preds =model.predict([real_arr, generated_arr])
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
    '''


