import tensorflow as tf
import numpy as np
import scipy.io
import time

class VGG19:
  def __init__(self, img, vgg_path = 'imagenet-vgg-verydeep-19.mat'):
    print("loading vgg file...")
    vgg_rawnet = scipy.io.loadmat(vgg_path)
    self.vgg_layers = vgg_rawnet['layers'][0]
    print("vgg file loaded.")
    print("building model...")
    start = time.time()
    self.model = self.build_model(img)
    end = time.time()
    print("build model finished. time: %ds" % (end - start))

  def build_model(self, input_img):
    graph = {}
    _, h, w, d     = input_img.shape
    print('constructing layers...')
    graph['input']   = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

    # print('LAYER GROUP 1')
    # graph['conv1_1'] = self.conv_layer(graph['input'], W=self.get_weights(self.vgg_layers, 0))
    # graph['relu1_1'] = self.relu_layer(graph['conv1_1'], b=self.get_bias(self.vgg_layers, 0))

    # graph['conv1_2'] = self.conv_layer(graph['relu1_1'], W=self.get_weights(self.vgg_layers, 2))
    # graph['relu1_2'] = self.relu_layer(graph['conv1_2'], b=self.get_bias(self.vgg_layers, 2))
    
    # graph['pool1']   = self.pool_layer(graph['relu1_2'])

    # print('LAYER GROUP 2')  
    # graph['conv2_1'] = self.conv_layer(graph['pool1'], W=self.get_weights(self.vgg_layers, 5))
    # graph['relu2_1'] = self.relu_layer(graph['conv2_1'], b=self.get_bias(self.vgg_layers, 5))
    
    # graph['conv2_2'] = self.conv_layer(graph['relu2_1'], W=self.get_weights(self.vgg_layers, 7))
    # graph['relu2_2'] = self.relu_layer(graph['conv2_2'], b=self.get_bias(self.vgg_layers, 7))
    
    # graph['pool2']   = self.pool_layer(graph['relu2_2'])
    
    # print('LAYER GROUP 3')
    # graph['conv3_1'] = self.conv_layer(graph['pool2'], W=self.get_weights(self.vgg_layers, 10))
    # graph['relu3_1'] = self.relu_layer(graph['conv3_1'], b=self.get_bias(self.vgg_layers, 10))

    # graph['conv3_2'] = self.conv_layer(graph['relu3_1'], W=self.get_weights(self.vgg_layers, 12))
    # graph['relu3_2'] = self.relu_layer(graph['conv3_2'], b=self.get_bias(self.vgg_layers, 12))

    # graph['conv3_3'] = self.conv_layer(graph['relu3_2'], W=self.get_weights(self.vgg_layers, 14))
    # graph['relu3_3'] = self.relu_layer(graph['conv3_3'], b=self.get_bias(self.vgg_layers, 14))

    # graph['conv3_4'] = self.conv_layer(graph['relu3_3'], W=self.get_weights(self.vgg_layers, 16))
    # graph['relu3_4'] = self.relu_layer(graph['conv3_4'], b=self.get_bias(self.vgg_layers, 16))

    # graph['pool3']   = self.pool_layer(graph['relu3_4'])

    # print('LAYER GROUP 4')
    # graph['conv4_1'] = self.conv_layer(graph['pool3'], W=self.get_weights(self.vgg_layers, 19))
    # graph['relu4_1'] = self.relu_layer(graph['conv4_1'], b=self.get_bias(self.vgg_layers, 19))

    # graph['conv4_2'] = self.conv_layer(graph['relu4_1'], W=self.get_weights(self.vgg_layers, 21))
    # graph['relu4_2'] = self.relu_layer(graph['conv4_2'], b=self.get_bias(self.vgg_layers, 21))

    # graph['conv4_3'] = self.conv_layer(graph['relu4_2'], W=self.get_weights(self.vgg_layers, 23))
    # graph['relu4_3'] = self.relu_layer(graph['conv4_3'], b=self.get_bias(self.vgg_layers, 23))

    # graph['conv4_4'] = self.conv_layer(graph['relu4_3'], W=self.get_weights(self.vgg_layers, 25))
    # graph['relu4_4'] = self.relu_layer(graph['conv4_4'], b=self.get_bias(self.vgg_layers, 25))

    # graph['pool4']   = self.pool_layer(graph['relu4_4'])

    # print('LAYER GROUP 5')
    # graph['conv5_1'] = self.conv_layer(graph['pool4'], W=self.get_weights(self.vgg_layers, 28))
    # graph['relu5_1'] = self.relu_layer(graph['conv5_1'], b=self.get_bias(self.vgg_layers, 28))

    # graph['conv5_2'] = self.conv_layer(graph['relu5_1'], W=self.get_weights(self.vgg_layers, 30))
    # graph['relu5_2'] = self.relu_layer(graph['conv5_2'], b=self.get_bias(self.vgg_layers, 30))

    # graph['conv5_3'] = self.conv_layer(graph['relu5_2'], W=self.get_weights(self.vgg_layers, 32))
    # graph['relu5_3'] = self.relu_layer(graph['conv5_3'], b=self.get_bias(self.vgg_layers, 32))

    # graph['conv5_4'] = self.conv_layer(graph['relu5_3'], W=self.get_weights(self.vgg_layers, 34))
    # graph['relu5_4'] = self.relu_layer(graph['conv5_4'], b=self.get_bias(self.vgg_layers, 34))

    # graph['pool5']   = self.pool_layer(graph['relu5_4'])

    print('LAYER GROUP 1')
    graph['conv1_1'] = self.conv_layer(graph['input'], 0)
    graph['conv1_2'] = self.conv_layer(graph['conv1_1'], 1)
    graph['pool1']   = self.pool_layer(graph['conv1_2'])

    print('LAYER GROUP 2')  
    graph['conv2_1'] = self.conv_layer(graph['pool1'], 3)
    graph['conv2_2'] = self.conv_layer(graph['conv2_1'], 4)
    graph['pool2']   = self.pool_layer(graph['relu2_2'])
    
    print('LAYER GROUP 3')
    graph['conv3_1'] = self.conv_layer(graph['pool2'], 6)
    graph['conv3_2'] = self.conv_layer(graph['conv3_1'], 7)
    graph['conv3_3'] = self.conv_layer(graph['conv3_2'], 8)
    graph['conv3_4'] = self.conv_layer(graph['conv3_3'], 9)
    graph['pool3']   = self.pool_layer(graph['conv3_4'])

    print('LAYER GROUP 4')
    graph['conv4_1'] = self.conv_layer(graph['pool3'], 11)
    graph['conv4_2'] = self.conv_layer(graph['conv4_1'], 12)
    graph['conv4_3'] = self.conv_layer(graph['conv4_2'], 13)
    graph['conv4_4'] = self.conv_layer(graph['conv4_3'], 14)
    graph['pool4']   = self.pool_layer(graph['conv4_4'])

    print('LAYER GROUP 5')
    graph['conv5_1'] = self.conv_layer(graph['pool4'], 16)
    graph['conv5_2'] = self.conv_layer(graph['conv5_1'], 17)
    graph['conv5_3'] = self.conv_layer(graph['conv5_2'], 18)
    graph['conv5_4'] = self.conv_layer(graph['conv5_3'], 19)
    graph['pool5']   = self.pool_layer(graph['conv5_4'])
    return graph

  def conv_layer(self, layer_input, layer_num):
    weights = self.get_weights(self.vgg_layers, layer_num)
    conv = tf.nn.conv2d(layer_input, weights, strides=[1, 1, 1, 1], padding='SAME')
    bias = self.get_bias(self.vgg_layers, layer_num)
    conv_bias = tf.nn.bias_add(conv, bias)
    conv_relu = tf.nn.relu(conv)
    return conv_relu

  # def relu_layer(self, layer_input, b):
  #   relu = tf.nn.relu(layer_input + b)
  #   return relu

  def pool_layer(self, layer_input):
    pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool

  def get_weights(self, vgg_layers, i):
    weights = self.vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W

  def get_bias(self, vgg_layers, i):
    bias = self.vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b
    
  def get_model(self):
    return self.model