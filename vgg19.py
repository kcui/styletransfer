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
    graph['conv1_1'] = self.conv_layer(graph['input'], 0)
    graph['conv1_2'] = self.conv_layer(graph['conv1_1'], 2)
    graph['pool1']   = self.pool_layer(graph['conv1_2'])

    graph['conv2_1'] = self.conv_layer(graph['pool1'], 5)
    graph['conv2_2'] = self.conv_layer(graph['conv2_1'], 7)
    graph['pool2']   = self.pool_layer(graph['conv2_2'])
    
    graph['conv3_1'] = self.conv_layer(graph['pool2'], 10)
    graph['conv3_2'] = self.conv_layer(graph['conv3_1'], 12)
    graph['conv3_3'] = self.conv_layer(graph['conv3_2'], 14)
    graph['conv3_4'] = self.conv_layer(graph['conv3_3'], 16)
    graph['pool3']   = self.pool_layer(graph['conv3_4'])

    graph['conv4_1'] = self.conv_layer(graph['pool3'], 19)
    graph['conv4_2'] = self.conv_layer(graph['conv4_1'], 21)
    graph['conv4_3'] = self.conv_layer(graph['conv4_2'], 23)
    graph['conv4_4'] = self.conv_layer(graph['conv4_3'], 25)
    graph['pool4']   = self.pool_layer(graph['conv4_4'])

    graph['conv5_1'] = self.conv_layer(graph['pool4'], 28)
    graph['conv5_2'] = self.conv_layer(graph['conv5_1'], 30)
    graph['conv5_3'] = self.conv_layer(graph['conv5_2'], 32)
    graph['conv5_4'] = self.conv_layer(graph['conv5_3'], 34)
    graph['pool5']   = self.pool_layer(graph['conv5_4'])
    return graph

  def conv_layer(self, layer_input, layer_num):
    weights = self.get_weights(self.vgg_layers, layer_num)
    conv = tf.nn.conv2d(layer_input, weights, strides=[1, 1, 1, 1], padding='SAME')
    bias = self.get_bias(self.vgg_layers, layer_num)
    conv_relu = tf.nn.relu(conv + bias)
    return conv_relu

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