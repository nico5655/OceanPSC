import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,backend
import tensorflow as tf
from .params import *


class PixelNormalization(layers.Layer):
    def __init__(self, **kwargs):
      super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
      values = inputs**2
      mean_values = backend.mean(values, axis=-1, keepdims=True)
      normalized = inputs / backend.sqrt(mean_values + 1e-8)
      return normalized

    def compute_output_shape(self, input_shape):
      return input_shape


class WeightedSum(layers.Add):
 def __init__(self, alpha=0.0, **kwargs):
  super(WeightedSum, self).__init__(**kwargs)
  self.alpha = backend.variable(alpha, name='ws_alpha')
  self.trainable=False
 
 def _merge_function(self, inputs):
  assert (len(inputs) == 2)
  output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
  return output


class MinibatchStdev(layers.Layer):
  def __init__(self, **kwargs):
    super(MinibatchStdev, self).__init__(**kwargs)

  def call(self, inputs):
    # variance along first axis for each pixel
    mean = backend.mean(inputs, axis=0, keepdims=True)
    variance = backend.mean(backend.square(inputs - mean), axis=0, keepdims=True)
    # add a small value to avoid possible div by zero when calculating mean
    variance += 1e-8
    stdev = backend.sqrt(variance)
    # calculate the mean standard deviation across each pixel coord
    mean_pix = backend.mean(stdev, keepdims=True)
    # scale this up to be the size of one input feature map for each sample
    shape = backend.shape(inputs)
    output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
    # concatenate to create output
    combined = backend.concatenate([inputs, output], axis=-1)
    return combined

  # define the output shape of the layer
  def compute_output_shape(self, input_shape):
    input_shape = list(input_shape)
    input_shape[-1] += 1#add one to the last dimension
    return tuple(input_shape)


class EqualizeLearningRate(layers.Wrapper):
    """
    Reference from WeightNormalization implementation of TF Addons
    EqualizeLearningRate wrapper works for keras CNN and Dense.
    """

    def __init__(self, layer, **kwargs):
        super(EqualizeLearningRate, self).__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)
            self.input_spec=self.layer.input_spec


        if not hasattr(self.layer, 'kernel'):
            raise ValueError('layer has no kernel for weights')


        kernel = self.layer.kernel

        # He constant
        self.fan_in, self.fan_out= self._compute_fans(kernel.shape)
        self.he_constant = tf.Variable(1.0 / np.sqrt(self.fan_in), dtype=tf.float32, trainable=False)

        self.v = kernel
        self.built = True
    
    def call(self, inputs, training=True):
        with tf.name_scope('compute_weights'):
            kernel = tf.identity(self.v * self.he_constant)
            self.layer.kernel = kernel
            update_kernel = tf.identity(self.layer.kernel)

            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
    
    def _compute_fans(self, shape):
        if len(shape) == 2:#dense
            fan_in = shape[0]
            fan_out = shape[1]
        else:#conv
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        return fan_in, fan_out


class LabelEmbeding(layers.Layer):
  def __init__(self,final_size,**kwargs):
    super(LabelEmbeding, self).__init__(**kwargs)
    
    self.embed=layers.Embedding(input_dim=num_classes, output_dim=embedding_dim, input_length=1)
    self.dense=EqualizeLearningRate(layers.Dense(final_size*final_size,kernel_initializer='he_normal'))
    self.reshape=layers.Reshape((final_size,final_size,1))
    self.final_size=final_size

  def call(self, label):
     x=self.embed(label)
     x=self.dense(x)
     x=self.reshape(x)
     return x

  def compute_output_shape(self,input_shape):
    return [input_shape[0],self.final_size,self.final_size,1]


class MeanStdEmbedding(layers.Layer):
  def __init__(self,final_size,**kwargs):
    super(MeanStdEmbedding, self).__init__(**kwargs)
    
    self.dense=EqualizeLearningRate(layers.Dense(final_size*final_size,kernel_initializer='he_normal'))
    self.reshape=layers.Reshape((final_size,final_size,1))
    self.final_size=final_size

  def call(self, x):
     x=self.dense(x)
     x=self.reshape(x)
     return x

  def compute_output_shape(self,input_shape):
    return [input_shape[0],self.final_size,self.final_size,1]

