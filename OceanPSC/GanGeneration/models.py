
from tensorflow.keras import layers,backend
import tensorflow as tf
import numpy as np
from .params import *
from .custom_layers import PixelNormalization,WeightedSum,MinibatchStdev,EqualizeLearningRate,LabelEmbeding,MeanStdEmbedding


def eq_conv2d(*args,**kwargs):
  name=None
  if 'name' in kwargs:
    name=kwargs['name']
    del kwargs['name']
  if not 'kernel_initializer' in kwargs:
    kwargs['kernel_initializer']='he_normal'
  if not 'bias_initializer' in kwargs:
    kwargs['bias_initializer']='zeros'
  if not 'padding' in kwargs:
    kwargs['padding']='same'
  conv=layers.Conv2D(*args, **kwargs)
  if name is None:
    return EqualizeLearningRate(conv)
  return EqualizeLearningRate(conv,name=name)

def add_discriminator_block(old_model):
  
  # get shape of existing model
  in_shape = list(old_model.input[0].shape)
  if len(in_shape)<3:
    in_shape = list(old_model.input[1].shape)

  old_size=in_shape[-2]
  new_size=2*old_size

  n_input_layers=old_model.layers.index(old_model.get_layer(f'input_process2_{old_size}'))+1
  embed_label_layer=old_model.get_layer('embed_label')
  embed_mean_std_layer=old_model.get_layer('embed_meanstd')

  input_shape = (new_size, new_size, in_shape[-1])

  in_image = layers.Input(shape=input_shape)
  label = layers.Input(shape=(1,), dtype='int32')
  mean_std=layers.Input(shape=(2,),dtype='float32')

  in_label_embedding=embed_label_layer(label)
  in_label_embedding=layers.Resizing(width=new_size,height=new_size,
                                     interpolation='nearest',name=f'resize_lab_{new_size}')(in_label_embedding)
  mean_std_embedding=embed_mean_std_layer(mean_std)
  mean_std_embedding=layers.Resizing(width=new_size,height=new_size,
                                     interpolation='nearest',name=f'resize_meanstd_{new_size}')(mean_std_embedding)

  concatenated = layers.Concatenate(axis=-1,name=f'concat_{new_size}')([in_image, in_label_embedding,mean_std_embedding])

  # define new input processing layer
  d = eq_conv2d(128, 1, name=f'input_process1_{new_size}')(concatenated)
  d = layers.LeakyReLU(alpha=0.2,name=f'input_process2_{new_size}')(d)

  # define new block
  d = eq_conv2d(128, 3)(d)
  d = layers.LeakyReLU(alpha=0.2)(d)

  d = eq_conv2d(128, 3)(d)
  d = layers.LeakyReLU(alpha=0.2)(d)

  d = layers.AveragePooling2D()(d)
  block_new = d

  # skip the input, 1x1 and activation for the old model
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
    
  model1 = tf.keras.Model([in_image,label,mean_std], d)
  
  # downsample the new larger image
  downsample = layers.AveragePooling2D()(in_image)

  # connect old input processing to downsampled new input

  old_label=embed_label_layer(label)
  old_label=old_model.get_layer(f'resize_lab_{old_size}')(old_label)

  old_ms=embed_mean_std_layer(mean_std)
  old_ms=old_model.get_layer(f'resize_meanstd_{old_size}')(old_ms)

  block_old=old_model.get_layer(f'concat_{old_size}')([downsample,old_label,old_ms])

  block_old = old_model.get_layer(f'input_process1_{old_size}')(block_old)
  block_old = old_model.get_layer(f'input_process2_{old_size}')(block_old)

  # fade in output of old model input layer with new input
  d = WeightedSum()([block_old, block_new])

  #add the rest of the old model layers
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
    
  model2 = tf.keras.Model([in_image,label,mean_std], d)
  return [model1, model2]

def define_discriminator(n_blocks, input_shape=[4,4,1]):

  start_size=input_shape[-2]
  model_list = []

  in_image = layers.Input(shape=input_shape)
  label = layers.Input(shape=(1,), dtype='int32')
  mean_std=layers.Input(shape=(2,),dtype='float32')

  embed_label=LabelEmbeding(128,name='embed_label')(label)
  label_embedding=layers.Resizing(width=start_size,height=start_size,
                                     interpolation='nearest',name=f'resize_lab_{start_size}')(embed_label)


  mean_std_embedding=MeanStdEmbedding(128,name='embed_meanstd')(mean_std)
  mean_std_embedding=layers.Resizing(width=start_size,height=start_size,
                                     interpolation='nearest',name=f'resize_meanstd_{start_size}')(mean_std_embedding)

  concatenated = layers.Concatenate(axis=-1,name=f'concat_{start_size}')([in_image, label_embedding,mean_std_embedding])
  # conv 1x1: input processing
  d = eq_conv2d(128, 1,name=f'input_process1_{start_size}')(concatenated)
  d = layers.LeakyReLU(alpha=0.2,name=f'input_process2_{start_size}')(d)

  d = MinibatchStdev()(d)
  d = eq_conv2d(128, 3)(d)
  d = layers.LeakyReLU(alpha=0.2)(d)
  d = eq_conv2d(128, 4)(d)
  d = layers.LeakyReLU(alpha=0.2)(d)

  # dense output layer
  d = layers.Flatten()(d)
  out_class = layers.Dense(1)(d)
  
  model = tf.keras.Model([in_image,label,mean_std], out_class)
  
  model_list.append([model, model])

  # create submodels
  for i in range(1, n_blocks):
    old_model = model_list[i - 1][0]
    models = add_discriminator_block(old_model)
    model_list.append(models)
  return model_list
 


def add_generator_block(old_model,i):

  # get the end of the last block
  block_end = old_model.layers[-2].output

  # upsample, and define new block
  upsampling = layers.UpSampling2D(size=2,interpolation='nearest')(block_end)
  g = eq_conv2d(128, 3)(upsampling)
  g = PixelNormalization()(g)
  g = layers.LeakyReLU(alpha=0.2)(g)

  g = eq_conv2d(128, 3)(g)
  g = PixelNormalization()(g)
  g = layers.LeakyReLU(alpha=0.2,name=f"inter_out_{i}")(g)

  # add new output layer
  out_image = eq_conv2d(1, 1)(g)
  
  model1 = tf.keras.Model(old_model.input, out_image)
  
  out_old = old_model.layers[-1]
  # connect the upsampling to the old output layer
  out_image2 = out_old(upsampling)
  # define new output image as the weighted sum of the old and new models
  merged = WeightedSum()([out_image2, out_image])
  
  model2 = tf.keras.Model(old_model.input, merged)
  return [model1, model2]

def define_generator(latent_dim, n_blocks, in_dim=4):

  model_list = []
  
  in_latent = layers.Input(shape=(latent_dim,))
  label = layers.Input(shape=(1,), dtype='int32')
  mean_std = layers.Input(shape=(2,), dtype='float32')

  label_embedding=LabelEmbeding(in_dim)(label)
  meanstd_embedding=MeanStdEmbedding(in_dim)(mean_std)

  g  = EqualizeLearningRate(layers.Dense(128 * in_dim * in_dim, kernel_initializer='he_normal', bias_initializer='zeros'))(in_latent)
  g = layers.Reshape((in_dim, in_dim, 128))(g)

  g=layers.Concatenate(name='intermediate_layer',axis=-1)([g, label_embedding,meanstd_embedding])
  
  g =eq_conv2d(128, 3)(g)
  g = PixelNormalization()(g)
  g = layers.LeakyReLU(alpha=0.2)(g)
  
  g = eq_conv2d(128, 3)(g)
  g = PixelNormalization()(g)
  g = layers.LeakyReLU(alpha=0.2)(g)
  # conv 1x1, output block
  out_image = eq_conv2d(1, 1)(g)#activation?
  
  model = tf.keras.Model([in_latent,label,mean_std], out_image)
  
  model_list.append([model, model])
    # create submodels
  for i in range(1, n_blocks):
    old_model = model_list[i - 1][0]
    models = add_generator_block(old_model,i)
    # store model
    model_list.append(models)
  return model_list

def update_fadein(models, step, n_steps):
  alpha = step / float(n_steps - 1)
  for model in models:
    for layer in model.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, alpha)