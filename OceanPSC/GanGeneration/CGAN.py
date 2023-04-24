import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from .models import define_generator,define_discriminator
from .params import *
from .data import denormalization


def create_intermediate_gen(model,i,inter_name):
  int_layer=model.get_layer(inter_name if i==5 else 'intermediate_layer')
  index=model.layers.index(int_layer)
  ninput=layers.Input(shape=(None,None,int_layer.get_output_shape_at(-1)[-1]))
  x=ninput
  for k in range(index+1,len(model.layers)):
    x=model.layers[k](x)
  return tf.keras.Model(ninput,x)


class CGAN:
  def __init__(self,training=True,max_depth=5,inter_d=2):
    inter_name=f'inter_out_{inter_d}'
    self.max_depth=max_depth
    self.gens=define_generator(bruit_dim,max_depth+1)
    self.diss=define_discriminator(max_depth+1)
    self.int_gens=[create_intermediate_gen(m[0],i,inter_name) for m,i in zip(self.gens,range(len(self.gens)))]
    self.int_outs=[tf.keras.Model(inputs=m[0].inputs,
                                         outputs=m[0].get_layer(inter_name if i==5 else 'intermediate_layer').output) for m,i in zip(self.gens,range(len(self.gens)))]

    self.training=training
    if training:
        self._current_depth=0
        self._fade_in=False
        self.setup_checkpoints()
        self.setup_training()
    else:
        self._current_depth=self.max_depth
        self._fade_in=False
        self.setup_checkpoints()
        self.load_latest().expect_partial()


  def setup_checkpoints(self):

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      1e-4 if self.current_depth==self.max_depth else 5e-4,
      decay_steps=20000,
      decay_rate=1,
      staircase=False)
    self.gen_optimizer = tf.keras.optimizers.Adam(lr_schedule,beta_1=0.,beta_2=0.99,epsilon=1e-8)
    self.dis_optimizer = tf.keras.optimizers.Adam(lr_schedule,beta_1=0.,beta_2=0.99,epsilon=1e-8)


    self.checkpoint = tf.train.Checkpoint(generateur_optimizer=self.gen_optimizer,
                                    discriminateur_optimizer=self.dis_optimizer,
                                    generateur=self.gen,
                                    discriminateur=self.dis)
    

  def load_latest(self):
    return self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

  @property
  def gen(self):
    return self.gens[self.current_depth][1 if self.fade_in else 0]

  @property
  def int_gen(self):
    return self.int_gens[self.current_depth]

  @property
  def int_out(self):
    return self.int_outs[self.current_depth]

  @property
  def dis(self):
    return self.diss[self.current_depth][1 if self.fade_in else 0]
  

  @property
  def current_depth(self):
    return self._current_depth
  @current_depth.setter
  def current_depth(self,value):
    assert value <= self.max_depth, f"Current depth can't be above {self.max_depth}"
    self._current_depth=value
    self.setup_checkpoints()
    self.setup_training()

  @property
  def fade_in(self):
    return self._fade_in
  @fade_in.setter
  def fade_in(self,value):
    self._fade_in=value
    self.setup_checkpoints()
    self.setup_training()

  
  def setup_training(self):
    if not self.training:
        return
    gen=self.gen
    dis=self.dis
    def optimize_dis(real_images,labels,mean_stds):
      batch_size=real_images.shape[0]
      bruit = tf.random.normal([batch_size, bruit_dim])
      epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
      with tf.GradientTape() as disc_tape:

        with tf.GradientTape() as gp_tape:
            fake_images = gen([bruit,labels,mean_stds], training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_images, tf.float32) + ((1 - epsilon) * fake_images)
            fake_mixed_pred = dis([fake_image_mixed,labels,mean_stds], training=True)
            
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

        real_pred = dis([real_images,labels,mean_stds], training=True)
        fake_pred = dis([fake_images,labels,mean_stds] , training=True)

        WD=tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
        D_loss = WD + LAMBDA * gradient_penalty + drift * tf.reduce_mean(real_pred ** 2)
        generateur_loss=-tf.reduce_mean(fake_pred)

      gradients_discriminateur = disc_tape.gradient(D_loss, dis.trainable_variables)
      self.dis_optimizer.apply_gradients(zip(gradients_discriminateur, dis.trainable_variables))
      return D_loss,WD,generateur_loss

    def optimize_gen(labels,mean_stds):
      batch_size=labels.shape[0]
      bruit = tf.random.normal([batch_size, bruit_dim])
      with tf.GradientTape() as gen_tape:
        fake_images = gen([bruit,labels,mean_stds], training=True)
        fake_pred = dis([fake_images,labels,mean_stds] , training=True)
        generateur_loss=-tf.reduce_mean(fake_pred)

      gradients_generateur = gen_tape.gradient(generateur_loss, gen.trainable_variables)
      self.gen_optimizer.apply_gradients(zip(gradients_generateur, gen.trainable_variables))
      return generateur_loss

    self.optimize_dis=tf.function(optimize_dis)
    self.optimize_gen=tf.function(optimize_gen)

  def train_step(self,real_images,labels,mean_stds,train_gen=True):
      assert self.training, "Can't call this function while not in training mode"
      D_loss,WD,generateur_loss=self.optimize_dis(real_images,labels,mean_stds)
      
      if train_gen:
        generateur_loss=self.optimize_gen(labels,mean_stds)


      return generateur_loss,D_loss,WD

  def call_on_intermediary(self,intermediary_input):
      if len(intermediary_input.shape)==3:
          intermediary_input=intermediary_input[None,:,:,:]
      result=self.int_gen([intermediary_input],training=False)
      return tf.squeeze(result)

  def __call__(self,label,bruit=None,mstds=None,normalized_mstds=None,denormalize_output=None,output_intermediary=False):
        
    assert (bruit is None) or (len(bruit.shape)<=2 and
                               bruit.shape[-1]==bruit_dim), f"Noise should be of shape (*,bruit_dim) not {bruit.shape}"
    
    if normalized_mstds is None:
        normalized_mstds=self.training or (mstds is None)
    if denormalize_output is None:
        denormalize_output=(not self.training) and (not output_intermediary)

    if type(label) is list:
        label=np.array(label)
    elif type(label) is int or type(label) is np.int32:
        if not bruit is None:
            if len(bruit.shape)==1:
                bruit=bruit[None,:]
            label=np.array(bruit.shape[0]*[label])
        else:
            label=np.array([label])
      
    assert len(label.shape)==1, f"You gave labels of shape {label.shape}. You should give label in 1D array"

    if bruit is None:
        bruit=tf.random.normal([label.shape[0],bruit_dim])
    
    assert bruit.shape[0]==label.shape[0], f"""You gave {label.shape[0]} samples of labels,
    and {bruit.shape[0]} samples of noise (noise of shape {bruit.shape})"""
    
    if mstds is None:
        mstds=np.zeros((label.shape[0],2))
        for k in range(label.shape[0]):
            mstds[k]=vals[label[k]]
    else:
        if type(mstds) is tuple or type(mstds) is list:
            mstds=np.array(mstds)
        if len(mstds.shape)==1:
            mstds=mstds[None,:]
        if mstds.shape[0]==1 and label.shape[0]!=1:
            mstds=np.tile(mstds.T,label.shape[0]).T

    assert len(mstds.shape)==2 and mstds.shape[1]==2 and mstds.shape[0]==label.shape[0],f"""You should give mean 
    and stds in ({label.shape[0]},2) shape not {mstds.shape}"""

    if not normalized_mstds:
        mstds[:,0] = mstds[:,0] / 5000
        mstds[:,1] = mstds[:,1] / 1000
        
    if output_intermediary:
        rslt=self.int_out([bruit,label,mstds],training=False)
    else:
        rslt=self.gen([bruit,label,mstds],training=False)
    
    if denormalize_output:
        #datapreprocessing denormalization outs an image in uint16 interval [0,2**16-1]
        rslt=denormalization(rslt,mstds)
        #now inverting the dataset_creation normalization in order to obtain a DEM in real world elevation interval
        rslt=rslt/(2**16-1)
        rslt=rslt*(maxi-mini)+mini

    return tf.squeeze(rslt)