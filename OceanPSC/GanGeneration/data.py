import tensorflow as tf
import numpy as np
from .params import *
import pathlib


def normalization(image,label):
    image=tf.cast(image,tf.float32)
    m=tf.reduce_mean(image)
    st=tf.math.reduce_std(image)
    image=image-m
    image=image/st
    m=m/(2**16-1)
    m=m*(maxi-mini)+mini
    st=st/(2**16-1)
    st=st*(maxi-mini)
    return image,label,tf.stack([m/5000,st/1000])

def denormalization(image,mean_stds):
    st=mean_stds[:,1][:,None,None,None]*1000
    m=mean_stds[:,0][:,None,None,None]*5000
    st=st/(maxi-mini)
    st=st*(2**16-1)
    m=(m-mini)/(maxi-mini)
    m=m*(2**16-1)
    image=image*st
    image=image+m
    return image

def load_dataset():  
    data_dir=pathlib.Path('train')
    def get_label(file_path):
      parts = tf.strings.split(file_path, os.path.sep)
      # The second to last is the class directory
      one_hot = parts[-2] == class_names
      return tf.argmax(one_hot)

    def decode_img(img):
      # Convert the compressed string to a uint16 tensor
      img = tf.io.decode_png(img, dtype=tf.uint16)
      return img

    def process_path(file_path):
      label = get_label(file_path)
      # Load the raw data from the file as a string
      img = tf.io.read_file(file_path)
      img = decode_img(img)
      return img, label



    list_ds = tf.data.Dataset.list_files(str(data_dir/'*'/'*'), shuffle=True)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))


    train_dataset=list_ds.map(process_path,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset=train_dataset.map(normalization,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset,class_names