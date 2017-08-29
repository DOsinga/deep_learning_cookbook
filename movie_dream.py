#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy as np
import PIL
import random

import time
from keras.applications import vgg16
from keras.layers import GRU
from keras import backend as K
from itertools import islice
from tqdm import tqdm
import scipy
from scipy.misc import imresize
from scipy import ndimage

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO


def normalize(x):
  # utility function to normalize a tensor by its L2 norm
  return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def visstd(a, s=0.1):
    '''Normalize and clip the image range for visualization'''
    a = (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5
    return np.uint8(np.clip(a, 0, 1) * 255)


def resize_img(img, size):
  img = np.copy(img)
  if K.image_data_format() == 'channels_first':
    factors = (1, 1,
               float(size[0]) / img.shape[2],
               float(size[1]) / img.shape[3])
  else:
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
  return scipy.ndimage.zoom(img, factors, order=1)


def main(frame_dir, movie_width, movie_height, max_frames, steps, zoom):
  zoom += 1.0
  model = vgg16.VGG16(weights='imagenet', include_top=False)
  layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
  print('Model loaded.')
  width = movie_width // 5
  height = movie_height // 5
  if K.image_data_format() == 'channels_first':
      img_data = np.random.uniform(size=(1, 3, height, width,)) + 128.
  else:
      img_data = np.random.uniform(size=(1, height, width, 3)) + 128.

  layer = layer_dict['block5_conv2']
  loss = 0
  all_neurons = list(range(max(x or 0 for x in layer.output_shape)))
  neurons = random.sample(all_neurons, 3)
  frame = 0

  input_img = model.input
  for octave in range(max_frames * 2):
      print(octave)
      if octave % 20 == 0:
          loss = K.variable(0.)
          neurons = neurons[1:]
          neurons.append(random.choice(all_neurons))
          print(neurons)
          for neuron in neurons:
              if K.image_data_format() == 'channels_first':
                  loss += K.mean(layer.output[:, neuron, :, :])
              else:
                  loss += K.mean(layer.output[:, :, :, neuron])

          grads = K.gradients(loss, input_img)[0]
          grads = normalize(grads)
          iterate = K.function([input_img], [loss, grads])

      if octave > 0:
          width = int(width * zoom)
          height = int(height * zoom)
          print(width, height)
          img_data = resize_img(img_data, (height, width))
          if width > movie_width:
              dX = (width - movie_width) // 2
              dY = (height - movie_height) // 2
              if K.image_data_format() == 'channels_first':
                  img_data = img_data[:, :, dY: -dY, dX: -dX]
              else:
                  img_data = img_data[:, dY: -dY, dX: -dX, :]
              width = movie_width
              height = movie_height
              # Add a little noise to create anchor points:
              img_data += np.random.uniform(size=img_data.shape, low=-0.1, high=0.1)


      for i in range(steps):
          loss_value, grads_value = iterate([img_data])
          img_data += grads_value

      if width == movie_width:
        im = PIL.Image.fromarray(visstd(img_data[0]))
        im.save('deep_movie/img_%04d.png' % frame)
        frame += 1
        if frame >= max_frames:
          break




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--frame_dir', type=str, default='deep_movie',
                      help='Where to store the generated frames')
  parser.add_argument('--width', type=int, default=320,
                      help='Movie width')
  parser.add_argument('--height', type=int, default=240,
                      help='Movie height')
  parser.add_argument('--max_frames', type=int, default=2000,
                      help='Number of frames in the movie')
  parser.add_argument('--steps', type=int, default=10,
                      help='Number of steps per frame')
  parser.add_argument('--zoom', type=float, default=0.1,
                      help='Zoom per step')

  args = parser.parse_args()

  main(args.frame_dir, args.width, args.height, args.max_frames, args.steps, args.zoom)
