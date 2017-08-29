#!/usr/bin/env python
import argparse
import glob

from gutenberg.acquire import load_etext
from gutenberg.query import get_etexts, get_metadata
from gutenberg.acquire import get_metadata_cache
from gutenberg.acquire.text import UnknownDownloadUriException
from gutenberg.cleanup import strip_headers
from gutenberg._domain_model.exceptions import CacheAlreadyExistsException

from keras.models import Input, Model, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
import keras.callbacks

from keras.optimizers import RMSprop
import random
import numpy as np
import re
import os

def data_generator(all_text, char_to_idx, batch_size, chunk_size):
    X = np.zeros((batch_size, chunk_size, len(char_to_idx)))
    y = np.zeros((batch_size, chunk_size, len(char_to_idx)))
    while True:
        for row in range(batch_size):
            idx = random.randrange(len(all_text) - chunk_size - 1)
            chunk = np.zeros((chunk_size + 1, len(char_to_idx)))
            for i in range(chunk_size + 1):
                chunk[i, char_to_idx[all_text[idx + i]]] = 1
            X[row, :, :] = chunk[:chunk_size]
            y[row, :, :] = chunk[1:]
        yield X, y

def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
    input = Input(shape=(None, num_chars), name='input')
    prev = input
    for i in range(num_layers):
        lstm = LSTM(num_nodes, return_sequences=True)(prev)
        if dropout:
            prev = Dropout(dropout)(lstm)
        else:
            prev = lstm
    dense = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(prev)
    model = Model(inputs=[input], outputs=[dense])
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def find_python(rootdir):
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for fn in filenames:
            if fn.endswith('.py'):
                matches.append(os.path.join(root, fn))

    return matches

def replacer(value):
    value = ''.join(ch for ch in value if ord(ch) < 127)
    if not ' ' in value:
        return value
    if sum(1 for ch in value if ch.isalpha()) > 6:
        return 'MSG'
    return value


def replace_literals(st):
    res = []
    start_text = start_quote = i = 0
    quote = ''
    while i < len(st):
        if quote:
            if st[i: i + len(quote)] == quote:
                quote = ''
                start_text = i
                res.append(replacer(st[start_quote: i]))
        elif st[i] in '"\'':
            quote = st[i]
            if i < len(st) - 2 and st[i + 1] == st[i + 2] == quote:
                quote = 3 * quote
            start_quote = i + len(quote)
            res.append(st[start_text: start_quote])
        if st[i] == '\n' and len(quote) == 1:
            start_text = i
            res.append(quote)
            quote = ''
        if st[i] == '\\':
            i += 1
        i += 1
    return ''.join(res) + st[start_text:]


def get_python_code():
  srcs = find_python(random.__file__.rsplit('/', 1)[0])
  COMMENT_RE = re.compile('#.*')
  python_code = []
  for fn in srcs: # ['/usr/lib/python3.5/difflib.py']: #srcs:
      with open(fn, 'r') as fin:
          src = fin.read()
      src = replace_literals(src)
      src = COMMENT_RE.sub('', src)
      python_code.append(src)

  return '\n\n\n'.join(python_code)



def main(data_source, mode):
  if data_source == 'python':
    training_text = get_python_code()
    model_dir = 'py_models'
  elif data_source == 'shakespeare':
    shakespeare = strip_headers(load_etext(100))
    training_text = shakespeare.split('\nTHE END', 1)[-1]
    model_dir = 'shakespeare_models'
  else:
    raise ValueError('Unknown source: ' + data_source)

  chars = list(sorted(set(training_text)))
  char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
  BATCH_SIZE = 512

  if mode == 'train':
    for chunk_size in (80, 120, 140):
      for num_layers in (2, 3, 4):
        model = char_rnn_model(len(chars), num_layers=num_layers, num_nodes=640, dropout=0)
        early = keras.callbacks.EarlyStopping(monitor='loss',
                                      min_delta=0.01,
                                      patience=2,
                                      verbose=0, mode='auto')
        model.fit_generator(
            data_generator(training_text, char_to_idx, batch_size=BATCH_SIZE, chunk_size=chunk_size),
            epochs=40,
            steps_per_epoch=2 * len(training_text) / (BATCH_SIZE * chunk_size),
            callbacks=[early,],
            verbose=2
        )
        model.save('%s/py_chunk_%d_layers_%d.h5' % (model_dir, chunk_size, num_layers))
  elif mode == 'eval':
    res = []
    for fn in glob.glob(os.path.join(model_dir, '*.h5')):
      bits = fn.split('/')[-1].split('_')
      print(bits)
      chunk_size = int(bits[1])
      model = load_model(fn)
      scores = model.evaluate_generator(
        data_generator(training_text, char_to_idx, batch_size=BATCH_SIZE, chunk_size=chunk_size),
        steps=2 * len(training_text) / (BATCH_SIZE * chunk_size),
      )
      print(fn, scores)
      res.append((fn, scores))

    print('')
    print('sorted results:')
    for fn, scores in sorted(res, key=lambda t:t[1]):
      print(fn, scores)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_source', type=str, default='python',
                      help='Where to get the data, python or shakespeare')
  parser.add_argument('--mode', type=str, default='eval',
                      help='Which mode to run in, eval or train')
  args = parser.parse_args()

  main(args.data_source, args.mode)
