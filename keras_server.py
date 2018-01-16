#!/usr/bin/env python
import pickle
from flask import Flask, request, redirect, flash, jsonify
from PIL import Image
from keras.applications import InceptionV3
from keras.engine import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np


app = Flask(__name__)
model = None
nbrs = None
image_names = None

@app.route('/', methods=['GET', 'POST'])
def return_size():
  if request.method == 'POST':
    file = request.files['file']
    if file:
      img = Image.open(file)
      target_size = int(max(model.input.shape[1:]))
      img = img.resize((target_size, target_size), Image.ANTIALIAS)
      pre_processed = preprocess_input(np.asarray([image.img_to_array(img)]))
      vec = model.predict(pre_processed)
      distances, indices = nbrs.kneighbors(vec)
      res = [{'distance': dist,
              'image_name': image_names[idx]}
             for dist, idx in zip(distances[0], indices[0])]
      return jsonify(results=res)

  return '''
  <h1>Upload new File</h1>
  <form action="" method=post enctype=multipart/form-data>
    <p><input type=file name=file>
       <input type=submit value=Upload>
  </form>
  '''


if __name__ == '__main__':
  with open('data/image_similarity.pck', 'rb') as fin:
      p = pickle.load(fin)
      image_names = p['image_names']
      nbrs = p['nbrs']
  base_model = InceptionV3(weights='imagenet', include_top=True)
  model = Model(inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output)

  app.run(port=5050, host='0.0.0.0')
