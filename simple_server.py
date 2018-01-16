#!/usr/bin/env python
from flask import Flask, request, redirect, flash, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def return_size():
  if request.method == 'POST':
    file = request.files['file']
    if file:
      image = Image.open(file)
      width, height = image.size
      return jsonify(results={'width': width, 'height': height})


  return '''
  <h1>Upload new File</h1>
  <form action="" method=post enctype=multipart/form-data>
    <p><input type=file name=file>
       <input type=submit value=Upload>
  </form>
  '''


if __name__ == '__main__':
  app.run(port=5050, host='0.0.0.0')
