from flask import Flask, render_template, request, make_response
from functools import wraps, update_wrapper
from PIL import Image
from predictor import Predictor

import numpy as np
import torch
import os

# Initialize the predictor object.
predictor = Predictor()

# Create constants for app root folder and image upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'img')

# Initialize Flask app.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Base endpoint to perform prediction.
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        upload = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
        upload.save(filepath)
        prediction = predictor.predict(request)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', prediction=None, image=None)
		
if __name__ == '__main__':
   app.run()
