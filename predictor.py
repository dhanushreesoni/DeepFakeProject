import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import model_from_json

from PIL import Image

size = 224

def preprocess_img(image):
	'''
	This method processes the image into the correct expected shape in the model (224, 224). 
	''' 
	if (image.mode == 'RGB'): 
		# Convert RGB to grayscale. 
		image = image.convert('L')
	image = image.resize((size, size))
	image = np.array(image)
	image = image.reshape(1, size, size, 1)
	return image

class Predictor: 
	def __init__(self):
		# load json and create model
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights("model.h5")
		self.model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

	def predict(self, request):
		'''
		This method reads the file uploaded from the Flask application POST request, 
		and performs a prediction using the loaded model. 
		'''
		f = request.files['image']
		src = Image.open(f)
		image = preprocess_img(src)
		prediction = self.model.predict(image)
		if prediction[0][0] >= prediction[0][1]:
			prediction = "FAKE with a " + str(round(prediction[0][0]*100, 5)) + "% chance"
		else:
			prediction = "REAL with a " + str(round(prediction[0][1]*100, 5)) + "% chance"
		return prediction
	