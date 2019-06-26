import keras

vgg16Model = keras.applications.vgg16.VGG16()

vgg16Model.summary()

from keras.preprocessing.image import load_image,img_to_array
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

image = load_image('books.jpg',target_size=(224,224))

image = img_to_array(image)

image = image.reshape((1,image.shape[0],image.shape[1],image.shape[0]))

image = preprocess_input(image)

prediction = vgg16Model.predict(image)
decode_predictions(prediction)