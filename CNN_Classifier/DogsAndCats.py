from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

trainDataGen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

testDataGen = ImageDataGenerator(rescale=1./255)

trainSet = trainDataGen.flow_from_directory(directory='training_set', target_size=(64,64), batch_size=32, class_mode='binary')

testSet = testDataGen.flow_from_directory(directory='test_set', target_size=(64,64), batch_size=32, class_mode='binary')

cClassifier = Sequential()

cClassifier.add(Convolution2D(filters=32, kernel_size=3, input_shape=(64,64,3), activation='relu'))

cClassifier.add(MaxPool2D(pool_size=(2,2)))

cClassifier.add(Convolution2D(32,(3,3),activation='relu'))

cClassifier.add(MaxPool2D(pool_size=(2,2)))

cClassifier.add(Flatten())

cClassifier.add(Dense(output_dim=128, activation='relu'))

cClassifier.add(Dense(output_dim=1, activation='sigmoid'))

cClassifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

cClassifier.fit_generator(trainSet,validation_data=testSet, validation_steps=2000,epochs=5, steps_per_epoch=8000)
