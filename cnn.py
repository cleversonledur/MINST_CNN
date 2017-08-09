
# Handwritten Character Recognition


# Description

# Your company is developing a software to digitize handwritten text. 
#Your team has already developed code to extract the image of each one 
#of the characters in a given text. You are given the task of developing 
#a machine learning model capable of reliably translating those images 
#into digital characters. After some research, you find the EMNIST dataset,
#which seems perfect for the task.

#Used bibliography: http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils  import np_utils


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

 	
seed = 123

np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()



num_pixels 	= X_train.shape[1] * X_train.shape[2]

X_train 	= X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test 		= X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


X_train = X_train / 255
X_test 	= X_test / 255

y_train 	= np_utils.to_categorical(y_train)
y_test 		= np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

print num_classes
model = Sequential()

model.add(Dense(
				num_pixels, 
				input_dim 			= num_pixels, 
				kernel_initializer 	= 'normal', 
				activation 			= 'relu'
				)
		 )

model.add(Dense( 
				 num_classes, 
				 kernel_initializer = 'normal', 
				 activation 		= 'softmax'
				)
		 )

model.compile(  
				loss		='categorical_crossentropy', 
				optimizer	='adam', 
				metrics		=['accuracy']
			  )


model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)


test_images = ['2.jpg', '3.jpg', '4.jpg', '5.jpg', '7.jpg', '9.jpg', ]

for img_name in test_images:

	image_test= load_image('test_images/resized/' + img_name)
	image_test = image_test / 255
	image_test = np.asarray(image_test)

	image_test = image_test.reshape(1,28*28)

	y_prob = model.predict(image_test)

	y_classes = y_prob.argmax(axis=-1)
	print y_prob
	print 'Result for ' + img_name + ':' + str(y_classes)

scores = model.evaluate(X_test, y_test, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))




