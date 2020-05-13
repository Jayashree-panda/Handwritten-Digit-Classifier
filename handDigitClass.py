from keras.datasets import mnist
from matplotlib import pyplot
from keras import utils as np_utils

def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = np_utils.to_categorical(trainY)
	testY = np_utils.to_categorical(testY)
	return trainX, trainY, testX, testY
	
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

trainX, trainY, testX, testY = load_dataset()
prep_pixels(trainX, testX)