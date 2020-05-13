from keras.datasets import mnist
from matplotlib import pyplot
from keras import utils as np_utils
# load dataset


# summarize loaded dataset
'''print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))'''

# plot first few images
'''for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()'''

	
# load train and test dataset
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

load_dataset()