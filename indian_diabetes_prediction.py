# Neural network with keras

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#Load dataset
data = loadtxt("C:/Users/Omomule Taiwo G/Desktop/deep_learning/pima-indians-diabetes.csv", delimiter=',')


# Split data into input (X) and output (y) variables
X = data[:,0:8]
y = data[:,8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the keral model
model = Sequential()  # models in Keras are defined as sequence of layers

model.add(Dense(12, input_dim=8, activation='relu')) # The "Dense" class defines fully-connected entwork structures of three layers. This
# is the first layer that accepts 12 hidden nodes or neurons, the number of features in the dataset=8, and relu activation function.

model.add(Dense(8, activation='relu')) # This is the second layer that accepts 8 hidden nodes and relu activation function
model.add(Dense(1, activation='sigmoid')) # This is the third layer, which is the output layer. It accepts one hidden node and the
#sigmoid activation function to ensure the netowrk output is between 0 and 1 and easy to map to either a probability of class 1, or
# snap to a hard classification of either class with a default threshold of 0.5

# Compile the Keras Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # The compile function uses efficient numerical
# libraries, called backend such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network for
# training and making predictions to run on your hardware, such as CPU or GPU or even distributed.
# The compile function accepts the loss function, here binary_crossentropy is used for binary classification problems. Another one
# can be used for multi-class classification problems. The compile function also accepts the weight optimization algorithms, here we
# used "adam". The performance metric to evaluate the keras model is also required. Because we are dealing with a classification
# problem, 'accuracy' is selected.

# Fit Keras Model

model.fit(X_train, y_train, epochs=150, batch_size=10)
# We can train or fit our model on our loaded data by calling the fit() function on the model.
#
# Training occurs over epochs and each epoch is split into batches.
#
# Epoch: One pass through all of the rows in the training dataset.
# Batch: One or more samples considered by the model within an epoch before weights are updated.
# One epoch is comprised of one or more batches, based on the chosen batch size and the model is
# fit for many epochs. For more on the difference between epochs and batches, see the post:

# For this problem, we will run for a small number of epochs (150) and use a relatively small batch size of 10.
#
# These configurations can be chosen experimentally by trial and error. We want to train the model enough
# so that it learns a good (or good enough) mapping of rows of input data to the output classification.
# The model will always have some error, but the amount of error will level out after some point for a
# given model configuration. This is called model convergence. This is where the work happens on your CPU or GPU.

# Evaluate Keras Model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
# We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.
# You can evaluate your model on your training dataset using the evaluate() function
# on your model and pass it the same input and output used to train the model.

# NOTE:  ideally, you could separate your data into train and test datasets for training and evaluation of your model

# The evaluate() will generate a prediction for each input and output pair and collect scores, including the average loss
# and any metrics you have configured, such as accuracy.
#
# The evaluate() function will return a list with two values. The first will be the loss of the model on the dataset
# and the second will be the accuracy of the model on the dataset.
# We are only interested in reporting the accuracy, so we will ignore the loss value.
