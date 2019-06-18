"""
Tutorial:
    https://www.tensorflow.org/tutorials/keras/basic_classification#preprocess_the_data
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def plot_image(prediction, real_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction)
    if predicted_label == real_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                            100*np.max(predictions),
                                            class_names[real_label],
                                            color=color))

def plot_value(prediction, real_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(prediction)

    thisplot[predicted_label].set_color('red')
    thisplot[real_label].set_color('blue')

# Load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalise the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create class name array
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Setup layers
layers = [keras.layers.Flatten(input_shape=(28,28)),            # Transforms the format of the images
            keras.layers.Dense(128, activation=tf.nn.relu),     # 
            keras.layers.Dense(10, activation=tf.nn.softmax)]
model = keras.Sequential(layers)

# Compile layers
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train model
number_of_epochs = 5
training_history = model.fit(train_images, train_labels, epochs=number_of_epochs)
training_accuracy = training_history.history['acc'][number_of_epochs-1]
print('Training Accuracy: ', training_accuracy)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_accuracy)

predictions = model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_cols * num_rows
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(predictions[i], test_labels[i], test_images[i])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value(predictions[i], test_labels[i])
plt.show()
