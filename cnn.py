from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

NUM_EPOCHS = 20

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    self.input_shape = (100, 100, 1)
    self.num_classes = 10 

    self.cnn = tf.keras.Sequential([
      tf.keras.layers.Conv2D(8, (19, 19), activation='relu', padding='same'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(3, 3)),
      tf.keras.layers.Conv2D(16, (17, 17), activation='relu', padding='same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3)),
      tf.keras.layers.Conv2D(32, (15, 15), activation='relu', padding='same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(3, 3)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Softmax(),
      tf.keras.layers.Dense(self.num_classes, activation='relu'),])

  def call(self, inputs):
    return self.cnn(inputs)

  def loss(self, logits, labels):
    pass

  def accuracy(self, logits, labels):
    pass

def train(model, train_inputs, train_labels):
  total_loss = 0
      
  with tf.GradientTape() as tape:
    x_hat = model(train_inputs, train_labels)
    batch_loss = loss_function(x_hat)
  gradients = tape.gradient(batch_loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  total_loss += batch_loss    
  
  return total_loss

def test(model, test_inputs, test_labels):
  pass

def main():
  # Compile the model with optimizer and loss function
  model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  # Display model summary
  model.summary()