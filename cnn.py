import preprocess
import tensorflow as tf

NUM_EPOCHS = 20

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    self.input_shape = (100, 100, 3)
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

  def accuracy(self, logits, labels):
    pass

def train(model, train_inputs, train_labels):
  loss = 0
  
  for epoch in range(NUM_EPOCHS):
    with tf.GradientTape() as tape:
      x_hat = model(train_inputs, train_labels)
      loss += model.loss(x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss/ NUM_EPOCHS

def test(model, test_inputs, test_labels):
  x_hat = model(test_inputs, test_labels)

  return model.loss(x_hat)

def main():
  
  model = Model()

  x_train, y_train, x_test, y_test = preprocess.load_data('FOLDER')

  train(model, x_train, y_train)

  test(model, x_test, y_test)