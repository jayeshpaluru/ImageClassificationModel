import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Pre-processing
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
