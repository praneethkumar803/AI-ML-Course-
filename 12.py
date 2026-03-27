import tensorflow as tf
# Error fixed: Corrected 'karas' to 'keras'
from tensorflow.keras import datasets, layers, models 
import numpy as np 

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0 

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


print("Starting Model Training...")
model.fit(train_images, train_labels, epochs=3, validation_split=0.1)

print("\nEvaluating Model...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)
