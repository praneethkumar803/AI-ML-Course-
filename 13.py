import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input

X = []
y = []
for i in range(1, 10):
    X.append([i, i+1, i+2, i+3])
    y.append(i+4)

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))


model = Sequential([
    Input(shape=(4, 1)), 
    SimpleRNN(20, activation='relu'),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse')

print("Starting RNN Model Training (9 samples, 500 epochs)...")
model.fit(X, y, epochs=500, verbose=0) 

test_input = np.array([7, 8, 9, 10]).reshape((1, 4, 1))
prediction = model.predict(test_input, verbose=0)

print("\nPrediction Test (Input: [7, 8, 9, 10])")
print("Predicted next number:", prediction[0][0])
print("Rounded prediction:", round(prediction[0][0]))
