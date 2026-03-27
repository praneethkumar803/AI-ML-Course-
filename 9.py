import numpy as np

x = np.array([[3,8], [2,6], [7,7]])
y = np.array([[50], [45], [90]]) / 100

X = x / np.amax(x, axis=0)

np.random.seed(1)
weights = np.random.rand(2,1)
bias = np.random.rand(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

for epoch in range(1000):
    z = np.dot(X, weights) + bias
    pred = sigmoid(z)

    error = y - pred
    weights += np.dot(X.T, error * pred * (1-pred)) * 0.1
    bias += np.sum(error * pred * (1-pred)) * 0.1

print("Final Predictions:", pred)
