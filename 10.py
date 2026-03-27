import numpy as np

x = np.array([[3, 8], [2, 6], [7, 7], [8, 3]])
y = np.array([[50], [45], [90], [95]]) / 100

X = x / np.amax(x, axis=0)

np.random.seed(2)

W1 = np.random.rand(2, 3)  
b1 = np.random.rand(1, 3)  
W2 = np.random.rand(3, 1)  
b2 = np.random.rand(1, 1)  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

lr = 0.1

for epoch in range(10000):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 
    output = sigmoid(z2)

    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_hidden_layer = d_output.dot(W2.T) * sigmoid_derivative(a1)
    W2 += a1.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden_layer) * lr  
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
print("Final Predictions:")
print(output)
