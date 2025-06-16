import numpy as np

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def forwardPass (w, x, b):
    # Calculate the lineal value
    z = w[0]*x[0] + w[1]*x[1] + b
    print(f"This is the linear result of the forward pass: {z}")
    
    # Add nonlineality with activation function
    sig = sigmoid(z)
    print(f"This is the result after using an activation function: {sig}\n")
    return sig, z