import forwardPass
import lossFunction
import backPropagation
import gradientDescent

input = [2,3]
weights = [0.1, -0.2]
bias = 0.0
learningRate = 0.1

# True output
y = 1

# First run the forward pass to make a prediction
y_pred, linealRes = forwardPass(weights, input, bias)

# Evaluate the prediction
loss = lossFunction(y, y_pred)

# With this loss function it is possible to calculate the gradients
gradients = backPropagation(y, y_pred, input, linealRes)

weights = gradientDescent(weights, learningRate, gradients)

# Repeat the process from the prediction