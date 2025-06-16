def gradientDescent (weights, lr, gradients):
    # Calculate the new weight of the first input
    weights[0] = weights[0] - lr * gradients[0]
    
    # Calculate the new weight of the second input
    weights[1] = weights[1] - lr * gradients[1]
    
    print(f"These are the updated weights for the neuron: [{weights[0]}, {weights[1]}]\n")
    
    return weights