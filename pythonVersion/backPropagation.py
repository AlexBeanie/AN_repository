import numpy as np
import forwardPass as fp

def backPropagation (y, y_pred, input, z):
    #derivative of Loss by prediction
    L_y = y_pred - y
    
    #derivative of prediction by lineal result
    y_z = fp.sigmoid(z) * (1 - fp.sigmoid(z))
    
    #the derivative of the lineal result by the weights are the inputs
    
    #gradient of the first weight
    L_w1 = L_y * y_z * input[0]
    
   #gradient of the second weight
    L_w2 = L_y * y_z * input[1] 
    
    print(f"Weight gradients: [{L_w1}, {L_w2}]\n")
    return L_w1, L_w2