from sklearn.metrics import mean_squared_error
import numpy as np

def lossFunction(y, y_pred):
    loss = mean_squared_error(y,y_pred) #TODO: Check if compiles
    print(f"Loss function result: {loss}")
    print(f"RMSE: {np.sqrt(loss)}")
    return
    