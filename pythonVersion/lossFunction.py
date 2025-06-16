import numpy as np

def lossFunction(y, y_pred):
    loss = (1/2) * np.power((y - y_pred),2)
    print(f"Loss function result: {loss}")
    print(f"RMSE: {np.sqrt(loss)}\n")
    return
    