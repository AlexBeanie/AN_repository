def backPropagation(y, y_pred, x, z):
    # dL/dz for BCE with sigmoid activation
    dz = y_pred - y
    # gradients of each weight
    dw1 = dz * x[0]
    dw2 = dz * x[1]
    # gradient of bias
    db  = dz
    print(f"[Backprop]    dw = [{dw1:.4f}, {dw2:.4f}], db = {db:.4f}\n")
    return [dw1, dw2], db
