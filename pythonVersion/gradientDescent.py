def gradientDescent(weights, bias, lr, grads, db):
    # Update weights
    weights[0] -= lr * grads[0]
    weights[1] -= lr * grads[1]
    # Update bias
    bias       -= lr * db
    print(f"[Update] weights = [{weights[0]:.4f}, {weights[1]:.4f}], bias = {bias:.4f}\n")
    return weights, bias
