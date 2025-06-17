import numpy as np

def lossFunction(y, y_pred):
    loss = (1/2) * np.power((y - y_pred),2)
    print(f"Loss function result: {loss}")
    print(f"RMSE: {np.sqrt(loss)}\n")
    return loss


def bce_loss(y, y_pred, eps=1e-15):
    """
    Compute binary cross-entropy loss.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities (floats in [0, 1]).
    eps : float, default=1e-15
        Small value to avoid log(0).

    Returns
    -------
    loss_per_sample : ndarray of shape (n_samples,)
        The BCE loss for each sample.
    mean_loss : float
        The average BCE loss over all samples.
    """
    # ensure numpy arrays
    y = np.array(y)
    y_pred = np.array(y_pred)
    # clip predictions to [eps, 1-eps]
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # compute per-sample loss
    loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    # statistics
    mean_loss = np.mean(loss)
    
    print(f"BCE loss per sample: {loss}")
    print(f"Mean BCE loss: {mean_loss:.6f}\n")
    return loss, mean_loss

    