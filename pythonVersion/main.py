# main.py
import forwardPass as fp
import lossFunction as lf
import backPropagation as bp
import gradientDescent as gd

# --- Data & hyperparameters ---
x             = [2.0, 3.0]   # feature vector
y_true        = 1.0          # label
weights       = [0.1, -0.2]  # initial weights
bias          = 0.0          # initial bias
learning_rate = 0.1
epochs        = 100

# --- Training loop ---
for epoch in range(1, epochs+1):
    print(f"=== Epoch {epoch} ===")
    # 1) Forward pass
    y_pred, z = fp.forwardPass(weights, x, bias)
    # 2) Loss
    _, loss = lf.bce_loss(y_true, y_pred)
    # 3) Backprop
    grads, db = bp.backPropagation(y_true, y_pred, x, z)
    # 4) Update params
    weights, bias = gd.gradientDescent(weights, bias, learning_rate, grads, db)

print(f"Training complete.\nFinal weights = {weights}, final bias = {bias:.4f}")
