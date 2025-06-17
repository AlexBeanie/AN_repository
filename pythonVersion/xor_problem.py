# main.py
import numpy as np

# ---- Activation & derivatives ----
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(a):
    # a = sigmoid(x), so derivative = a * (1 - a)
    return a * (1 - a)

# ---- XOR dataset ----
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([
    [0],
    [1],
    [1],
    [0],
])

# ---- Hyperparameters ----
input_dim  = 2
hidden_dim = 2
output_dim = 1

learning_rate = 0.1
epochs        = 10_000
print_every   = 1_000

# ---- Weight & bias initialization ----
rng = np.random.RandomState(1)
W1 = rng.uniform(-1, 1, (input_dim,  hidden_dim))
b1 = np.zeros((1, hidden_dim))
W2 = rng.uniform(-1, 1, (hidden_dim, output_dim))
b2 = np.zeros((1, output_dim))

# ---- Training loop ----
for epoch in range(1, epochs + 1):
    # Forward pass
    z1 = X.dot(W1) + b1        # (4×2) dot (2×2) → (4×2)
    a1 = sigmoid(z1)           # hidden activations
    z2 = a1.dot(W2) + b2       # (4×2) dot (2×1) → (4×1)
    a2 = sigmoid(z2)           # output predictions

    # Compute MSE loss
    loss = np.mean((y - a2) ** 2)

    # Backpropagation
    # Output layer gradient
    d_loss_a2 = 2 * (a2 - y) / y.shape[0]      # ∂L/∂a2
    d_a2_z2   = sigmoid_derivative(a2)         # ∂a2/∂z2
    dz2       = d_loss_a2 * d_a2_z2            # ∂L/∂z2

    dW2 = a1.T.dot(dz2)                        # ∂L/∂W2
    db2 = np.sum(dz2, axis=0, keepdims=True)   # ∂L/∂b2

    # Hidden layer gradient
    da1_dz1 = sigmoid_derivative(a1)           # ∂a1/∂z1
    dz1     = dz2.dot(W2.T) * da1_dz1          # ∂L/∂z1

    dW1 = X.T.dot(dz1)                         # ∂L/∂W1
    db1 = np.sum(dz1, axis=0, keepdims=True)   # ∂L/∂b1

    # Parameter update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % print_every == 0:
        print(f"Epoch {epoch:5d}   loss = {loss:.6f}")

# ---- Results ----
print("\nFinal outputs after training:")
for x_in, y_true, y_hat in zip(X, y, a2):
    print(f"  input {x_in} → predicted {y_hat[0]:.4f}   (target {y_true[0]})")
