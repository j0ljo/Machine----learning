import cv2 
import numpy as np

# Load dataset

data = [
    [200, 200, 200, 1],  # bright
    [50, 50, 50, 0],     # dark
    [180, 170, 160, 1],  # bright
    [30, 40, 50, 0]      # dark
]

# Normalise RGB values
# X = ( xi - x_min(0)/ x_max(255) - x_min )

data = np.array(data)
X = data[:, :3] / 255.0  # normalize RGB to [0, 1]
y = data[:, 3].reshape(-1, 1)  # labels column vector

# Define Sigmoid neuron

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)



# Initialize weights & bias

np.random.seed(42)
weights = np.random.randn(3, 1) * 0.01 # small random numebers
bias = 0.0
lr = 0.1 # Learning rate
epochs = 1000

# Training loop

for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    a = sigmoid(z)

    # Loss (binary cross entropy)
    loss = -np.mean(y * np.log(a + 1e-8) + (1-y) * np.log(1-a+ 1e-8))

    # Backward pass
    dz = a - y
    dw = np.dot(X.T, dz) / len(X)
    db = np.mean(dz)

    # Update weights
    weights -= lr * dw
    bias -= lr * db
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Prediction function
def predict(rgb_values):
    rgb_norm = np.array(rgb_values) / 255.0
    z = np.dot(rgb_norm, weights) + bias
    prob = sigmoid(z)
    return 1 if prob > 0.5 else 0, prob


# Test with an actual image
image = cv2.imread('Images/lowbrightness.jpg')


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
avg_rgb = np.mean(image_rgb, axis=(0, 1))

label, probability = predict(avg_rgb)
print(f"Prediction: {'Bright' if label == 1 else 'Dark'} (prob={probability[0]:.2f})")
