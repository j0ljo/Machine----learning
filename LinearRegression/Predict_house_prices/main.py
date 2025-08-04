import matplotlib.pyplot as plt

# Data
X = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]
y = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]

# Initialize
m = 0.0
b = 0.0
learning_rate = 0.0000001
epochs = 1000
n = len(X)

# Run one epoch to capture early curve
y_pred = [m * x + b for x in X]
dm = (-2 / n) * sum((y[i] - y_pred[i]) * X[i] for i in range(n))
db = (-2 / n) * sum((y[i] - y_pred[i]) for i in range(n))
initial_m = m - learning_rate * dm
initial_b = b - learning_rate * db

# Train
for epoch in range(epochs):
    y_pred = [m * x + b for x in X]

    dm = (-2 / n) * sum((y[i] - y_pred[i]) * X[i] for i in range(n))
    db = (-2 / n) * sum((y[i] - y_pred[i]) for i in range(n))

    m = m - learning_rate * dm
    b = b - learning_rate * db

    if epoch % 100 == 0:
        loss = sum((y[i] - y_pred[i])**2 for i in range(n)) / n
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plot
plt.scatter(X, y, color='blue', label='Data Points')

x_line = [min(X), max(X)]

# Plot early line (after 1 step)
y_line_initial = [initial_m * x + initial_b for x in x_line]
plt.plot(x_line, y_line_initial, color='lightgray', linestyle='--', label='Initial Line')

# Plot final line
y_line_final = [m * x + b for x in x_line]
plt.plot(x_line, y_line_final, color='black', linewidth=2, label='Final Line')

plt.xlabel('Square Footage')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression - House Prices')
plt.legend()
plt.grid(True)
plt.show()
