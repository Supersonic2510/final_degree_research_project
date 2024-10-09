import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data (a simple quadratic function with some noise)
np.random.seed(42)
X = np.sort(np.random.rand(20, 1) * 10, axis=0)
y = 0.5 * X**2 + X + 2 + np.random.randn(20, 1) * 2

# Create an overfitted model (high-degree polynomial)
poly_features = PolynomialFeatures(degree=10)
X_poly = poly_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training Data', s=50)

# Plot the true function (for reference)
X_fit = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = 0.5 * X_fit**2 + X_fit + 2
plt.plot(X_fit, y_true, color='green', linestyle='--', label='True Function', linewidth=2)

# Plot the overfitted model
X_fit_poly = poly_features.transform(X_fit)
y_fit_poly = model.predict(X_fit_poly)
plt.plot(X_fit, y_fit_poly, color='red', linestyle='-', label='Overfitted Model', linewidth=2)

# Customize the plot
plt.title('Overfitting in Machine Learning', fontsize=14)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid(True)

plt.show()
