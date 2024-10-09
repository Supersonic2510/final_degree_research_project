import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions and their derivatives
def none(x):
    return x

def none_derivative(x):
    return np.ones_like(x)

def heaviside_step(x):
    return np.heaviside(x, 0.5)

def heaviside_step_derivative(x):
    return np.zeros_like(x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
           (0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) *
           (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)))

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def mish_derivative(x):
    omega = 4 * (x + 1) + 4 * np.exp(2*x) + np.exp(3*x) + np.exp(x) * (4*x + 6)
    delta = 2 * np.exp(x) + np.exp(2*x) + 2
    return np.exp(x) * omega / delta**2

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, 1, alpha * np.exp(x))

# List of activation functions and their derivatives
activation_functions = [
    ("None", none, none_derivative),
    ("Heaviside Step", heaviside_step, heaviside_step_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
    ("ELU", elu, elu_derivative),
    ("GELU", gelu, gelu_derivative),
    ("Swish", swish, swish_derivative),
    ("Mish", mish, mish_derivative),
    ("SELU", selu, selu_derivative)
]

# Define the range of x values
x = np.linspace(-3, 3, 400)

# Plotting the activation functions and their derivatives in separate figures
for name, func, deriv in activation_functions:
    plt.figure(figsize=(12, 6))  # Wider aspect ratio for better visibility

    # Plot the activation function
    plt.subplot(1, 2, 1)
    plt.plot(x, func(x), label=f'{name} Activation', linewidth=3)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # X-axis at y=0
    plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Y-axis at x=0
    plt.title(f'{name} Activation Function')
    plt.grid(True)
    plt.legend()

    # Plot the derivative
    plt.subplot(1, 2, 2)
    plt.plot(x, deriv(x), label=f'{name} Derivative', color='red', linewidth=3)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # X-axis at y=0
    plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Y-axis at x=0
    plt.title(f'{name} Derivative')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
