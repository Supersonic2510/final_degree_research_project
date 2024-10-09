import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline


# Helper function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Polynomial Logistic Regression for Classification (Moons Dataset)
def plot_polynomial_logistic_regression(X, y):
    # Use Polynomial Features with Logistic Regression
    polynomial_logistic_regression = make_pipeline(
        PolynomialFeatures(degree=3),
        LogisticRegression()
    )

    polynomial_logistic_regression.fit(X, y)

    plot_decision_boundary(polynomial_logistic_regression, X, y,
                           "Linear Regression")


# Support Vector Machines (SVM)
def plot_svm(X, y):
    model = SVC(kernel='rbf', C=1, gamma=0.5)
    model.fit(X, y)
    plot_decision_boundary(model, X, y,
                           "Support Vector Machine (SVM)")


# k-Nearest Neighbors (KNN)
def plot_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    plot_decision_boundary(model, X, y,
                           "k-Nearest Neighbors (KNN)")


# Decision Trees
def plot_decision_tree(X, y):
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X, y)
    plot_decision_boundary(model, X, y,
                           "Decision Tree")


# Random Forests
def plot_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    plot_decision_boundary(model, X, y,
                           "Random Forest")


# Generate the Moons dataset with 1000 samples
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# Generate plots
plot_polynomial_logistic_regression(X, y)
plot_svm(X, y)
plot_knn(X, y)
plot_decision_tree(X, y)
plot_random_forest(X, y)
