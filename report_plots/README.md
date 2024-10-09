
# Report Plots

## Overview

This project contains manually generated plots using a combination of Python libraries, including `Matplotlib`, `Seaborn`, and `Scikit-learn`. These plots are used for visualizing key aspects of machine learning models, including semi-supervised learning, overfitting, performance metrics, and more.

## Example Plots

### 1. Semi-Supervised Learning Plot

This plot visualizes semi-supervised learning using KMeans clustering and SVM decision boundaries for classifying animal species.

```python
# Apply KMeans clustering to the unlabeled data
kmeans = KMeans(n_clusters=3, random_state=42).fit(combined_data)

# Use SVM to draw decision boundaries based on labeled data
svm = SVC(kernel='linear', C=1.0)
svm.fit(labeled_data, labeled_labels)

# Plot decision boundaries
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
```

### 2. Overfitting Plot

This plot demonstrates overfitting in machine learning, comparing a simple quadratic function and a high-degree polynomial fit.


```python
# Create an overfitted model (high-degree polynomial)
poly_features = PolynomialFeatures(degree=10)
X_poly = poly_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Plot the overfitted model
X_fit_poly = poly_features.transform(X_fit)
y_fit_poly = model.predict(X_fit_poly)
plt.plot(X_fit, y_fit_poly, color='red', label='Overfitted Model')
```

### 3. Performance Metrics Plot

This script visualizes classification performance metrics, such as precision-recall and ROC curves, and displays a confusion matrix.

```python
# Precision-recall values
precision, recall, thresholds = precision_recall_curve(y_classification, y_pred_proba)

# Plot Precision vs Recall
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
```

### 4. Activation Functions Plot

This script generates activation function plots for a variety of neural network functions, including Sigmoid, ReLU, and GELU.

```python
# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid Derivative
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### 5. Bounding Boxes Plot

This script draws bounding boxes on images based on the provided JSON annotations.

```python
# Draw bounding boxes on the image
for obj in json_data['objects']:
    bbox = obj['bbox']
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
```

---

## Licensing Information

### Project License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). For more details, see the [LICENSE](LICENSE.md) file.