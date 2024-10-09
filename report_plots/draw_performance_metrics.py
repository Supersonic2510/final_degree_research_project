import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import matplotlib.cm as cm

# 1. Classification Metrics Visualization
# Generate the Moons dataset for classification
X_classification, y_classification = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Train a polynomial logistic regression model
polynomial_logistic_regression = make_pipeline(PolynomialFeatures(degree=3), LogisticRegression())
polynomial_logistic_regression.fit(X_classification, y_classification)

# Predictions
y_pred_classification = polynomial_logistic_regression.predict(X_classification)
y_pred_proba = polynomial_logistic_regression.predict_proba(X_classification)[:, 1]

# Calculate precision-recall values
precision, recall, thresholds = precision_recall_curve(y_classification, y_pred_proba)

# Calculate ROC values
fpr, tpr, _ = roc_curve(y_classification, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Confusion Matrix
conf_matrix = confusion_matrix(y_classification, y_pred_classification)

# Plot Precision-Recall Tradeoff with dual Y-axis (Precision & Recall)
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot Precision vs Threshold
ax1.plot(thresholds, precision[:-1], color='blue', label='Precision')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Precision', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Recall vs Threshold
ax2 = ax1.twinx()
ax2.plot(thresholds, recall[:-1], color='red', label='Recall')
ax2.set_ylabel('Recall', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Precision-Recall Trade-off')
fig.tight_layout()
plt.grid(True)
plt.show()

# Plot AUROC with shaded area and AUC value
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label=f'AUC = {roc_auc:.2f}')
plt.fill_between(fpr, tpr, color='green', alpha=0.3)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Confusion Matrix Plot with Seaborn
plt.figure(figsize=(6, 6))

# Create a desaturated color palette with more transparency
cmap = sns.color_palette("coolwarm", desat=0.7)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, cbar=False, annot_kws={"size": 16})

# Annotate the confusion matrix with white text and adjusted placement
plt.text(0.5, 0.2, 'TN', fontsize=18, color='white', ha='center', va='center', weight='bold')
plt.text(1.5, 0.2, 'FP', fontsize=18, color='white', ha='center', va='center', weight='bold')
plt.text(0.5, 1.2, 'FN', fontsize=18, color='white', ha='center', va='center', weight='bold')
plt.text(1.5, 1.2, 'TP', fontsize=18, color='white', ha='center', va='center', weight='bold')

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
plt.show()

# 2. Clustering Metrics Visualization
# Generate a synthetic dataset for clustering with true labels
X_clustering, y_clustering = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred_clustering = kmeans.fit_predict(X_clustering)

# Calculate silhouette scores for each instance
silhouette_avg = silhouette_score(X_clustering, y_pred_clustering)
silhouette_vals = silhouette_samples(X_clustering, y_pred_clustering)

# Plotting the silhouette diagram
plt.figure(figsize=(10, 7))
y_lower, y_upper = 0, 0
n_clusters = np.unique(y_pred_clustering).shape[0]
for i in range(n_clusters):
    ith_cluster_silhouette_values = silhouette_vals[y_pred_clustering == i]
    ith_cluster_silhouette_values.sort()

    y_upper += len(ith_cluster_silhouette_values)
    color = cm.nipy_spectral(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    y_lower = y_upper

plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.title("Silhouette Plot for the various clusters")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")

plt.yticks([])
plt.xticks(np.arange(-0.1, 1.1, 0.2))
plt.show()
