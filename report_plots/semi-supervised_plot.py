import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# Generate synthetic data
np.random.seed(42)

# Labeled data (small set)
labeled_data = np.array([
    [1.5, 2.5],
    [2.0, 3.0],
    [3.5, 2.0]
])

# Labels for the small labeled set (encoded as integers for model training)
labeled_labels = np.array([0, 1, 2])  # Assume 0: Species A, 1: Species B, 2: Species C

# Unlabeled data (large set)
unlabeled_data = np.random.randn(50, 2) + [3, 3]

# Combined data for clustering (to simulate semi-supervised learning)
combined_data = np.vstack((labeled_data, unlabeled_data))

# Apply KMeans clustering to the unlabeled data
kmeans = KMeans(n_clusters=3, random_state=42).fit(combined_data)

# Use SVM to draw decision boundaries based on labeled data
svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
svm.fit(labeled_data, labeled_labels)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot labeled data
plt.scatter(labeled_data[:, 0], labeled_data[:, 1], color='black', marker='o', s=100, label='Labeled Data')

# Plot unlabeled data
plt.scatter(unlabeled_data[:, 0], unlabeled_data[:, 1], color='gray', marker='x', s=60, label='Unlabeled Data')

# Plot decision boundaries
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

# Annotate labeled data points
species_labels = ['Species A', 'Species B', 'Species C']
for i, txt in enumerate(species_labels):
    plt.annotate(txt, (labeled_data[i, 0] + 0.1, labeled_data[i, 1] + 0.1), fontsize=12, color='black')

# Remove inner axes spines
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.gca().spines['left'].set_color('none')

# Define arrow positions slightly outside the plot range
arrow_start_x = xlim[0]
arrow_end_x = xlim[1]
arrow_start_y = ylim[0]
arrow_end_y = ylim[1]

# Add arrowheads to the outer axes (using annotations to simulate)
plt.annotate('', xy=(arrow_end_x, ylim[0]), xytext=(arrow_start_x, ylim[0]), arrowprops=dict(arrowstyle='->', lw=4, color='black'))
plt.annotate('', xy=(xlim[0], arrow_end_y), xytext=(xlim[0], arrow_start_y), arrowprops=dict(arrowstyle='->', lw=4, color='black'))

# Remove tick labels
plt.xticks([])
plt.yticks([])

# Add labels for the axes
plt.text(arrow_end_x - 0.3, ylim[0] + 0.1, 'Feature 1', ha='center', va='center', fontsize=14)
plt.text(xlim[0] + 0.1, arrow_end_y - 0.5, 'Feature 2', ha='center', va='center', rotation=90, fontsize=14)

plt.title('Semi-Supervised Learning: Animal Species Classification with Boundaries', fontsize=14)
plt.show()
