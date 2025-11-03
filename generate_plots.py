"""
Generate and save visualization plots for Data Science Encyclopedia
Run this script to generate all the plots
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create images directory
os.makedirs('static/images', exist_ok=True)

# Set figure defaults for consistent styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fb'
plt.rcParams['axes.edgecolor'] = '#e2e8f0'
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['text.color'] = '#0a0e27'
plt.rcParams['axes.labelcolor'] = '#0a0e27'
plt.rcParams['xtick.color'] = '#4a5568'
plt.rcParams['ytick.color'] = '#4a5568'

def save_plot(filename):
    """Save plot with consistent settings"""
    plt.tight_layout()
    plt.savefig(f'static/images/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated {filename}")

# 1. Linear Regression Example
def generate_linear_regression():
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(50) * 2
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, s=80, color='#0066ff', label='Data points', edgecolors='white', linewidth=1.5)
    plt.plot(X, y_pred, color='#00d9a3', linewidth=3, label='Regression line')
    plt.xlabel('Feature (X)', fontsize=12, fontweight='bold')
    plt.ylabel('Target (y)', fontsize=12, fontweight='bold')
    plt.title('Linear Regression Example', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    save_plot('linear_regression.png')

# 2. Classification Example
def generate_classification():
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1, flip_y=0.1)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, s=80, 
                         color='#0066ff', label='Class 0', edgecolors='white', linewidth=1.5)
    scatter = plt.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, s=80, 
                         color='#00d9a3', label='Class 1', edgecolors='white', linewidth=1.5)
    plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
    plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
    plt.title('Binary Classification Problem', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    save_plot('classification.png')

# 3. K-Means Clustering
def generate_kmeans():
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0)
    
    plt.figure(figsize=(10, 6))
    colors = ['#0066ff', '#00d9a3', '#ff6b6b']
    for i in range(3):
        plt.scatter(X[y==i, 0], X[y==i, 1], alpha=0.7, s=80, 
                   color=colors[i], label=f'Cluster {i+1}', 
                   edgecolors='white', linewidth=1.5)
    plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
    plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
    plt.title('K-Means Clustering (K=3)', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    save_plot('kmeans.png')

# 4. Normal Distribution
def generate_normal_distribution():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, color='#0066ff', edgecolor='white', linewidth=1.5)
    plt.xlabel('Value', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Normal Distribution (μ=0, σ=1)', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    save_plot('normal_distribution.png')

# 5. Box Plot
def generate_boxplot():
    np.random.seed(42)
    data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'], 
                      patch_artist=True, widths=0.6)
    
    colors = ['#0066ff', '#00d9a3', '#8b5cf6']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title('Box Plot Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    save_plot('boxplot.png')

# 6. Correlation Heatmap
def generate_correlation_heatmap():
    np.random.seed(42)
    data = np.random.randn(50, 5)
    corr = np.corrcoef(data.T)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
                yticklabels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
    plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
    save_plot('correlation_heatmap.png')

# 7. PCA Visualization
def generate_pca():
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_informative=3, 
                               n_redundant=0, n_classes=2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.7, s=80, 
               color='#0066ff', label='Class 0', edgecolors='white', linewidth=1.5)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.7, s=80, 
               color='#00d9a3', label='Class 1', edgecolors='white', linewidth=1.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
              fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
              fontsize=12, fontweight='bold')
    plt.title('PCA Dimensionality Reduction', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    save_plot('pca.png')

# 8. Decision Tree
def generate_decision_tree():
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1)
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    plt.figure(figsize=(14, 8))
    plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'],
             class_names=['Class 0', 'Class 1'], rounded=True, fontsize=10)
    plt.title('Decision Tree Structure', fontsize=14, fontweight='bold', pad=20)
    save_plot('decision_tree.png')

# 9. Gradient Descent
def generate_gradient_descent():
    def loss_function(x):
        return x**2 + 2*x + 1
    
    x_vals = np.linspace(-4, 2, 100)
    y_vals = loss_function(x_vals)
    
    # Gradient descent steps
    learning_rate = 0.3
    x = -3
    steps = []
    for _ in range(10):
        steps.append((x, loss_function(x)))
        gradient = 2*x + 2
        x = x - learning_rate * gradient
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, color='#0066ff', linewidth=3, label='Loss function')
    
    step_x, step_y = zip(*steps)
    plt.scatter(step_x, step_y, color='#ff6b6b', s=100, zorder=5, 
               edgecolors='white', linewidth=2, label='Gradient descent steps')
    plt.plot(step_x, step_y, 'r--', alpha=0.5, linewidth=2)
    
    plt.xlabel('Parameter Value', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Gradient Descent Optimization', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    save_plot('gradient_descent.png')

# 10. ROC Curve
def generate_roc_curve():
    from sklearn.metrics import roc_curve, auc
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    y_scores[y_true == 1] += 0.3
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='#0066ff', linewidth=3, 
            label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve Example', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True, loc='lower right')
    plt.grid(True, alpha=0.3)
    save_plot('roc_curve.png')

# Main execution
if __name__ == '__main__':
    print("\n🎨 Generating visualizations for Data Science Encyclopedia...\n")
    
    generate_linear_regression()
    generate_classification()
    generate_kmeans()
    generate_normal_distribution()
    generate_boxplot()
    generate_correlation_heatmap()
    generate_pca()
    generate_decision_tree()
    generate_gradient_descent()
    generate_roc_curve()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Images saved to: static/images/\n")

