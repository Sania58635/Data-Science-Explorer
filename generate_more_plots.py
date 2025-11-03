"""
Generate additional visualizations for Data Science Encyclopedia
Data Cleaning, Visualization, and Deep Learning sections
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
import plotly.graph_objects as go
import plotly.express as px
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set figure defaults
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fb'
plt.rcParams['axes.edgecolor'] = '#e2e8f0'
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['text.color'] = '#0a0e27'
plt.rcParams['axes.labelcolor'] = '#0a0e27'
plt.rcParams['xtick.color'] = '#4a5568'
plt.rcParams['ytick.color'] = '#4a5568'

os.makedirs('static/images', exist_ok=True)

def save_plot(filename):
    """Save plot with consistent settings"""
    plt.tight_layout()
    plt.savefig(f'static/images/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated {filename}")

# ============= DATA CLEANING VISUALIZATIONS =============

# 1. Missing Data Pattern
def generate_missing_data():
    np.random.seed(42)
    # Create data with missing patterns
    data = pd.DataFrame({
        'Feature_A': np.random.randn(100),
        'Feature_B': np.random.randn(100),
        'Feature_C': np.random.randn(100),
        'Feature_D': np.random.randn(100)
    })
    
    # Introduce missing data patterns
    data.loc[np.random.choice(100, 20, replace=False), 'Feature_A'] = np.nan
    data.loc[np.random.choice(100, 15, replace=False), 'Feature_B'] = np.nan
    data.loc[np.random.choice(100, 25, replace=False), 'Feature_C'] = np.nan
    data.loc[np.random.choice(100, 10, replace=False), 'Feature_D'] = np.nan
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='RdYlGn_r', yticklabels=False)
    plt.title('Missing Data Pattern Visualization', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Samples', fontsize=12, fontweight='bold')
    save_plot('missing_data_pattern.png')

# 2. Outlier Detection with IQR
def generate_outlier_detection():
    np.random.seed(42)
    # Normal data with outliers
    data = np.concatenate([np.random.normal(50, 10, 100), [10, 15, 90, 95, 100]])
    
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    plt.figure(figsize=(10, 6))
    
    # Box plot
    ax1 = plt.subplot(1, 2, 1)
    bp = plt.boxplot(data, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#0066ff')
    bp['boxes'][0].set_alpha(0.7)
    plt.axhline(lower_bound, color='#ff6b6b', linestyle='--', linewidth=2, label='IQR bounds')
    plt.axhline(upper_bound, color='#ff6b6b', linestyle='--', linewidth=2)
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title('Box Plot with Outliers', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Scatter plot
    ax2 = plt.subplot(1, 2, 2)
    outliers = (data < lower_bound) | (data > upper_bound)
    plt.scatter(np.arange(len(data))[~outliers], data[~outliers], 
               color='#0066ff', s=80, alpha=0.7, label='Normal', edgecolors='white', linewidth=1.5)
    plt.scatter(np.arange(len(data))[outliers], data[outliers], 
               color='#ff6b6b', s=100, alpha=0.9, label='Outliers', edgecolors='white', linewidth=2)
    plt.axhline(lower_bound, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(upper_bound, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.5)
    plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title('Outlier Detection (IQR Method)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Outlier Detection', fontsize=14, fontweight='bold', y=1.02)
    save_plot('outlier_detection.png')

# 3. One-Hot Encoding Visualization
def generate_encoding_comparison():
    categories = ['Red', 'Blue', 'Green', 'Red', 'Blue']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Label Encoding
    label_encoded = [0, 1, 2, 0, 1]
    ax1.bar(range(len(categories)), label_encoded, color=['#ff6b6b', '#0066ff', '#00d9a3', '#ff6b6b', '#0066ff'], 
           alpha=0.7, edgecolor='white', linewidth=2)
    ax1.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Encoded Value', fontsize=12, fontweight='bold')
    ax1.set_title('Label Encoding', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # One-Hot Encoding
    one_hot = pd.get_dummies(categories)
    im = ax2.imshow(one_hot.T, cmap='RdYlBu_r', aspect='auto', alpha=0.8)
    ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Binary Features', fontsize=12, fontweight='bold')
    ax2.set_title('One-Hot Encoding', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories)
    ax2.set_yticks(range(len(one_hot.columns)))
    ax2.set_yticklabels(one_hot.columns)
    plt.colorbar(im, ax=ax2)
    
    plt.suptitle('Categorical Encoding Methods', fontsize=14, fontweight='bold', y=1.02)
    save_plot('encoding_comparison.png')

# 4. Feature Scaling Comparison
def generate_scaling_comparison():
    np.random.seed(42)
    data = np.random.exponential(50, 100)
    
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    data_standard = scaler_standard.fit_transform(data.reshape(-1, 1)).flatten()
    data_minmax = scaler_minmax.fit_transform(data.reshape(-1, 1)).flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Original
    axes[0].hist(data, bins=20, color='#0066ff', alpha=0.7, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].text(0.5, 0.95, f'Range: [{data.min():.1f}, {data.max():.1f}]', 
                transform=axes[0].transAxes, ha='center', va='top', fontsize=10)
    
    # Standardized
    axes[1].hist(data_standard, bins=20, color='#00d9a3', alpha=0.7, edgecolor='white', linewidth=1.5)
    axes[1].set_title('Standardized (Z-score)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].text(0.5, 0.95, f'Mean: {data_standard.mean():.2f}, Std: {data_standard.std():.2f}', 
                transform=axes[1].transAxes, ha='center', va='top', fontsize=10)
    
    # Min-Max
    axes[2].hist(data_minmax, bins=20, color='#8b5cf6', alpha=0.7, edgecolor='white', linewidth=1.5)
    axes[2].set_title('Min-Max Scaled', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].text(0.5, 0.95, f'Range: [{data_minmax.min():.2f}, {data_minmax.max():.2f}]', 
                transform=axes[2].transAxes, ha='center', va='top', fontsize=10)
    
    plt.suptitle('Feature Scaling Methods Comparison', fontsize=14, fontweight='bold', y=1.02)
    save_plot('scaling_comparison.png')

# 5. Feature Importance
def generate_feature_importance():
    features = ['Age', 'Income', 'Credit Score', 'Debt Ratio', 'Employment Years', 
               'Previous Loans', 'Payment History', 'Education Level']
    importance = np.array([0.25, 0.22, 0.18, 0.12, 0.10, 0.06, 0.05, 0.02])
    
    # Sort by importance
    idx = np.argsort(importance)
    features_sorted = [features[i] for i in idx]
    importance_sorted = importance[idx]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.7, len(features)))
    bars = plt.barh(features_sorted, importance_sorted, color=colors, 
                    alpha=0.8, edgecolor='white', linewidth=2)
    
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Feature Importance Ranking', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_sorted)):
        plt.text(val + 0.01, i, f'{val:.2f}', va='center', fontweight='bold', fontsize=10)
    
    save_plot('feature_importance.png')

# ============= VISUALIZATION SECTION =============

# 6. Multiple Chart Types
def generate_chart_types():
    np.random.seed(42)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Histogram
    ax1 = plt.subplot(2, 3, 1)
    data_hist = np.random.normal(100, 15, 1000)
    plt.hist(data_hist, bins=30, color='#0066ff', alpha=0.7, edgecolor='white', linewidth=1)
    plt.xlabel('Value', fontsize=10, fontweight='bold')
    plt.ylabel('Frequency', fontsize=10, fontweight='bold')
    plt.title('Histogram', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Scatter Plot
    ax2 = plt.subplot(2, 3, 2)
    x = np.random.randn(100)
    y = 2*x + np.random.randn(100)*0.5
    plt.scatter(x, y, alpha=0.6, s=60, color='#00d9a3', edgecolors='white', linewidth=1)
    plt.xlabel('X Variable', fontsize=10, fontweight='bold')
    plt.ylabel('Y Variable', fontsize=10, fontweight='bold')
    plt.title('Scatter Plot', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Line Chart
    ax3 = plt.subplot(2, 3, 3)
    x_line = np.linspace(0, 10, 50)
    y_line = np.sin(x_line) + np.random.randn(50)*0.1
    plt.plot(x_line, y_line, color='#8b5cf6', linewidth=2.5, marker='o', 
            markersize=4, markerfacecolor='white', markeredgewidth=1.5)
    plt.xlabel('Time', fontsize=10, fontweight='bold')
    plt.ylabel('Value', fontsize=10, fontweight='bold')
    plt.title('Line Chart', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    colors_bar = ['#0066ff', '#00d9a3', '#8b5cf6', '#ff6b6b', '#fbbf24']
    plt.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='white', linewidth=2)
    plt.xlabel('Category', fontsize=10, fontweight='bold')
    plt.ylabel('Value', fontsize=10, fontweight='bold')
    plt.title('Bar Chart', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Violin Plot
    ax5 = plt.subplot(2, 3, 5)
    data_violin = [np.random.normal(0, std, 100) for std in range(1, 4)]
    parts = plt.violinplot(data_violin, positions=[1, 2, 3], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#0066ff', '#00d9a3', '#8b5cf6'][i])
        pc.set_alpha(0.7)
    plt.xlabel('Group', fontsize=10, fontweight='bold')
    plt.ylabel('Value', fontsize=10, fontweight='bold')
    plt.title('Violin Plot', fontsize=11, fontweight='bold')
    plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Area Chart
    ax6 = plt.subplot(2, 3, 6)
    x_area = np.arange(50)
    y1 = np.random.rand(50).cumsum()
    y2 = np.random.rand(50).cumsum()
    plt.fill_between(x_area, y1, alpha=0.6, color='#0066ff', label='Series 1')
    plt.fill_between(x_area, y2, alpha=0.6, color='#00d9a3', label='Series 2')
    plt.xlabel('Time', fontsize=10, fontweight='bold')
    plt.ylabel('Cumulative Value', fontsize=10, fontweight='bold')
    plt.title('Area Chart', fontsize=11, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Essential Chart Types', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_plot('chart_types.png')

# 7. Interactive Plotly Visualization (save as HTML)
def generate_plotly_interactive():
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(200),
        'y': np.random.randn(200),
        'category': np.random.choice(['A', 'B', 'C'], 200),
        'size': np.random.randint(10, 50, 200)
    })
    
    fig = px.scatter(df, x='x', y='y', color='category', size='size',
                    title='Interactive Scatter Plot (Plotly)',
                    hover_data=['size'],
                    color_discrete_map={'A': '#0066ff', 'B': '#00d9a3', 'C': '#8b5cf6'})
    
    fig.update_layout(
        plot_bgcolor='#f8f9fb',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color='#0a0e27'),
        title_font_size=16,
        title_font_family='Arial',
        title_font_color='#0a0e27'
    )
    
    fig.write_html('static/images/plotly_interactive.html')
    print("✓ Generated plotly_interactive.html")

# 8. Time Series with Trend
def generate_time_series():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.linspace(100, 150, 365)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.randn(365) * 5
    values = trend + seasonal + noise
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, color='#0066ff', linewidth=2, alpha=0.7, label='Actual Data')
    plt.plot(dates, trend, color='#ff6b6b', linewidth=2.5, linestyle='--', label='Trend')
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title('Time Series with Trend Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    save_plot('time_series.png')

# ============= DEEP LEARNING VISUALIZATIONS =============

# 9. Neural Network Architecture
def generate_neural_network_diagram():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Layer positions
    layers = [4, 6, 6, 3]  # neurons per layer
    layer_x = [1, 3.5, 6, 8.5]
    
    colors = ['#0066ff', '#00d9a3', '#8b5cf6', '#ff6b6b']
    
    # Draw neurons
    neurons_pos = []
    for l, (n_neurons, x) in enumerate(zip(layers, layer_x)):
        y_positions = np.linspace(2, 8, n_neurons)
        layer_neurons = []
        for y in y_positions:
            circle = plt.Circle((x, y), 0.3, color=colors[l], alpha=0.7, 
                               edgecolor='white', linewidth=2, zorder=2)
            ax.add_patch(circle)
            layer_neurons.append((x, y))
        neurons_pos.append(layer_neurons)
    
    # Draw connections
    for l in range(len(layers)-1):
        for n1 in neurons_pos[l]:
            for n2 in neurons_pos[l+1]:
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]], 
                       color='gray', alpha=0.2, linewidth=0.5, zorder=1)
    
    # Labels
    ax.text(1, 0.5, 'Input\nLayer', ha='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 0.5, 'Hidden\nLayer 1', ha='center', fontsize=11, fontweight='bold')
    ax.text(6, 0.5, 'Hidden\nLayer 2', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.5, 0.5, 'Output\nLayer', ha='center', fontsize=11, fontweight='bold')
    
    plt.title('Feedforward Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
    save_plot('neural_network_architecture.png')

# 10. Activation Functions
def generate_activation_functions():
    x = np.linspace(-5, 5, 100)
    
    # Define activation functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.1*x)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid, color='#0066ff', linewidth=3)
    axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linewidth=0.5)
    axes[0, 0].axvline(0, color='k', linewidth=0.5)
    axes[0, 0].set_ylabel('Output', fontsize=11, fontweight='bold')
    
    # Tanh
    axes[0, 1].plot(x, tanh, color='#00d9a3', linewidth=3)
    axes[0, 1].set_title('Tanh: tanh(x)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linewidth=0.5)
    axes[0, 1].axvline(0, color='k', linewidth=0.5)
    
    # ReLU
    axes[1, 0].plot(x, relu, color='#8b5cf6', linewidth=3)
    axes[1, 0].set_title('ReLU: max(0, x)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linewidth=0.5)
    axes[1, 0].axvline(0, color='k', linewidth=0.5)
    axes[1, 0].set_xlabel('Input (x)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Output', fontsize=11, fontweight='bold')
    
    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu, color='#ff6b6b', linewidth=3)
    axes[1, 1].set_title('Leaky ReLU: max(0.1x, x)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linewidth=0.5)
    axes[1, 1].axvline(0, color='k', linewidth=0.5)
    axes[1, 1].set_xlabel('Input (x)', fontsize=11, fontweight='bold')
    
    plt.suptitle('Neural Network Activation Functions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_plot('activation_functions.png')

# 11. Training Loss Curves
def generate_training_curves():
    np.random.seed(42)
    epochs = np.arange(1, 51)
    
    # Simulate training curves
    train_loss = 2.0 * np.exp(-0.08 * epochs) + 0.1 + np.random.randn(50) * 0.02
    val_loss = 2.0 * np.exp(-0.06 * epochs) + 0.2 + np.random.randn(50) * 0.03
    
    train_acc = 1 - 0.8 * np.exp(-0.08 * epochs) + np.random.randn(50) * 0.01
    val_acc = 1 - 0.8 * np.exp(-0.06 * epochs) + np.random.randn(50) * 0.015
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_loss, color='#0066ff', linewidth=2.5, label='Training Loss', marker='o', markersize=3)
    ax1.plot(epochs, val_loss, color='#00d9a3', linewidth=2.5, label='Validation Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_acc, color='#8b5cf6', linewidth=2.5, label='Training Accuracy', marker='o', markersize=3)
    ax2.plot(epochs, val_acc, color='#ff6b6b', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Neural Network Training Progress', fontsize=14, fontweight='bold', y=1.02)
    save_plot('training_curves.png')

# 12. Confusion Matrix
def generate_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    
    # Simulate predictions
    y_true = np.random.randint(0, 3, 200)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.choice(200, 30, replace=False)
    y_pred[errors] = np.random.randint(0, 3, 30)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
               linewidths=2, cbar_kws={"shrink": 0.8},
               xticklabels=['Class 0', 'Class 1', 'Class 2'],
               yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    save_plot('confusion_matrix.png')

# Main execution
if __name__ == '__main__':
    print("\n🎨 Generating additional visualizations...\n")
    print("📊 DATA CLEANING SECTION:")
    generate_missing_data()
    generate_outlier_detection()
    generate_encoding_comparison()
    generate_scaling_comparison()
    generate_feature_importance()
    
    print("\n📈 VISUALIZATION SECTION:")
    generate_chart_types()
    generate_plotly_interactive()
    generate_time_series()
    
    print("\n🔮 DEEP LEARNING SECTION:")
    generate_neural_network_diagram()
    generate_activation_functions()
    generate_training_curves()
    generate_confusion_matrix()
    
    print("\n✅ All additional visualizations generated successfully!")
    print("📁 Total images in static/images/\n")

