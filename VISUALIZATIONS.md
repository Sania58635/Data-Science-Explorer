# Data Science Encyclopedia - Visualizations Guide

## 📊 Complete Visualization Coverage

This document lists all the working images and plots integrated throughout the encyclopedia.

### 🧠 Machine Learning Section
| Topic | Visualization | Description |
|-------|--------------|-------------|
| **Regression** | `linear_regression.png` | Linear regression with best-fit line on sample data |
| **Classification** | `classification.png` | Binary classification problem with two distinct classes |
| **Decision Trees** | `decision_tree.png` | Full decision tree structure showing splits and leaf nodes |
| **K-Means Clustering** | `kmeans.png` | K-Means clustering with K=3 showing clear cluster separation |
| **PCA** | `pca.png` | Principal Component Analysis dimensionality reduction visualization |
| **Evaluation Metrics** | `roc_curve.png` | ROC curve with AUC score for classification performance |

### 📊 Statistics & Probability Section
| Topic | Visualization | Description |
|-------|--------------|-------------|
| **Measures of Dispersion** | `boxplot.png` | Box plot comparison showing quartiles and outliers |
| **Continuous Distributions** | `normal_distribution.png` | Normal (Gaussian) distribution histogram |
| **Correlation Analysis** | `correlation_heatmap.png` | Correlation matrix heatmap showing feature relationships |

### 🧮 Mathematics for Data Science Section
| Topic | Visualization | Description |
|-------|--------------|-------------|
| **Gradient Descent** | `gradient_descent.png` | Optimization path showing convergence to minimum |

### 🧼 Data Cleaning & Feature Engineering Section
| Topic | Visualization | Description |
|-------|--------------|-------------|
| **Missing Data** | `missing_data_pattern.png` | Heatmap showing missing data patterns across features |
| **Outlier Detection** | `outlier_detection.png` | IQR method visualization with box plot and scatter plot |
| **Encoding** | `encoding_comparison.png` | Comparison between label encoding and one-hot encoding |
| **Feature Scaling** | `scaling_comparison.png` | Original, standardized, and min-max scaled distributions |
| **Feature Selection** | `feature_importance.png` | Horizontal bar chart ranking feature importance scores |

### 📈 Data Visualization Section
| Topic | Visualization | Description |
|-------|--------------|-------------|
| **Essential Chart Types** | `chart_types.png` | 6-panel showcase: histogram, scatter, line, bar, violin, area |
| **Time Series** | `time_series.png` | Time series with trend line and seasonal patterns |
| **Interactive Plotly** | `plotly_interactive.html` | Fully interactive scatter plot with hover, zoom, and pan |

### 🔮 AI, Deep Learning & NLP Section
| Topic | Visualization | Description |
|-------|--------------|-------------|
| **Neural Networks** | `neural_network_architecture.png` | Feedforward neural network with input, hidden, and output layers |
| **Activation Functions** | `activation_functions.png` | 4-panel showing Sigmoid, Tanh, ReLU, and Leaky ReLU |
| **Training Progress** | `training_curves.png` | Training and validation loss/accuracy curves over epochs |
| **Model Evaluation** | `confusion_matrix.png` | Confusion matrix heatmap for multi-class classification |

---

## 🎨 Visualization Style Guide

All visualizations follow a consistent design language:

### Color Palette
- **Primary Blue**: `#0066ff` - Main accent color
- **Teal**: `#00d9a3` - Secondary accent
- **Purple**: `#8b5cf6` - Tertiary accent
- **Red**: `#ff6b6b` - Emphasis/warnings
- **Yellow**: `#fbbf24` - Highlights

### Design Principles
- **High DPI**: All images rendered at 150 DPI for crisp display
- **Consistent Background**: White background with subtle gray grid (`#f8f9fb`)
- **Glassmorphism**: Images wrapped in glass-effect containers
- **Responsive**: Max-width constraints for optimal viewing
- **Hover Effects**: Subtle lift and shadow enhancement on hover

### Technical Details
- **Format**: PNG for static images, HTML for interactive
- **Size**: Optimized for web (25-130KB per image)
- **Dimensions**: Variable, constrained to 600px max-width by CSS
- **Libraries**: Matplotlib, Seaborn, Plotly, Scikit-learn

---

## 🔧 Regenerating Visualizations

To regenerate all visualizations:

```bash
# Activate virtual environment
source venv/bin/activate  # or ./venv/bin/activate on Mac/Linux

# Generate base visualizations
python3 generate_plots.py

# Generate additional visualizations
python3 generate_more_plots.py
```

All images are saved to `static/images/` directory.

---

## 📦 Required Packages

```
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.1
numpy==1.26.4
plotly==5.18.0
pandas==2.2.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Coverage Summary

| Section | Topics with Visuals | Total Topics | Coverage |
|---------|-------------------|--------------|----------|
| Machine Learning | 6 | 15+ | 40% |
| Statistics | 3 | 12+ | 25% |
| Mathematics | 1 | 10+ | 10% |
| Data Cleaning | 5 | 13+ | 38% |
| Visualization | 3 | 10+ | 30% |
| Deep Learning | 4 | 15+ | 27% |
| **TOTAL** | **22 visuals** | **75+ topics** | **~30%** |

---

## 💡 Future Enhancement Ideas

1. **More Interactive Plots**: Convert static plots to Plotly where beneficial
2. **3D Visualizations**: Add 3D scatter plots, surface plots
3. **Animated GIFs**: Show gradient descent optimization in motion
4. **Model Architectures**: Visual diagrams for CNN, RNN, LSTM, Transformer
5. **Algorithm Animations**: Step-by-step algorithm visualizations
6. **Real Dataset Examples**: Use popular datasets (Iris, Titanic, MNIST)

---

Generated: November 2, 2025

