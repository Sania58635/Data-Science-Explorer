"""
Data Cleaning & Feature Engineering Content Module
Preparing data for modeling excellence
"""

def get_content():
    return {
        'summary': """
        <div class="topic-summary">
            <h3>📋 What is Data Cleaning & Feature Engineering?</h3>
            <p>Data cleaning fixes errors and handles missing values, while feature engineering creates new variables that improve model performance. These preprocessing steps transform raw data into analysis-ready formats and often matter more than algorithm choice.</p>
            <p><strong>Used in:</strong> All real-world data projects, Kaggle competitions, Production ML pipelines, Data Warehousing, and ETL processes.</p>
        </div>
        """,
        'topics': [
            {
                'id': 'missing-data',
                'title': 'Handling Missing Data',
                'description': 'Strategies for incomplete datasets',
                'subtopics': [
                    {
                        'name': 'Types of Missingness',
                        'content': """
                        <h3>Missing Data Mechanisms</h3>
                        <p><strong>MCAR (Missing Completely At Random):</strong> Missingness independent of any data</p>
                        <ul><li>Example: Sensor randomly fails</li></ul>
                        <p><strong>MAR (Missing At Random):</strong> Missingness depends on observed data</p>
                        <ul><li>Example: Younger people skip income questions</li></ul>
                        <p><strong>MNAR (Missing Not At Random):</strong> Missingness depends on unobserved data</p>
                        <ul><li>Example: High earners don't report income</li></ul>
                        <p><strong>Impact:</strong> Understanding mechanism guides imputation strategy</p>
                        <div class="visual">
                            <img src="/static/images/missing_data_pattern.png" alt="Missing Data Pattern" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Imputation Methods',
                        'content': """
                        <h3>Missing Value Imputation</h3>
                        <p><strong>Deletion Approaches:</strong></p>
                        <ul>
                            <li><strong>Listwise Deletion:</strong> Remove rows with any missing values</li>
                            <li><strong>Pairwise Deletion:</strong> Use available data for each analysis</li>
                            <li><strong>Risk:</strong> Loss of data, potential bias</li>
                        </ul>
                        <p><strong>Simple Imputation:</strong></p>
                        <ul>
                            <li><strong>Mean/Median:</strong> Replace with central tendency (numerical)</li>
                            <li><strong>Mode:</strong> Most frequent value (categorical)</li>
                            <li><strong>Forward/Backward Fill:</strong> Carry values in time series</li>
                        </ul>
                        <p><strong>Advanced Methods:</strong></p>
                        <ul>
                            <li><strong>KNN Imputation:</strong> Use similar samples' values</li>
                            <li><strong>MICE:</strong> Multiple Imputation by Chained Equations</li>
                            <li><strong>Model-Based:</strong> Train regression to predict missing values</li>
                        </ul>
                        <div class="example">
                            <pre><code>from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)</code></pre>
                        </div>
                        """
                    }
                ]
            },
            {
                'id': 'outlier-detection',
                'title': 'Outlier Detection',
                'description': 'Identifying and handling anomalies',
                'subtopics': [
                    {
                        'name': 'Statistical Methods',
                        'content': """
                        <h3>Statistical Outlier Detection</h3>
                        <p><strong>Z-Score Method:</strong> Points beyond ±3 standard deviations</p>
                        <ul><li>Assumes normal distribution</li></ul>
                        <p><strong>IQR Method:</strong> Outliers below Q1-1.5×IQR or above Q3+1.5×IQR</p>
                        <ul><li>Robust to distribution shape</li></ul>
                        <p><strong>Modified Z-Score:</strong> Uses median absolute deviation (MAD)</p>
                        <ul><li>More robust to outliers than standard Z-score</li></ul>
                        <div class="visual">
                            <img src="/static/images/outlier_detection.png" alt="Outlier Detection Methods" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <pre><code>import numpy as np

# IQR method
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
outliers = (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Machine Learning Methods',
                        'content': """
                        <h3>ML-Based Outlier Detection</h3>
                        <p><strong>Isolation Forest:</strong> Isolates anomalies using random splits</p>
                        <ul><li>Outliers require fewer splits to isolate</li></ul>
                        <p><strong>Local Outlier Factor (LOF):</strong> Density-based anomaly detection</p>
                        <ul><li>Compares local density to neighbors</li></ul>
                        <p><strong>One-Class SVM:</strong> Learn boundary around normal data</p>
                        <p><strong>Autoencoders:</strong> High reconstruction error indicates outliers</p>
                        <p><strong>Handling Strategies:</strong></p>
                        <ul>
                            <li>Remove if data errors</li>
                            <li>Cap/floor (winsorization)</li>
                            <li>Transform (log, Box-Cox)</li>
                            <li>Keep if legitimate extreme values</li>
                        </ul>
                        """
                    }
                ]
            },
            {
                'id': 'encoding',
                'title': 'Encoding Categorical Variables',
                'description': 'Converting categories to numerical format',
                'subtopics': [
                    {
                        'name': 'Encoding Techniques',
                        'content': """
                        <h3>Categorical Encoding Methods</h3>
                        <p><strong>Label Encoding:</strong> Assign integer to each category</p>
                        <ul>
                            <li>Good for: Ordinal variables (low/medium/high)</li>
                            <li>Risk: Implies ordering when none exists</li>
                        </ul>
                        <p><strong>One-Hot Encoding:</strong> Binary column for each category</p>
                        <ul>
                            <li>Good for: Nominal variables without order</li>
                            <li>Risk: Curse of dimensionality with high cardinality</li>
                        </ul>
                        <p><strong>Target Encoding:</strong> Replace category with target mean</p>
                        <ul>
                            <li>Good for: High-cardinality features</li>
                            <li>Risk: Target leakage (use cross-validation)</li>
                        </ul>
                        <p><strong>Frequency Encoding:</strong> Replace with occurrence frequency</p>
                        <p><strong>Binary Encoding:</strong> Convert to binary digits (compact)</p>
                        <div class="visual">
                            <img src="/static/images/encoding_comparison.png" alt="Encoding Methods Comparison" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <pre><code>import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'])

# Label encoding
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])</code></pre>
                        </div>
                        """
                    }
                ]
            },
            {
                'id': 'scaling',
                'title': 'Feature Scaling',
                'description': 'Normalizing feature ranges',
                'subtopics': [
                    {
                        'name': 'Scaling Methods',
                        'content': """
                        <h3>Feature Scaling Techniques</h3>
                        <p><strong>Standardization (Z-Score Normalization):</strong> (x - μ) / σ</p>
                        <ul>
                            <li>Results in mean=0, std=1</li>
                            <li>Good for: Algorithms assuming normal distribution (SVM, logistic regression)</li>
                        </ul>
                        <p><strong>Min-Max Scaling:</strong> (x - min) / (max - min)</p>
                        <ul>
                            <li>Scales to [0,1] range</li>
                            <li>Good for: Neural networks, image data</li>
                            <li>Sensitive to outliers</li>
                        </ul>
                        <p><strong>Robust Scaling:</strong> Uses median and IQR (robust to outliers)</p>
                        <p><strong>MaxAbs Scaling:</strong> Scale by maximum absolute value</p>
                        <p><strong>When to Scale:</strong></p>
                        <ul>
                            <li><strong>Required:</strong> SVM, neural networks, k-NN, PCA</li>
                            <li><strong>Not Required:</strong> Tree-based models (Random Forest, XGBoost)</li>
                        </ul>
                        <div class="visual">
                            <img src="/static/images/scaling_comparison.png" alt="Feature Scaling Comparison" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <pre><code>from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Min-Max scaling
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X_train)</code></pre>
                        </div>
                        """
                    }
                ]
            },
            {
                'id': 'feature-selection',
                'title': 'Feature Selection',
                'description': 'Choosing the most informative features',
                'subtopics': [
                    {
                        'name': 'Filter Methods',
                        'content': """
                        <h3>Filter-Based Selection</h3>
                        <p><strong>Correlation Analysis:</strong> Remove highly correlated features</p>
                        <p><strong>Variance Threshold:</strong> Remove low-variance features</p>
                        <p><strong>Chi-Square Test:</strong> Statistical independence test (categorical target)</p>
                        <p><strong>Mutual Information:</strong> Non-linear dependency measure</p>
                        <p><strong>ANOVA F-test:</strong> Univariate feature significance</p>
                        <p><strong>Advantages:</strong> Fast, model-agnostic, good for high-dimensional data</p>
                        """
                    },
                    {
                        'name': 'Wrapper Methods',
                        'content': """
                        <h3>Wrapper-Based Selection</h3>
                        <p><strong>Recursive Feature Elimination (RFE):</strong> Iteratively remove weakest features</p>
                        <p><strong>Forward Selection:</strong> Start empty, add best features iteratively</p>
                        <p><strong>Backward Elimination:</strong> Start full, remove worst features iteratively</p>
                        <p><strong>Advantages:</strong> Considers feature interactions, optimized for specific model</p>
                        <p><strong>Disadvantages:</strong> Computationally expensive, risk of overfitting</p>
                        """
                    },
                    {
                        'name': 'Embedded Methods',
                        'content': """
                        <h3>Embedded Feature Selection</h3>
                        <p><strong>Lasso (L1 Regularization):</strong> Shrinks coefficients to zero</p>
                        <p><strong>Tree Feature Importance:</strong> Gini/entropy importance from tree models</p>
                        <p><strong>Elastic Net:</strong> Combines L1 and L2 regularization</p>
                        <p><strong>SHAP Values:</strong> Shapley-based feature importance</p>
                        <p><strong>PCA (Principal Component Analysis):</strong> Linear dimensionality reduction</p>
                        <ul>
                            <li>Creates new features (principal components)</li>
                            <li>Captures maximum variance</li>
                            <li>Orthogonal components</li>
                        </ul>
                        <p><strong>Advantages:</strong> Balance between filter and wrapper methods</p>
                        <div class="visual">
                            <img src="/static/images/feature_importance.png" alt="Feature Importance Ranking" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    }
                ]
            }
        ]
    }

