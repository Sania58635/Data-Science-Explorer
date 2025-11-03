"""
Machine Learning Content Module
Comprehensive coverage of ML algorithms and techniques
"""

def get_content():
    return {
        'summary': """
        <div class="topic-summary">
            <h3>📋 What is Machine Learning?</h3>
            <p>Machine Learning enables computers to learn from data and make predictions without explicit programming. It powers modern AI from spam filters to self-driving cars.</p>
            <p><strong>Used in:</strong> Healthcare (diagnosis), Finance (fraud detection), E-commerce (recommendations), Robotics, Marketing, and Transportation.</p>
        </div>
        """,
        'topics': [
            {
                'id': 'supervised-learning',
                'title': 'Supervised Learning',
                'description': 'Learning from labeled data to make predictions',
                'subtopics': [
                    {
                        'name': 'Regression',
                        'content': """
                        <h3>Regression Analysis</h3>
                        <p><strong>Objective:</strong> Predict continuous numerical values</p>
                        <p><strong>Key Algorithms:</strong></p>
                        <ul>
                            <li><strong>Linear Regression:</strong> Models linear relationships using least squares optimization</li>
                            <li><strong>Polynomial Regression:</strong> Captures nonlinear patterns through feature transformation</li>
                            <li><strong>Ridge & Lasso:</strong> Regularized regression preventing overfitting through L2/L1 penalties</li>
                        </ul>
                        <div class="visual">
                            <img src="/static/images/linear_regression.png" alt="Linear Regression Visualization" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <h4>Example: House Price Prediction</h4>
                            <pre><code>from sklearn.linear_model import LinearRegression

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices
predictions = model.predict(X_test)</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Classification',
                        'content': """
                        <h3>Classification Algorithms</h3>
                        <p><strong>Objective:</strong> Predict categorical class labels</p>
                        <p><strong>Popular Methods:</strong></p>
                        <ul>
                            <li><strong>Logistic Regression:</strong> Probabilistic binary classification using sigmoid function</li>
                            <li><strong>Random Forest:</strong> Ensemble of decision trees with bagging for robust predictions</li>
                            <li><strong>Gradient Boosting:</strong> Sequential ensemble learning (XGBoost, LightGBM, CatBoost)</li>
                        </ul>
                        <div class="visual">
                            <img src="/static/images/classification.png" alt="Classification Problem Visualization" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <h4>Example: Customer Churn Prediction</h4>
                            <pre><code>from sklearn.ensemble import RandomForestClassifier

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict churn probability
churn_prob = clf.predict_proba(X_test)[:, 1]</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Decision Trees',
                        'content': """
                        <h3>Decision Trees</h3>
                        <p><strong>Concept:</strong> Hierarchical model splitting data based on feature thresholds</p>
                        <p><strong>Splitting Criteria:</strong></p>
                        <ul>
                            <li><strong>Gini Impurity:</strong> Measures classification purity at nodes</li>
                            <li><strong>Information Gain:</strong> Entropy reduction from splits</li>
                            <li><strong>Variance Reduction:</strong> For regression tasks</li>
                        </ul>
                        <p><strong>Advantages:</strong> Interpretable, handles nonlinear relationships, no feature scaling needed</p>
                        <p><strong>Challenges:</strong> Prone to overfitting, requires pruning or ensemble methods</p>
                        <div class="visual">
                            <img src="/static/images/decision_tree.png" alt="Decision Tree Structure" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Support Vector Machines',
                        'content': """
                        <h3>Support Vector Machines (SVM)</h3>
                        <p><strong>Core Idea:</strong> Find optimal hyperplane maximizing margin between classes</p>
                        <p><strong>Kernel Trick:</strong> Transform data to higher dimensions for nonlinear separation</p>
                        <ul>
                            <li><strong>Linear Kernel:</strong> For linearly separable data</li>
                            <li><strong>RBF Kernel:</strong> Gaussian kernel for complex boundaries</li>
                            <li><strong>Polynomial Kernel:</strong> Captures polynomial relationships</li>
                        </ul>
                        <p><strong>Applications:</strong> Text classification, image recognition, bioinformatics</p>
                        """
                    },
                    {
                        'name': 'Naive Bayes',
                        'content': """
                        <h3>Naive Bayes Classifier</h3>
                        <p><strong>Foundation:</strong> Probabilistic classification using Bayes' theorem</p>
                        <p><strong>Assumption:</strong> Features are conditionally independent given the class</p>
                        <p><strong>Variants:</strong></p>
                        <ul>
                            <li><strong>Gaussian NB:</strong> For continuous features with normal distribution</li>
                            <li><strong>Multinomial NB:</strong> For discrete counts (text classification)</li>
                            <li><strong>Bernoulli NB:</strong> For binary features</li>
                        </ul>
                        <p><strong>Strengths:</strong> Fast training, works well with high dimensions, effective for text</p>
                        """
                    }
                ]
            },
            {
                'id': 'unsupervised-learning',
                'title': 'Unsupervised Learning',
                'description': 'Discovering patterns in unlabeled data',
                'subtopics': [
                    {
                        'name': 'K-Means Clustering',
                        'content': """
                        <h3>K-Means Clustering</h3>
                        <p><strong>Algorithm:</strong> Partition data into K clusters by minimizing within-cluster variance</p>
                        <p><strong>Process:</strong></p>
                        <ol>
                            <li>Initialize K centroids randomly</li>
                            <li>Assign points to nearest centroid</li>
                            <li>Update centroids as cluster means</li>
                            <li>Repeat until convergence</li>
                        </ol>
                        <p><strong>Applications:</strong> Customer segmentation, image compression, anomaly detection</p>
                        <p><strong>Limitation:</strong> Requires specifying K, sensitive to initialization (use K-means++)</p>
                        <div class="visual">
                            <img src="/static/images/kmeans.png" alt="K-Means Clustering Visualization" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'DBSCAN',
                        'content': """
                        <h3>DBSCAN (Density-Based Spatial Clustering)</h3>
                        <p><strong>Approach:</strong> Groups densely connected points, identifies outliers</p>
                        <p><strong>Parameters:</strong></p>
                        <ul>
                            <li><strong>eps (ε):</strong> Neighborhood radius</li>
                            <li><strong>min_samples:</strong> Minimum points for dense region</li>
                        </ul>
                        <p><strong>Advantages:</strong> Discovers arbitrary-shaped clusters, robust to outliers, no K specification</p>
                        <p><strong>Use Cases:</strong> Geospatial analysis, fraud detection, network analysis</p>
                        """
                    },
                    {
                        'name': 'PCA (Principal Component Analysis)',
                        'content': """
                        <h3>Principal Component Analysis</h3>
                        <p><strong>Purpose:</strong> Dimensionality reduction through linear transformation</p>
                        <p><strong>Methodology:</strong> Project data onto principal components (eigenvectors of covariance matrix)</p>
                        <p><strong>Benefits:</strong></p>
                        <ul>
                            <li>Reduces feature space while preserving variance</li>
                            <li>Removes multicollinearity</li>
                            <li>Accelerates model training</li>
                            <li>Enables visualization of high-dimensional data</li>
                        </ul>
                        <p><strong>Interpretation:</strong> Components explain variance in decreasing order (scree plot analysis)</p>
                        <div class="visual">
                            <img src="/static/images/pca.png" alt="PCA Dimensionality Reduction" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    }
                ]
            },
            {
                'id': 'reinforcement-learning',
                'title': 'Reinforcement Learning',
                'description': 'Learning through interaction and rewards',
                'subtopics': [
                    {
                        'name': 'Q-Learning',
                        'content': """
                        <h3>Q-Learning Algorithm</h3>
                        <p><strong>Paradigm:</strong> Model-free reinforcement learning using Q-table</p>
                        <p><strong>Q-Value:</strong> Expected future reward for taking action in a state</p>
                        <p><strong>Update Rule:</strong> Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]</p>
                        <p><strong>Components:</strong></p>
                        <ul>
                            <li><strong>α (alpha):</strong> Learning rate</li>
                            <li><strong>γ (gamma):</strong> Discount factor for future rewards</li>
                            <li><strong>ε (epsilon):</strong> Exploration-exploitation tradeoff</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'Deep Q-Networks (DQN)',
                        'content': """
                        <h3>Deep Q-Networks</h3>
                        <p><strong>Innovation:</strong> Neural networks approximate Q-function for complex state spaces</p>
                        <p><strong>Key Techniques:</strong></p>
                        <ul>
                            <li><strong>Experience Replay:</strong> Store and sample past experiences for stable training</li>
                            <li><strong>Target Network:</strong> Separate network for stable Q-value targets</li>
                            <li><strong>Double DQN:</strong> Reduces overestimation bias</li>
                        </ul>
                        <p><strong>Applications:</strong> Game playing (Atari), robotics, autonomous systems</p>
                        """
                    }
                ]
            },
            {
                'id': 'evaluation-metrics',
                'title': 'Evaluation Metrics',
                'description': 'Measuring model performance effectively',
                'subtopics': [
                    {
                        'name': 'Classification Metrics',
                        'content': """
                        <h3>Classification Evaluation Metrics</h3>
                        <p><strong>Accuracy:</strong> (TP + TN) / Total → Overall correctness</p>
                        <p><strong>Precision:</strong> TP / (TP + FP) → Positive prediction accuracy</p>
                        <p><strong>Recall (Sensitivity):</strong> TP / (TP + FN) → True positive detection rate</p>
                        <p><strong>F1 Score:</strong> Harmonic mean of precision and recall</p>
                        <p><strong>ROC-AUC:</strong> Area under Receiver Operating Characteristic curve</p>
                        <ul>
                            <li>AUC = 1.0: Perfect classifier</li>
                            <li>AUC = 0.5: Random guessing</li>
                            <li>Higher AUC = Better discrimination ability</li>
                        </ul>
                        <p><strong>Confusion Matrix:</strong> Detailed breakdown of predictions vs actuals</p>
                        <div class="visual">
                            <img src="/static/images/roc_curve.png" alt="ROC Curve Example" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Regression Metrics',
                        'content': """
                        <h3>Regression Evaluation Metrics</h3>
                        <p><strong>MAE (Mean Absolute Error):</strong> Average absolute deviation from true values</p>
                        <p><strong>MSE (Mean Squared Error):</strong> Average squared error, penalizes large errors</p>
                        <p><strong>RMSE (Root Mean Squared Error):</strong> Square root of MSE, same units as target</p>
                        <p><strong>R² (Coefficient of Determination):</strong> Proportion of variance explained (0 to 1)</p>
                        <p><strong>MAPE:</strong> Mean Absolute Percentage Error for relative accuracy</p>
                        """
                    }
                ]
            },
            {
                'id': 'model-selection',
                'title': 'Model Selection & Validation',
                'description': 'Techniques for robust model development',
                'subtopics': [
                    {
                        'name': 'Cross-Validation',
                        'content': """
                        <h3>Cross-Validation Strategies</h3>
                        <p><strong>K-Fold CV:</strong> Split data into K folds, train on K-1, validate on 1, rotate</p>
                        <p><strong>Stratified K-Fold:</strong> Maintains class distribution in each fold</p>
                        <p><strong>Time Series CV:</strong> Forward chaining for temporal data</p>
                        <p><strong>Leave-One-Out CV:</strong> Use single sample for validation (small datasets)</p>
                        <p><strong>Purpose:</strong> Assess model generalization, reduce overfitting, tune hyperparameters</p>
                        """
                    },
                    {
                        'name': 'Regularization',
                        'content': """
                        <h3>Regularization Techniques</h3>
                        <p><strong>L1 Regularization (Lasso):</strong> Adds absolute coefficient penalty, promotes sparsity</p>
                        <p><strong>L2 Regularization (Ridge):</strong> Adds squared coefficient penalty, shrinks weights</p>
                        <p><strong>Elastic Net:</strong> Combines L1 and L2 for balanced regularization</p>
                        <p><strong>Dropout:</strong> Randomly deactivate neurons during training (neural networks)</p>
                        <p><strong>Early Stopping:</strong> Halt training when validation performance degrades</p>
                        <p><strong>Benefit:</strong> Prevents overfitting, improves generalization to unseen data</p>
                        """
                    }
                ]
            }
            ]
        }

