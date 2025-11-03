"""
Mathematics for Data Science Content Module
Essential mathematical foundations
"""

def get_content():
    return {
        'summary': """
        <div class="topic-summary">
            <h3>📋 What is Mathematics for Data Science?</h3>
            <p>Mathematics provides the language and tools for building and understanding data science algorithms. Linear algebra handles data transformations, calculus optimizes models, and information theory measures information content.</p>
            <p><strong>Used in:</strong> Neural Networks (linear algebra), Model Optimization (calculus), Dimensionality Reduction (eigenvalues), Compression (information theory), and Algorithm Design.</p>
        </div>
        """,
        'topics': [
            {
                'id': 'linear-algebra',
                'title': 'Linear Algebra',
                'description': 'The language of data transformations',
                'subtopics': [
                    {
                        'name': 'Vectors',
                        'content': """
                        <h3>Vectors and Vector Operations</h3>
                        <p><strong>Vector:</strong> Ordered array of numbers representing magnitude and direction</p>
                        <p><strong>Operations:</strong></p>
                        <ul>
                            <li><strong>Addition:</strong> Component-wise sum</li>
                            <li><strong>Scalar Multiplication:</strong> Scale all components</li>
                            <li><strong>Dot Product:</strong> v·w = Σ vᵢwᵢ → measures similarity</li>
                            <li><strong>Cross Product:</strong> Produces perpendicular vector (3D)</li>
                        </ul>
                        <p><strong>Vector Norms:</strong></p>
                        <ul>
                            <li>L1 Norm: |v|₁ = Σ|vᵢ| (Manhattan distance)</li>
                            <li>L2 Norm: |v|₂ = √(Σvᵢ²) (Euclidean distance)</li>
                            <li>L∞ Norm: Maximum absolute value</li>
                        </ul>
                        <div class="example">
                            <pre><code>import numpy as np

v = np.array([3, 4])
norm_l2 = np.linalg.norm(v)  # 5.0
norm_l1 = np.linalg.norm(v, 1)  # 7.0</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Matrices',
                        'content': """
                        <h3>Matrix Operations</h3>
                        <p><strong>Matrix:</strong> 2D array of numbers, represents linear transformations</p>
                        <p><strong>Key Operations:</strong></p>
                        <ul>
                            <li><strong>Matrix Multiplication:</strong> (A×B)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ</li>
                            <li><strong>Transpose:</strong> Aᵀ flips rows and columns</li>
                            <li><strong>Inverse:</strong> A⁻¹ such that A×A⁻¹ = I (identity)</li>
                            <li><strong>Determinant:</strong> Scalar representing transformation scaling</li>
                        </ul>
                        <p><strong>Special Matrices:</strong></p>
                        <ul>
                            <li><strong>Identity Matrix (I):</strong> 1s on diagonal, 0s elsewhere</li>
                            <li><strong>Diagonal Matrix:</strong> Non-zero only on diagonal</li>
                            <li><strong>Symmetric Matrix:</strong> A = Aᵀ</li>
                            <li><strong>Orthogonal Matrix:</strong> Columns are orthonormal vectors</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'Eigenvalues and Eigenvectors',
                        'content': """
                        <h3>Eigendecomposition</h3>
                        <p><strong>Eigenvector:</strong> Vector that only scales under transformation: $Av = \\lambda v$</p>
                        <p><strong>Eigenvalue ($\\lambda$):</strong> Scaling factor for corresponding eigenvector</p>
                        <p><strong>Applications:</strong></p>
                        <ul>
                            <li><strong>PCA:</strong> Principal components are eigenvectors of covariance matrix</li>
                            <li><strong>Spectral Clustering:</strong> Uses eigenvectors of graph Laplacian</li>
                            <li><strong>PageRank:</strong> Dominant eigenvector of web graph</li>
                            <li><strong>Stability Analysis:</strong> System dynamics and convergence</li>
                        </ul>
                        <p><strong>Singular Value Decomposition (SVD):</strong> $A = U\\Sigma V^T$</p>
                        <ul>
                            <li>Generalizes eigendecomposition to non-square matrices</li>
                            <li>Foundation of matrix factorization and recommender systems</li>
                        </ul>
                        """
                    }
                ]
            },
            {
                'id': 'calculus',
                'title': 'Calculus for Machine Learning',
                'description': 'Optimization through derivatives',
                'subtopics': [
                    {
                        'name': 'Derivatives and Gradients',
                        'content': """
                        <h3>Differentiation</h3>
                        <p><strong>Derivative:</strong> Rate of change of function with respect to variable</p>
                        <p><strong>Gradient:</strong> Vector of partial derivatives → direction of steepest ascent</p>
                        <p><strong>Chain Rule:</strong> Derivative of composition: (f∘g)'(x) = f'(g(x))·g'(x)</p>
                        <p><strong>In Machine Learning:</strong></p>
                        <ul>
                            <li>Gradients guide parameter updates during training</li>
                            <li>Backpropagation applies chain rule to neural networks</li>
                            <li>Jacobian matrix: All first-order partial derivatives</li>
                            <li>Hessian matrix: Second-order derivatives for curvature</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'Gradient Descent',
                        'content': """
                        <h3>Gradient Descent Optimization</h3>
                        <p><strong>Algorithm:</strong> Iteratively update parameters opposite to gradient direction</p>
                        <p><strong>Update Rule:</strong></p>
                        $$\\theta = \\theta - \\alpha\\nabla J(\\theta)$$
                        <ul>
                            <li>$\\theta$: Model parameters</li>
                            <li>$\\alpha$: Learning rate (step size)</li>
                            <li>$\\nabla J(\\theta)$: Gradient of loss function</li>
                        </ul>
                        <p><strong>Variants:</strong></p>
                        <ul>
                            <li><strong>Batch Gradient Descent:</strong> Use entire dataset for each update</li>
                            <li><strong>Stochastic Gradient Descent (SGD):</strong> Use single random sample</li>
                            <li><strong>Mini-Batch Gradient Descent:</strong> Use small batch of samples</li>
                            <li><strong>Adam:</strong> Adaptive learning rates with momentum</li>
                            <li><strong>RMSprop:</strong> Adaptive learning rate per parameter</li>
                        </ul>
                        <div class="visual">
                            <img src="/static/images/gradient_descent.png" alt="Gradient Descent Optimization" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Optimization Landscape',
                        'content': """
                        <h3>Understanding Optimization</h3>
                        <p><strong>Local vs Global Minimum:</strong> Local minima trap, global minimum is optimal</p>
                        <p><strong>Saddle Points:</strong> Zero gradient but not minimum (common in high dimensions)</p>
                        <p><strong>Challenges:</strong></p>
                        <ul>
                            <li><strong>Vanishing Gradients:</strong> Gradients become too small (deep networks)</li>
                            <li><strong>Exploding Gradients:</strong> Gradients grow exponentially</li>
                            <li><strong>Plateaus:</strong> Flat regions with slow convergence</li>
                        </ul>
                        <p><strong>Solutions:</strong> Momentum, adaptive learning rates, gradient clipping, careful initialization</p>
                        """
                    }
                ]
            },
            {
                'id': 'optimization',
                'title': 'Optimization Theory',
                'description': 'Finding optimal solutions',
                'subtopics': [
                    {
                        'name': 'Convex Optimization',
                        'content': """
                        <h3>Convexity in Optimization</h3>
                        <p><strong>Convex Function:</strong> Line segment between any two points lies above function</p>
                        <p><strong>Property:</strong> Any local minimum is also global minimum</p>
                        <p><strong>Convex Problems:</strong></p>
                        <ul>
                            <li>Linear regression (MSE loss)</li>
                            <li>Logistic regression (log loss)</li>
                            <li>Support Vector Machines</li>
                            <li>Lasso and Ridge regression</li>
                        </ul>
                        <p><strong>Benefits:</strong> Guaranteed convergence, efficient algorithms, theoretical guarantees</p>
                        <p><strong>Non-Convex:</strong> Neural networks (multiple local minima, more complex optimization)</p>
                        """
                    },
                    {
                        'name': 'Lagrange Multipliers',
                        'content': """
                        <h3>Constrained Optimization</h3>
                        <p><strong>Problem:</strong> Optimize f(x) subject to constraints g(x) = 0</p>
                        <p><strong>Lagrangian:</strong> L(x,λ) = f(x) + λg(x)</p>
                        <p><strong>Solution:</strong> Find where ∇L = 0 (gradient with respect to x and λ)</p>
                        <p><strong>Applications:</strong></p>
                        <ul>
                            <li><strong>SVM:</strong> Maximize margin subject to classification constraints</li>
                            <li><strong>Portfolio Optimization:</strong> Maximize returns with budget constraint</li>
                            <li><strong>KKT Conditions:</strong> Generalization for inequality constraints</li>
                        </ul>
                        """
                    }
                ]
            },
            {
                'id': 'information-theory',
                'title': 'Information Theory',
                'description': 'Quantifying information and uncertainty',
                'subtopics': [
                    {
                        'name': 'Entropy',
                        'content': """
                        <h3>Shannon Entropy</h3>
                        <p><strong>Definition:</strong></p>
                        $$H(X) = -\\sum_{x} p(x) \\log_2 p(x)$$
                        <p><strong>Interpretation:</strong> Average information content, measure of uncertainty/randomness</p>
                        <p><strong>Properties:</strong></p>
                        <ul>
                            <li>H(X) ≥ 0 (always non-negative)</li>
                            <li>Maximum when all outcomes equally likely (uniform distribution)</li>
                            <li>Zero when outcome is certain (deterministic)</li>
                        </ul>
                        <p><strong>Applications:</strong></p>
                        <ul>
                            <li><strong>Decision Trees:</strong> Information gain for splitting</li>
                            <li><strong>Feature Selection:</strong> Choose features with high information</li>
                            <li><strong>Compression:</strong> Optimal coding based on entropy</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'KL Divergence',
                        'content': """
                        <h3>Kullback-Leibler Divergence</h3>
                        <p><strong>Formula:</strong></p>
                        $$D_{KL}(P||Q) = \\sum_{x} P(x) \\log\\frac{P(x)}{Q(x)}$$
                        <p><strong>Interpretation:</strong> Measures how distribution Q differs from distribution P</p>
                        <p><strong>Properties:</strong></p>
                        <ul>
                            <li>Always non-negative: D_KL(P||Q) ≥ 0</li>
                            <li>Zero if and only if P = Q</li>
                            <li>NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)</li>
                        </ul>
                        <p><strong>Applications:</strong></p>
                        <ul>
                            <li><strong>Variational Autoencoders:</strong> Regularization term</li>
                            <li><strong>Model Comparison:</strong> Compare predicted vs true distributions</li>
                            <li><strong>Information Bottleneck:</strong> Compress representations</li>
                        </ul>
                        <p><strong>Cross-Entropy:</strong> $H(P,Q) = H(P) + D_{KL}(P||Q)$ → common loss function</p>
                        """
                    },
                    {
                        'name': 'Mutual Information',
                        'content': """
                        <h3>Mutual Information</h3>
                        <p><strong>Definition:</strong></p>
                        $$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$
                        <p><strong>Interpretation:</strong> Amount of information shared between variables</p>
                        <p><strong>Properties:</strong></p>
                        <ul>
                            <li>I(X;Y) ≥ 0 (zero for independent variables)</li>
                            <li>Symmetric: I(X;Y) = I(Y;X)</li>
                            <li>I(X;X) = H(X) (self-information is entropy)</li>
                        </ul>
                        <p><strong>Applications:</strong></p>
                        <ul>
                            <li><strong>Feature Selection:</strong> Select features with high mutual information with target</li>
                            <li><strong>Clustering:</strong> Measure cluster quality</li>
                            <li><strong>Causal Discovery:</strong> Identify dependencies</li>
                        </ul>
                        """
                    }
                ]
            }
        ]
    }

