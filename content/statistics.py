"""
Statistics & Probability Content Module
Foundational statistical concepts for data science
"""

def get_content():
    return {
        'summary': """
        <div class="topic-summary">
            <h3>📋 What is Statistics & Probability?</h3>
            <p>Statistics and probability provide the mathematical foundation for analyzing data, measuring uncertainty, and making informed decisions. These tools help us understand patterns, test hypotheses, and quantify confidence in our conclusions.</p>
            <p><strong>Used in:</strong> A/B Testing, Survey Analysis, Risk Assessment, Quality Control, Clinical Trials, and all Machine Learning model evaluation.</p>
        </div>
        """,
        'topics': [
            {
                'id': 'descriptive-stats',
                'title': 'Descriptive Statistics',
                'description': 'Summarizing and describing data characteristics',
                'subtopics': [
                    {
                        'name': 'Measures of Central Tendency',
                        'content': """
                        <h3>Central Tendency Metrics</h3>
                        <p><strong>Mean (Average):</strong> Sum of values divided by count → sensitive to outliers</p>
                        <p><strong>Median:</strong> Middle value when sorted → robust to outliers</p>
                        <p><strong>Mode:</strong> Most frequently occurring value</p>
                        <div class="example">
                            <h4>Python Example</h4>
                            <pre><code>import numpy as np

data = [10, 20, 20, 30, 100]
mean = np.mean(data)      # 36
median = np.median(data)  # 20
# Mode: 20 (appears twice)</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Measures of Dispersion',
                        'content': """
                        <h3>Variability Metrics</h3>
                        <p><strong>Variance:</strong> Average squared deviation from mean</p>
                        <p><strong>Standard Deviation:</strong> Square root of variance, same units as data</p>
                        <p><strong>Range:</strong> Difference between max and min values</p>
                        <p><strong>Interquartile Range (IQR):</strong> Q3 - Q1, measures middle 50% spread</p>
                        <p><strong>Coefficient of Variation:</strong> (StdDev / Mean) × 100 for relative variability</p>
                        <div class="visual">
                            <img src="/static/images/boxplot.png" alt="Box Plot Visualization" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Distribution Shape',
                        'content': """
                        <h3>Distribution Characteristics</h3>
                        <p><strong>Skewness:</strong> Asymmetry of distribution</p>
                        <ul>
                            <li>Positive skew: Right tail is longer</li>
                            <li>Negative skew: Left tail is longer</li>
                            <li>Zero skew: Symmetric distribution</li>
                        </ul>
                        <p><strong>Kurtosis:</strong> Tailedness and peakedness</p>
                        <ul>
                            <li>High kurtosis: Heavy tails, sharp peak</li>
                            <li>Low kurtosis: Light tails, flat peak</li>
                        </ul>
                        """
                    }
                ]
            },
            {
                'id': 'probability-distributions',
                'title': 'Probability Distributions',
                'description': 'Key statistical distributions and their applications',
                'subtopics': [
                    {
                        'name': 'Discrete Distributions',
                        'content': """
                        <h3>Discrete Probability Distributions</h3>
                        <p><strong>Bernoulli:</strong> Single binary trial (success/failure)</p>
                        <p><strong>Binomial:</strong> Number of successes in n independent trials</p>
                        <ul>
                            <li>Parameters: n (trials), p (success probability)</li>
                            <li>Application: Quality control, A/B testing</li>
                        </ul>
                        <p><strong>Poisson:</strong> Count of events in fixed interval</p>
                        <ul>
                            <li>Parameter: λ (average rate)</li>
                            <li>Application: Customer arrivals, rare events</li>
                        </ul>
                        <p><strong>Geometric:</strong> Trials until first success</p>
                        """
                    },
                    {
                        'name': 'Continuous Distributions',
                        'content': """
                        <h3>Continuous Probability Distributions</h3>
                        <p><strong>Normal (Gaussian):</strong> Bell-shaped, symmetric distribution</p>
                        <ul>
                            <li>Parameters: μ (mean), σ (standard deviation)</li>
                            <li>68-95-99.7 rule for standard deviations</li>
                            <li>Foundation of many statistical tests</li>
                        </ul>
                        <p><strong>Exponential:</strong> Time between events in Poisson process</p>
                        <p><strong>Uniform:</strong> Equal probability across range</p>
                        <p><strong>t-Distribution:</strong> Similar to normal, heavier tails (small samples)</p>
                        <p><strong>Chi-Square:</strong> Sum of squared normal variables (goodness-of-fit tests)</p>
                        <div class="visual">
                            <img src="/static/images/normal_distribution.png" alt="Normal Distribution" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Central Limit Theorem',
                        'content': """
                        <h3>Central Limit Theorem (CLT)</h3>
                        <p><strong>Principle:</strong> Sample means of any distribution approach normal distribution 
                        as sample size increases</p>
                        <p><strong>Implications:</strong></p>
                        <ul>
                            <li>Enables inference about population from samples</li>
                            <li>Justifies normal-based confidence intervals</li>
                            <li>Works regardless of underlying distribution</li>
                        </ul>
                        <p><strong>Rule of Thumb:</strong> Sample size ≥ 30 typically sufficient</p>
                        """
                    }
                ]
            },
            {
                'id': 'hypothesis-testing',
                'title': 'Hypothesis Testing',
                'description': 'Statistical inference and decision making',
                'subtopics': [
                    {
                        'name': 'Fundamentals',
                        'content': """
                        <h3>Hypothesis Testing Framework</h3>
                        <p><strong>Null Hypothesis (H₀):</strong> Default assumption of no effect</p>
                        <p><strong>Alternative Hypothesis (H₁):</strong> Claim we're testing for</p>
                        <p><strong>p-value:</strong> Probability of observing results assuming H₀ is true</p>
                        <ul>
                            <li>p < 0.05: Reject null hypothesis (typically)</li>
                            <li>p ≥ 0.05: Fail to reject null hypothesis</li>
                        </ul>
                        <p><strong>Significance Level (α):</strong> Threshold for rejection (commonly 0.05)</p>
                        <p><strong>Type I Error:</strong> Rejecting true null hypothesis (false positive)</p>
                        <p><strong>Type II Error:</strong> Failing to reject false null hypothesis (false negative)</p>
                        """
                    },
                    {
                        'name': 'Common Tests',
                        'content': """
                        <h3>Statistical Tests</h3>
                        <p><strong>t-test:</strong> Compare means of two groups</p>
                        <ul>
                            <li>One-sample: Compare sample mean to population value</li>
                            <li>Two-sample: Compare means of two independent groups</li>
                            <li>Paired: Compare before/after measurements</li>
                        </ul>
                        <p><strong>ANOVA:</strong> Compare means across multiple groups</p>
                        <p><strong>Chi-Square Test:</strong> Test independence between categorical variables</p>
                        <p><strong>Mann-Whitney U:</strong> Non-parametric alternative to t-test</p>
                        <p><strong>Kolmogorov-Smirnov:</strong> Test distribution fit</p>
                        """
                    }
                ]
            },
            {
                'id': 'correlation-causation',
                'title': 'Correlation vs Causation',
                'description': 'Understanding relationships between variables',
                'subtopics': [
                    {
                        'name': 'Correlation Analysis',
                        'content': """
                        <h3>Measuring Correlation</h3>
                        <p><strong>Pearson Correlation (r):</strong> Linear relationship strength (-1 to +1)</p>
                        <ul>
                            <li>r = +1: Perfect positive correlation</li>
                            <li>r = 0: No linear correlation</li>
                            <li>r = -1: Perfect negative correlation</li>
                        </ul>
                        <p><strong>Spearman Rank Correlation:</strong> Monotonic relationship, robust to outliers</p>
                        <p><strong>Kendall Tau:</strong> Concordance between rankings</p>
                        <p><strong>Warning:</strong> Correlation does NOT imply causation!</p>
                        <div class="visual">
                            <img src="/static/images/correlation_heatmap.png" alt="Correlation Heatmap" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Establishing Causation',
                        'content': """
                        <h3>Causal Inference</h3>
                        <p><strong>Requirements for Causation:</strong></p>
                        <ul>
                            <li><strong>Temporal Precedence:</strong> Cause must precede effect</li>
                            <li><strong>Covariation:</strong> Cause and effect must be correlated</li>
                            <li><strong>No Confounders:</strong> Rule out alternative explanations</li>
                        </ul>
                        <p><strong>Methods:</strong></p>
                        <ul>
                            <li><strong>Randomized Controlled Trials (RCTs):</strong> Gold standard</li>
                            <li><strong>Instrumental Variables:</strong> Exploit exogenous variation</li>
                            <li><strong>Propensity Score Matching:</strong> Balance treatment groups</li>
                            <li><strong>Difference-in-Differences:</strong> Before/after analysis</li>
                        </ul>
                        """
                    }
                ]
            },
            {
                'id': 'confidence-intervals',
                'title': 'Confidence Intervals & Sampling',
                'description': 'Quantifying uncertainty in estimates',
                'subtopics': [
                    {
                        'name': 'Confidence Intervals',
                        'content': """
                        <h3>Confidence Intervals</h3>
                        <p><strong>Definition:</strong> Range likely to contain true population parameter</p>
                        <p><strong>95% Confidence Interval:</strong> If we repeated sampling 100 times, 
                        95 intervals would contain true parameter</p>
                        <p><strong>Formula (Mean):</strong></p>
                        $$\\bar{x} \\pm (t \\times SE)$$
                        <ul>
                            <li>$\\bar{x}$: Sample mean</li>
                            <li>$t$: Critical value from t-distribution</li>
                            <li>$SE$: Standard error $\\left(\\frac{\\sigma}{\\sqrt{n}}\\right)$</li>
                        </ul>
                        <p><strong>Interpretation:</strong> Wider interval = more uncertainty, narrower = more precision</p>
                        """
                    },
                    {
                        'name': 'Sampling Methods',
                        'content': """
                        <h3>Sampling Techniques</h3>
                        <p><strong>Simple Random Sampling:</strong> Each member has equal selection probability</p>
                        <p><strong>Stratified Sampling:</strong> Divide population into strata, sample from each</p>
                        <p><strong>Cluster Sampling:</strong> Sample entire clusters/groups</p>
                        <p><strong>Systematic Sampling:</strong> Select every kth element</p>
                        <p><strong>Bootstrap Sampling:</strong> Resample with replacement for uncertainty estimation</p>
                        <p><strong>Sampling Bias:</strong> Selection bias, non-response bias, survivorship bias</p>
                        """
                    }
                ]
            }
        ]
    }

