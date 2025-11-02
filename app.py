from flask import Flask, render_template_string, request, jsonify, session
import pandas as pd
import numpy as np
import os
import json
from io import StringIO
import secrets

# Hardcoded Gemini API Key
GEMINI_API_KEY = ""

# Initialize Gemini
from google import genai
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', secrets.token_hex(32))

METRICS_CATALOG = [
    {
        "id": "accuracy",
        "name": "Accuracy",
        "meaning": "The proportion of correct predictions out of all predictions made.",
        "when_to_use": [
            "When classes are balanced",
            "When all types of errors are equally important",
            "For simple classification benchmarks"
        ],
        "pitfalls": [
            "Misleading on imbalanced datasets (e.g., 95% accuracy on 95% majority class)",
            "Doesn't distinguish between false positives and false negatives",
            "Can hide poor performance on minority classes"
        ],
        "formula": "accuracy = (TP + TN) / (TP + TN + FP + FN)",
        "sklearn_snippet": "from sklearn.metrics import accuracy_score\ny_true = [0, 1, 1, 0]\ny_pred = [0, 1, 0, 0]\naccuracy_score(y_true, y_pred)  # 0.75",
        "related": ["precision", "recall", "f1_score", "balanced_accuracy"]
    },
    {
        "id": "precision",
        "name": "Precision",
        "meaning": "Of all positive predictions, what proportion were actually correct?",
        "when_to_use": [
            "When false positives are costly (e.g., spam detection)",
            "When you want to be confident in positive predictions",
            "In information retrieval tasks"
        ],
        "pitfalls": [
            "Can be high even if you miss many true positives",
            "Doesn't account for false negatives",
            "Misleading when used alone without recall"
        ],
        "formula": "precision = TP / (TP + FP)",
        "sklearn_snippet": "from sklearn.metrics import precision_score\ny_true = [0, 1, 1, 0, 1]\ny_pred = [0, 1, 0, 0, 1]\nprecision_score(y_true, y_pred)  # 1.0",
        "related": ["recall", "f1_score", "accuracy", "average_precision"]
    },
    {
        "id": "recall",
        "name": "Recall (Sensitivity)",
        "meaning": "Of all actual positives, what proportion did we correctly identify?",
        "when_to_use": [
            "When false negatives are costly (e.g., disease detection)",
            "When it's critical to find all positive cases",
            "In fraud detection or anomaly detection"
        ],
        "pitfalls": [
            "Can be high by predicting everything as positive",
            "Doesn't account for false positives",
            "May lead to many false alarms if optimized alone"
        ],
        "formula": "recall = TP / (TP + FN)",
        "sklearn_snippet": "from sklearn.metrics import recall_score\ny_true = [0, 1, 1, 0, 1]\ny_pred = [0, 1, 1, 1, 1]\nrecall_score(y_true, y_pred)  # 1.0",
        "related": ["precision", "f1_score", "specificity", "sensitivity"]
    },
    {
        "id": "f1_score",
        "name": "F1 Score",
        "meaning": "The harmonic mean of precision and recall, balancing both metrics.",
        "when_to_use": [
            "When you need balance between precision and recall",
            "On imbalanced datasets",
            "When false positives and false negatives are both important"
        ],
        "pitfalls": [
            "Weights precision and recall equally (may not match business needs)",
            "Less interpretable than precision or recall alone",
            "Can be misleading on multiclass problems without micro/macro averaging"
        ],
        "formula": "F1 = 2 × (precision × recall) / (precision + recall)",
        "sklearn_snippet": "from sklearn.metrics import f1_score\ny_true = [0, 1, 1, 0, 1]\ny_pred = [0, 1, 0, 0, 1]\nf1_score(y_true, y_pred)  # 0.67",
        "related": ["precision", "recall", "fbeta_score", "accuracy"]
    },
    {
        "id": "roc_auc",
        "name": "ROC AUC",
        "meaning": "Area under the receiver operating characteristic curve; measures the model's ability to distinguish between classes across all thresholds.",
        "when_to_use": [
            "When you care about ranking quality",
            "To compare models independent of threshold",
            "On binary classification with probability outputs"
        ],
        "pitfalls": [
            "Can be misleading on highly imbalanced datasets",
            "Doesn't tell you the optimal threshold to use",
            "Treats all misclassification costs as equal"
        ],
        "formula": "AUC = ∫ TPR(FPR) d(FPR) from 0 to 1",
        "sklearn_snippet": "from sklearn.metrics import roc_auc_score\ny_true = [0, 0, 1, 1]\ny_scores = [0.1, 0.4, 0.35, 0.8]\nroc_auc_score(y_true, y_scores)  # 0.75",
        "related": ["precision_recall_curve", "average_precision", "log_loss"]
    },
    {
        "id": "mse",
        "name": "Mean Squared Error (MSE)",
        "meaning": "Average of squared differences between predicted and actual values.",
        "when_to_use": [
            "For regression tasks",
            "When large errors are particularly undesirable",
            "When you want to penalize outliers heavily"
        ],
        "pitfalls": [
            "Sensitive to outliers (squares the errors)",
            "Not in the same units as the target variable",
            "Hard to interpret the magnitude"
        ],
        "formula": "MSE = (1/n) × Σ(y_true - y_pred)²",
        "sklearn_snippet": "from sklearn.metrics import mean_squared_error\ny_true = [3, -0.5, 2, 7]\ny_pred = [2.5, 0.0, 2, 8]\nmean_squared_error(y_true, y_pred)  # 0.375",
        "related": ["rmse", "mae", "r2_score", "mape"]
    },
    {
        "id": "rmse",
        "name": "Root Mean Squared Error (RMSE)",
        "meaning": "Square root of MSE; brings the error metric back to the original units.",
        "when_to_use": [
            "For regression when you want interpretable units",
            "When large errors matter more than small ones",
            "As a more interpretable alternative to MSE"
        ],
        "pitfalls": [
            "Still sensitive to outliers",
            "Harder to differentiate between models than MSE",
            "May overemphasize large errors"
        ],
        "formula": "RMSE = √(MSE) = √((1/n) × Σ(y_true - y_pred)²)",
        "sklearn_snippet": "from sklearn.metrics import mean_squared_error\nimport numpy as np\ny_true = [3, -0.5, 2, 7]\ny_pred = [2.5, 0.0, 2, 8]\nnp.sqrt(mean_squared_error(y_true, y_pred))  # 0.612",
        "related": ["mse", "mae", "r2_score"]
    },
    {
        "id": "mae",
        "name": "Mean Absolute Error (MAE)",
        "meaning": "Average of absolute differences between predicted and actual values.",
        "when_to_use": [
            "When all errors should be weighted equally",
            "When outliers shouldn't be over-penalized",
            "For more robust regression metrics"
        ],
        "pitfalls": [
            "Doesn't distinguish between small and large errors as much",
            "May underweight important large errors",
            "Less commonly optimized in ML libraries"
        ],
        "formula": "MAE = (1/n) × Σ|y_true - y_pred|",
        "sklearn_snippet": "from sklearn.metrics import mean_absolute_error\ny_true = [3, -0.5, 2, 7]\ny_pred = [2.5, 0.0, 2, 8]\nmean_absolute_error(y_true, y_pred)  # 0.5",
        "related": ["mse", "rmse", "r2_score", "median_absolute_error"]
    },
    {
        "id": "r2_score",
        "name": "R² Score (Coefficient of Determination)",
        "meaning": "Proportion of variance in the target explained by the model. 1.0 is perfect, 0.0 means the model is no better than the mean.",
        "when_to_use": [
            "To measure overall regression model fit",
            "To compare models on the same dataset",
            "When you want a normalized metric (0 to 1 scale)"
        ],
        "pitfalls": [
            "Can be negative if the model is worse than a horizontal line",
            "Increases with more features (even irrelevant ones)",
            "Doesn't indicate if predictions are biased"
        ],
        "formula": "R² = 1 - (SS_res / SS_tot) where SS_res = Σ(y - ŷ)², SS_tot = Σ(y - ȳ)²",
        "sklearn_snippet": "from sklearn.metrics import r2_score\ny_true = [3, -0.5, 2, 7]\ny_pred = [2.5, 0.0, 2, 8]\nr2_score(y_true, y_pred)  # 0.948",
        "related": ["adjusted_r2", "mse", "mae"]
    },
    {
        "id": "log_loss",
        "name": "Log Loss (Cross-Entropy Loss)",
        "meaning": "Penalizes confident but wrong predictions more heavily; measures probability estimate quality.",
        "when_to_use": [
            "When you care about probability calibration",
            "For neural network training",
            "When confident wrong predictions are very bad"
        ],
        "pitfalls": [
            "Hard to interpret the absolute values",
            "Very sensitive to probabilities near 0 or 1",
            "Can be unstable with poor probability estimates"
        ],
        "formula": "LogLoss = -(1/n) × Σ[y log(p) + (1-y) log(1-p)]",
        "sklearn_snippet": "from sklearn.metrics import log_loss\ny_true = [0, 0, 1, 1]\ny_pred_proba = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]\nlog_loss(y_true, y_pred_proba)  # 0.173",
        "related": ["roc_auc", "brier_score", "cross_entropy"]
    }
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Metrics Explorer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f7;
            color: #1d1d1f;
            line-height: 1.6;
        }
        
        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 24px 0;
            margin-bottom: 24px;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        h1 {
            text-align: center;
            font-size: 32px;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            background: white;
            padding: 8px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        .tab-button {
            flex: 1;
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            color: #6e6e73;
        }
        
        .tab-button.active {
            background: #007aff;
            color: white;
        }
        
        .tab-button:hover:not(.active) {
            background: #f5f5f7;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        .search-box {
            width: 100%;
            padding: 14px 18px;
            font-size: 16px;
            border: 2px solid #e5e5e7;
            border-radius: 10px;
            margin-bottom: 16px;
            transition: border-color 0.2s;
        }
        
        .search-box:focus {
            outline: none;
            border-color: #007aff;
        }
        
        .metric-list {
            max-height: 300px;
            overflow-y: auto;
            border: 2px solid #e5e5e7;
            border-radius: 10px;
            background: #fafafa;
        }
        
        .metric-item {
            padding: 12px 16px;
            cursor: pointer;
            border-bottom: 1px solid #e5e5e7;
            transition: background 0.2s;
        }
        
        .metric-item:hover {
            background: #f0f0f2;
        }
        
        .metric-item:last-child {
            border-bottom: none;
        }
        
        .metric-detail {
            display: none;
        }
        
        .metric-detail.show {
            display: block;
        }
        
        .metric-header {
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 2px solid #e5e5e7;
        }
        
        .metric-name {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .metric-meaning {
            font-size: 18px;
            color: #6e6e73;
        }
        
        .section {
            margin: 20px 0;
        }
        
        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #1d1d1f;
        }
        
        ul {
            list-style: none;
            padding-left: 0;
        }
        
        ul li {
            padding: 6px 0 6px 20px;
            position: relative;
        }
        
        ul li:before {
            content: "•";
            position: absolute;
            left: 4px;
            color: #007aff;
            font-weight: bold;
        }
        
        .collapsible {
            background: #f5f5f7;
            padding: 10px 14px;
            border-radius: 8px;
            cursor: pointer;
            user-select: none;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .collapsible:hover {
            background: #e8e8ed;
        }
        
        .collapsible-content {
            display: none;
            margin-top: 8px;
        }
        
        .collapsible-content.show {
            display: block;
        }
        
        .formula-code {
            background: #f5f5f7;
            padding: 16px;
            border-radius: 8px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            overflow-x: auto;
        }
        
        .pills {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        
        .pill {
            background: #007aff;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .pill:hover {
            background: #0051d5;
        }
        
        .upload-area {
            border: 2px dashed #007aff;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: #f9f9fb;
        }
        
        .upload-area:hover {
            background: #f0f0f5;
        }
        
        .upload-area.dragover {
            border-color: #0051d5;
            background: #e8f4ff;
        }
        
        input[type="file"] {
            display: none;
        }
        
        select {
            width: 100%;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid #e5e5e7;
            border-radius: 10px;
            margin-bottom: 12px;
            background: white;
            cursor: pointer;
        }
        
        select:focus {
            outline: none;
            border-color: #007aff;
        }
        
        button {
            background: #007aff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        button:hover {
            background: #0051d5;
        }
        
        button:disabled {
            background: #c7c7cc;
            cursor: not-allowed;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }
        
        .stat-box {
            background: #f5f5f7;
            padding: 16px;
            border-radius: 10px;
        }
        
        .stat-label {
            font-size: 13px;
            color: #6e6e73;
            margin-bottom: 4px;
        }
        
        .stat-value {
            font-size: 20px;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        pre {
            background: #f5f5f7;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 13px;
        }
        
        .chat-container {
            height: calc(100vh - 400px);
            min-height: 400px;
            display: flex;
            flex-direction: column;
        }
        
        .floating-chat {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            border-top: 2px solid #e5e5e7;
            box-shadow: 0 -4px 12px rgba(0,0,0,0.1);
            z-index: 1000;
            padding: 12px 0;
        }
        
        .floating-chat-inner {
            max-width: 100%;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .chat-toggle {
            background: #007aff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .chat-toggle:hover {
            background: #0051d5;
        }
        
        .chat-expanded {
            display: none;
        }
        
        .chat-expanded.show {
            display: block;
        }
        
        .chat-log {
            max-height: 300px;
            overflow-y: auto;
            padding: 16px;
            background: #fafafa;
            border-radius: 12px;
            margin-bottom: 12px;
        }
        
        .chat-message {
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message.user {
            align-items: flex-end;
        }
        
        .chat-message.assistant {
            align-items: flex-start;
        }
        
        .chat-bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .chat-message.user .chat-bubble {
            background: #007aff;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .chat-message.assistant .chat-bubble {
            background: white;
            color: #1d1d1f;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        
        .chat-composer {
            position: sticky;
            bottom: 0;
            background: white;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.06);
            display: flex;
            gap: 12px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid #e5e5e7;
            border-radius: 24px;
            resize: none;
            font-family: inherit;
        }
        
        .chat-input:focus {
            outline: none;
            border-color: #007aff;
        }
        
        .send-button {
            padding: 12px 28px;
            border-radius: 24px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 12px;
            }
            
            .chat-bubble {
                max-width: 85%;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ML Metrics Explorer</h1>
        </header>
        
        <div class="tabs">
            <button class="tab-button active" onclick="switchTab('explore')">Explore Metrics</button>
            <button class="tab-button" onclick="switchTab('clarify')">Clarify (CSV + Chat)</button>
        </div>
        
        <div id="explore-tab" class="tab-content active">
            <div class="card">
                <input type="text" id="metric-search" class="search-box" placeholder="Search metrics..." oninput="filterMetrics()">
                <div id="metric-list" class="metric-list"></div>
            </div>
            
            <div id="metric-detail" class="card metric-detail"></div>
        </div>
        
        <div id="clarify-tab" class="tab-content">
            <div class="card">
                <div id="upload-area" class="upload-area" onclick="document.getElementById('csv-file').click()">
                    <p style="font-size: 18px; margin-bottom: 8px;">📊 Click or drag CSV file here</p>
                    <p style="color: #6e6e73; font-size: 14px;">Upload your dataset to analyze</p>
                    <input type="file" id="csv-file" accept=".csv" onchange="handleFileUpload(event)">
                </div>
            </div>
            
            <div id="column-selector" class="card" style="display: none;">
                <h3 style="margin-bottom: 16px;">Select Column to Analyze</h3>
                <select id="column-select" onchange="analyzeColumn()">
                    <option value="">-- Choose a column --</option>
                </select>
                <select id="target-select">
                    <option value="">-- Optional: Choose target column --</option>
                </select>
            </div>
            
            <div id="analysis-result" class="card" style="display: none;"></div>
            
            <div id="csv-chat-section" style="display: none;">
                <div class="card">
                    <div class="chat-container">
                        <div id="csv-chat-log" class="chat-log"></div>
                        <div class="chat-composer">
                            <textarea id="csv-chat-input" class="chat-input" placeholder="Ask a question about your data..." rows="1"></textarea>
                            <button class="send-button" onclick="sendCSVMessage()">Send</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="floating-chat">
        <div class="floating-chat-inner">
            <button class="chat-toggle" onclick="toggleChat()">💬 Ask AI about metrics</button>
            <div id="chat-expanded" class="chat-expanded">
                <div id="chat-log" class="chat-log">
                    <div style="text-align: center; color: #6e6e73; padding: 20px;">
                        Select a metric and ask questions about it!
                    </div>
                </div>
                <div class="chat-composer">
                    <textarea id="chat-input" class="chat-input" placeholder="Ask about the selected metric..." rows="1"></textarea>
                    <button class="send-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const metrics = {{ metrics_json | safe }};
        let currentColumn = null;
        let currentContext = null;
        let currentMetric = null;
        let chatExpanded = false;
        
        function switchTab(tab) {
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
        }
        
        function renderMetricList(filteredMetrics = metrics) {
            const list = document.getElementById('metric-list');
            list.innerHTML = filteredMetrics.map(m => 
                `<div class="metric-item" onclick="showMetric('${m.id}')">${m.name}</div>`
            ).join('');
        }
        
        function levenshteinDistance(a, b) {
            const matrix = [];
            
            for (let i = 0; i <= b.length; i++) {
                matrix[i] = [i];
            }
            
            for (let j = 0; j <= a.length; j++) {
                matrix[0][j] = j;
            }
            
            for (let i = 1; i <= b.length; i++) {
                for (let j = 1; j <= a.length; j++) {
                    if (b.charAt(i - 1) === a.charAt(j - 1)) {
                        matrix[i][j] = matrix[i - 1][j - 1];
                    } else {
                        matrix[i][j] = Math.min(
                            matrix[i - 1][j - 1] + 1,
                            matrix[i][j - 1] + 1,
                            matrix[i - 1][j] + 1
                        );
                    }
                }
            }
            
            return matrix[b.length][a.length];
        }
        
        function fuzzyScore(str, query) {
            str = str.toLowerCase();
            query = query.toLowerCase();
            
            if (str === query) return 1000;
            if (str.startsWith(query)) return 500;
            if (str.includes(query)) return 200;
            
            const distance = levenshteinDistance(str, query);
            const maxLength = Math.max(str.length, query.length);
            
            if (distance <= Math.ceil(query.length * 0.3)) {
                const similarity = 1 - (distance / maxLength);
                return Math.floor(similarity * 100);
            }
            
            const words = str.split(/\s+/);
            for (const word of words) {
                const wordDistance = levenshteinDistance(word, query);
                if (wordDistance <= Math.ceil(query.length * 0.3)) {
                    const similarity = 1 - (wordDistance / Math.max(word.length, query.length));
                    return Math.floor(similarity * 80);
                }
            }
            
            return 0;
        }
        
        function filterMetrics() {
            const query = document.getElementById('metric-search').value.toLowerCase().trim();
            
            if (!query) {
                renderMetricList(metrics);
                return;
            }
            
            const scoredMetrics = metrics.map(m => {
                const nameScore = fuzzyScore(m.name, query);
                const idScore = fuzzyScore(m.id, query);
                const meaningScore = fuzzyScore(m.meaning, query) * 0.5;
                const maxScore = Math.max(nameScore, idScore, meaningScore);
                
                return { metric: m, score: maxScore };
            }).filter(item => item.score > 0);
            
            scoredMetrics.sort((a, b) => b.score - a.score);
            
            renderMetricList(scoredMetrics.map(item => item.metric));
        }
        
        function showMetric(id) {
            const metric = metrics.find(m => m.id === id);
            if (!metric) return;
            
            currentMetric = metric;
            
            const detail = document.getElementById('metric-detail');
            detail.innerHTML = `
                <div class="metric-header">
                    <div class="metric-name">${metric.name}</div>
                    <div class="metric-meaning">${metric.meaning}</div>
                </div>
                
                <div class="section">
                    <div class="section-title">When to Use</div>
                    <ul>
                        ${metric.when_to_use.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
                
                <div class="section">
                    <div class="collapsible" onclick="toggleCollapsible(event)">
                        Top Pitfalls (click to expand)
                    </div>
                    <div class="collapsible-content">
                        <ul>
                            ${metric.pitfalls.map(item => `<li>${item}</li>`).join('')}
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">Formula</div>
                    <div class="formula-code">${metric.formula}</div>
                </div>
                
                <div class="section">
                    <div class="collapsible" onclick="toggleCollapsible(event)">
                        Scikit-learn Example (click to expand)
                    </div>
                    <div class="collapsible-content">
                        <pre>${metric.sklearn_snippet}</pre>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">Related Metrics</div>
                    <div class="pills">
                        ${metric.related.map(r => `<div class="pill" onclick="showMetricByName('${r}')">${r.replace(/_/g, ' ')}</div>`).join('')}
                    </div>
                </div>
            `;
            detail.classList.add('show');
            detail.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        function showMetricByName(name) {
            const metric = metrics.find(m => m.id === name || m.name.toLowerCase() === name.toLowerCase());
            if (metric) showMetric(metric.id);
        }
        
        function toggleCollapsible(event) {
            const content = event.target.nextElementSibling;
            content.classList.toggle('show');
        }
        
        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();
            
            if (data.success) {
                document.getElementById('column-select').innerHTML = 
                    '<option value="">-- Choose a column --</option>' +
                    data.columns.map(col => `<option value="${col}">${col}</option>`).join('');
                document.getElementById('target-select').innerHTML = 
                    '<option value="">-- Optional: Choose target column --</option>' +
                    data.columns.map(col => `<option value="${col}">${col}</option>`).join('');
                document.getElementById('column-selector').style.display = 'block';
            }
        }
        
        async function analyzeColumn() {
            const column = document.getElementById('column-select').value;
            const target = document.getElementById('target-select').value;
            if (!column) return;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ column, target })
                });
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    console.error('Analysis error:', data.error);
                    return;
                }
                
                currentColumn = column;
                currentContext = data;
                
                let html = `<h3>Analysis: ${column}</h3>`;
                
                const statsHtml = `
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Non-null Count</div>
                            <div class="stat-value">${data.non_null_count}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Missing %</div>
                            <div class="stat-value">${data.missing_pct.toFixed(1)}%</div>
                        </div>
                        ${data.dtype === 'numeric' ? `
                            <div class="stat-box">
                                <div class="stat-label">Mean</div>
                                <div class="stat-value">${data.mean.toFixed(2)}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Std Dev</div>
                                <div class="stat-value">${data.std.toFixed(2)}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Min</div>
                                <div class="stat-value">${data.min.toFixed(2)}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">25th %ile</div>
                                <div class="stat-value">${data.p25.toFixed(2)}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Median</div>
                                <div class="stat-value">${data.median.toFixed(2)}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">75th %ile</div>
                                <div class="stat-value">${data.p75.toFixed(2)}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Max</div>
                                <div class="stat-value">${data.max.toFixed(2)}</div>
                            </div>
                        ` : `
                            <div class="stat-box">
                                <div class="stat-label">Unique Values</div>
                                <div class="stat-value">${data.unique_count}</div>
                            </div>
                        `}
                    </div>
                `;
                
                html += statsHtml;
                
                if (data.dtype === 'categorical' && data.top_categories) {
                    html += `
                        <div class="section">
                            <div class="section-title">Top 10 Categories</div>
                            <ul>
                                ${data.top_categories.map(cat => 
                                    `<li>${cat.value}: ${cat.count} (${cat.pct.toFixed(1)}%)</li>`
                                ).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                if (data.target_analysis) {
                    html += `
                        <div class="section">
                            <div class="section-title">Target Relationship (${target})</div>
                            ${data.target_analysis}
                        </div>
                    `;
                }
                
                if (data.preview) {
                    html += `
                        <div class="section">
                            <div class="section-title">Preview (First 5 Rows)</div>
                            <pre>${data.preview}</pre>
                        </div>
                    `;
                }
                
                document.getElementById('analysis-result').innerHTML = html;
                document.getElementById('analysis-result').style.display = 'block';
                document.getElementById('csv-chat-section').style.display = 'block';
            } catch (error) {
                console.error('Error in analyzeColumn:', error);
                alert('Error analyzing column: ' + error.message);
            }
        }
        
        function toggleChat() {
            chatExpanded = !chatExpanded;
            const expanded = document.getElementById('chat-expanded');
            const button = document.querySelector('.chat-toggle');
            
            if (chatExpanded) {
                expanded.classList.add('show');
                button.textContent = '✕ Close Chat';
            } else {
                expanded.classList.remove('show');
                button.textContent = '💬 Ask AI about metrics';
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            if (!currentMetric) {
                addMessageToChat('assistant', 'Please select a metric first to get context-aware help!');
                return;
            }
            
            addMessageToChat('user', message);
            input.value = '';
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message, 
                    metric: currentMetric
                })
            });
            const data = await response.json();
            
            addMessageToChat('assistant', data.response);
        }
        
        async function sendCSVMessage() {
            const input = document.getElementById('csv-chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            addCSVMessageToChat('user', message);
            input.value = '';
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message, 
                    column: currentColumn,
                    context: currentContext
                })
            });
            const data = await response.json();
            
            addCSVMessageToChat('assistant', data.response);
        }
        
        function addMessageToChat(role, content) {
            const chatLog = document.getElementById('chat-log');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${role}`;
            messageDiv.innerHTML = `<div class="chat-bubble">${content}</div>`;
            chatLog.appendChild(messageDiv);
            chatLog.scrollTop = chatLog.scrollHeight;
        }
        
        function addCSVMessageToChat(role, content) {
            const chatLog = document.getElementById('csv-chat-log');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${role}`;
            messageDiv.innerHTML = `<div class="chat-bubble">${content}</div>`;
            chatLog.appendChild(messageDiv);
            chatLog.scrollTop = chatLog.scrollHeight;
        }
        
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        document.getElementById('csv-chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendCSVMessage();
            }
        });
        
        const uploadArea = document.getElementById('upload-area');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) {
                document.getElementById('csv-file').files = e.dataTransfer.files;
                handleFileUpload({ target: { files: [file] } });
            }
        });
        
        renderMetricList();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, metrics_json=json.dumps(METRICS_CATALOG))

@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        
        session['csv_data'] = content
        session['columns'] = df.columns.tolist()
        
        return jsonify({
            'success': True,
            'columns': df.columns.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze_column():
    try:
        data = request.json
        column = data['column']
        target = data.get('target', '')
        
        csv_content = session.get('csv_data')
        df = pd.read_csv(StringIO(csv_content))
        
        col_data = df[column]
        result = {
            'column': column,
            'non_null_count': int(col_data.count()),
            'missing_pct': float((col_data.isna().sum() / len(col_data)) * 100)
        }
        
        # Check if column is numeric or can be converted
        if pd.api.types.is_numeric_dtype(col_data):
            result['dtype'] = 'numeric'
            result['mean'] = float(col_data.mean())
            result['std'] = float(col_data.std())
            result['min'] = float(col_data.min())
            result['p25'] = float(col_data.quantile(0.25))
            result['median'] = float(col_data.median())
            result['p75'] = float(col_data.quantile(0.75))
            result['max'] = float(col_data.max())
        else:
            # Try to convert to numeric
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if numeric_data.notna().sum() > 0:  # At least some numeric values
                    result['dtype'] = 'numeric'
                    result['mean'] = float(numeric_data.mean())
                    result['std'] = float(numeric_data.std())
                    result['min'] = float(numeric_data.min())
                    result['p25'] = float(numeric_data.quantile(0.25))
                    result['median'] = float(numeric_data.median())
                    result['p75'] = float(numeric_data.quantile(0.75))
                    result['max'] = float(numeric_data.max())
                else:
                    result['dtype'] = 'categorical'
                    result['unique_count'] = int(col_data.nunique())
                    
                    value_counts = col_data.value_counts()
                    top_10 = value_counts.head(10)
                    total = len(col_data)
                    result['top_categories'] = [
                        {'value': str(val), 'count': int(count), 'pct': float((count / total) * 100)}
                        for val, count in top_10.items()
                    ]
            except:
                result['dtype'] = 'categorical'
                result['unique_count'] = int(col_data.nunique())
                
                value_counts = col_data.value_counts()
                top_10 = value_counts.head(10)
                total = len(col_data)
                result['top_categories'] = [
                    {'value': str(val), 'count': int(count), 'pct': float((count / total) * 100)}
                    for val, count in top_10.items()
                ]
        
        if target and target in df.columns and target != column:
            target_data = df[target]
            
            try:
                # Check if target is numeric
                if pd.api.types.is_numeric_dtype(target_data):
                    target_numeric = target_data
                else:
                    target_numeric = pd.to_numeric(target_data, errors='coerce')
                
                # Only proceed if we have valid numeric data
                if target_numeric.notna().sum() > 0:
                    if result['dtype'] == 'numeric':
                        # Both columns numeric - calculate correlation
                        combined = pd.DataFrame({column: col_data, target: target_numeric}).dropna()
                        if len(combined) > 1:
                            corr = combined[column].corr(combined[target])
                            result['target_analysis'] = f"Pearson correlation: {corr:.3f}"
                    else:
                        # Categorical column vs numeric target - group means
                        combined = pd.DataFrame({column: col_data, target: target_numeric}).dropna()
                        if len(combined) > 0:
                            group_means = combined.groupby(column)[target].mean().sort_values(ascending=False).head(5)
                            analysis = "<ul>"
                            for val, mean in group_means.items():
                                analysis += f"<li>{val}: {mean:.2f}</li>"
                            analysis += "</ul>"
                            result['target_analysis'] = analysis
            except Exception as e:
                print(f"Target analysis error: {e}")
                pass
        
        preview = df[[column]].head(5).to_string()
        result['preview'] = preview
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data['message']
        column = data.get('column')
        context = data.get('context')
        metric = data.get('metric')
        
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({'role': 'user', 'content': message})
        
        try:
            context_str = ""
            if metric:
                context_str = f"\n\nContext: User is asking about the '{metric['name']}' metric. "
                context_str += f"Definition: {metric['meaning']} "
                context_str += f"Formula: {metric['formula']} "
                context_str += f"When to use: {', '.join(metric['when_to_use'][:2])}. "
            elif column and context:
                context_str = f"\n\nContext: Analyzing column '{column}'. "
                if context.get('dtype') == 'numeric':
                    context_str += f"Numeric column with mean={context.get('mean', 0):.2f}, std={context.get('std', 0):.2f}. "
                else:
                    context_str += f"Categorical column with {context.get('unique_count', 0)} unique values. "
            
            history_str = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in session['chat_history'][-5:]
            ])
            
            prompt = f"You are a helpful ML metrics and data science assistant. Keep responses under 180 words, be grounded and practical.{context_str}\n\nConversation:\n{history_str}\n\nProvide a concise, helpful response."
            
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            assistant_message = response.text
        except Exception as e:
            assistant_message = f"Error: {str(e)}"
        
        session['chat_history'].append({'role': 'assistant', 'content': assistant_message})
        session.modified = True
        
        return jsonify({'response': assistant_message})
    except Exception as e:
        return jsonify({'response': f'Error processing request: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
