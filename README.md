# ML Metrics Explorer

**Last Updated:** November 2, 2025

## Overview

A Flask-based web application for exploring machine learning metrics and analyzing CSV datasets with AI-powered chat assistance. Features a clean, Apple-inspired light theme with intuitive navigation and interactive components.

## Features

### 1. Explore Metrics Tab
- **Built-in Catalog**: 10+ common ML metrics (accuracy, precision, recall, F1, ROC AUC, MSE, RMSE, MAE, R², log loss)
- **Fuzzy Search**: Levenshtein distance-based search that handles typos (e.g., "accuarcy" finds "accuracy", "precsion" finds "precision")
  - Scoring: Exact match (1000), starts with (500), contains (200), edit distance similarity (0-100), word matches (0-80)
  - Tolerates up to 30% edit distance for typo correction
- **Detailed Metric View**:
  - Plain-English meaning
  - When-to-use bullet points
  - Top pitfalls (collapsible section)
  - LaTeX-style formula displayed as code
  - Scikit-learn code snippet (collapsible)
  - Related metrics as clickable pills

### 2. Clarify (CSV + Chat) Tab
- **CSV Upload**: Drag-and-drop or click to upload CSV files
- **Column Analysis**:
  - Automatic dtype detection (numeric vs categorical)
  - Numeric stats: mean, std, min, p25, median, p75, max
  - Categorical stats: unique count, top 10 categories with percentages
  - Missing data analysis (non-null count, missing %)
- **Target Correlation**:
  - Numeric ↔ Numeric: Pearson correlation coefficient
  - Categorical → Numeric: Top 5 groups by target mean
- **Preview Table**: First 5 rows in ASCII format
- **ChatGPT-Style Interface**:
  - Sticky composer bar at bottom
  - Scrollable chat log with user/assistant bubbles
  - Context-aware responses using column analysis data
  - Gemini AI integration (with fallback when API key not set)
  - Per-session chat history

## Technical Stack

### Backend
- **Flask**: Web framework
- **Pandas**: CSV processing and data analysis
- **NumPy**: Statistical computations
- **google-generativeai**: Optional Gemini AI integration

### Frontend
- **Vanilla HTML/CSS/JavaScript**: No external UI frameworks
- **Responsive Design**: Works on mobile and desktop
- **Light Theme**: Apple-inspired with rounded cards, clean spacing

## Project Structure

```
.
├── app.py              # Single-file Flask application (all code)
├── .gitignore          # Python ignore patterns
├── replit.md           # This documentation
├── pyproject.toml      # Python dependencies
└── uv.lock             # Locked dependencies
```

## Configuration

### Environment Variables

- `SESSION_SECRET`: Flask session secret (auto-generated if not set)
- `GEMINI_API_KEY`: Optional - enables AI-powered chat responses (graceful fallback without it)

### Running the App

```bash
python app.py
```

The app runs on `http://0.0.0.0:5000` by default.

## Architecture Decisions

**Single-File Design**: All HTML, CSS, JavaScript, and Python code in one `app.py` file for simplicity and portability.

**Session-Based State**: CSV data and chat history stored in Flask sessions for simplicity (suitable for single-user development use).

**Graceful Degradation**: Chat works with local fallback responses when Gemini API key is not configured.

**No External CSS/JS**: All styling and interactivity embedded in the HTML template for minimal dependencies.

## Usage Guide

### Exploring Metrics
1. Type in the search box to filter metrics (e.g., "precision", "regression")
2. Click any metric to view details
3. Click related metric pills to navigate between metrics
4. Expand collapsible sections for pitfalls and code examples

### Analyzing CSV Data
1. Switch to "Clarify (CSV + Chat)" tab
2. Upload a CSV file (drag-and-drop or click)
3. Select a column to analyze
4. Optionally select a target column for correlation analysis
5. Review the statistics and preview
6. Ask questions in the chat about your data

### Chat Tips
- Ask about patterns, correlations, or which metrics to use
- Reference specific columns or statistics from the analysis
- Keep questions focused for better responses (<180 words)

## Recent Changes

**November 2, 2025**: Initial implementation
- Built single-file Flask app with embedded HTML/CSS/JS (1107 lines)
- Implemented Explore Metrics tab with 10 common ML metrics
- Implemented Levenshtein distance-based fuzzy search with typo tolerance
- Implemented Clarify tab with CSV upload, drag-and-drop support, and comprehensive analysis
- Added automatic dtype detection (numeric vs categorical) with appropriate statistics
- Implemented target correlation analysis (Pearson r for numeric, group means for categorical)
- Added ChatGPT-style chat interface with Gemini AI integration (graceful fallback)
- Created clean Apple-like light theme with rounded cards, good spacing, and responsive design
- Configured workflow to run on port 5000 with webview output

## Future Enhancements

- Add metric comparison feature (side-by-side view)
- Implement CSV data visualization with charts
- Export functionality for chat conversations
- Metric favorites/bookmarking system
- Conversation branching and metric references in chat
