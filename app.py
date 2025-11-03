"""
Data Science Encyclopedia - Main Application
A comprehensive educational platform for data science learning
"""

from flask import Flask, render_template, jsonify, request
from content import machine_learning, statistics, mathematics, data_cleaning, visualization, ai_deep_learning
from ai_assistant import get_ai_response

app = Flask(__name__)

# Content mapping with buzzword-rich descriptions
SECTIONS = {
    'machine-learning': {
        'title': '🧠 Machine Learning',
        'icon': '🧠',
        'content': machine_learning.get_content()
    },
    'statistics': {
        'title': '📊 Statistics & Probability',
        'icon': '📊',
        'content': statistics.get_content()
    },
    'mathematics': {
        'title': '🧮 Mathematics for Data Science',
        'icon': '🧮',
        'content': mathematics.get_content()
    },
    'data-cleaning': {
        'title': '🧼 Data Cleaning & Feature Engineering',
        'icon': '🧼',
        'content': data_cleaning.get_content()
    },
    'visualization': {
        'title': '📈 Data Visualization',
        'icon': '📈',
        'content': visualization.get_content()
    },
    'ai-deep-learning': {
        'title': '🔮 AI, Deep Learning & NLP',
        'icon': '🔮',
        'content': ai_deep_learning.get_content()
    }
}


@app.route('/')
def index():
    """Main encyclopedia interface"""
    return render_template('index.html', sections=SECTIONS)


@app.route('/api/section/<section_id>')
def get_section(section_id):
    """API endpoint for dynamic content loading"""
    if section_id in SECTIONS:
        return jsonify(SECTIONS[section_id])
    return jsonify({'error': 'Section not found'}), 404


@app.route('/api/ai-chat', methods=['POST'])
def ai_chat():
    """API endpoint for AI chat assistant powered by Gemini"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get AI response from Gemini
        response = get_ai_response(user_message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5002))  # use 5002 locally, $PORT on Render
    app.run(debug=True, host="0.0.0.0", port=port)

