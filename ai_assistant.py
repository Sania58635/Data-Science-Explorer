"""
Gemini AI Assistant Integration
Handles AI chat functionality using Google's Gemini API
"""

from google import genai
import os

def get_gemini_client():
    """Get configured Gemini client"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    return genai.Client()

def get_ai_response(user_message):
    """
    Get response from Gemini AI
    
    Args:
        user_message (str): The user's question or message
        
    Returns:
        str: AI response
    """
    
    # Get client
    client = get_gemini_client()
    if not client:
        return "Please configure your Gemini API key in ai_assistant.py"
    
    try:
        # Create a data science focused system prompt
        system_prompt = """You are an expert Data Science AI assistant. You help students and practitioners understand:
        - Machine Learning algorithms and techniques
        - Statistics and Probability concepts
        - Mathematics for Data Science (Linear Algebra, Calculus, Optimization)
        - Data Cleaning and Feature Engineering
        - Data Visualization best practices
        - AI, Deep Learning, and NLP
        
        Provide clear, concise explanations with examples when helpful.
        Use markdown formatting for code blocks with triple backticks.
        Be friendly, encouraging, and educational.
        """
        
        # Combine system prompt with user message
        full_prompt = f"{system_prompt}\n\nUser question: {user_message}"
        
        # Generate response using latest API
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt
        )
        
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}. Please check your API key and try again."

def get_code_explanation(code_snippet, language="python"):
    """
    Get explanation for a code snippet
    
    Args:
        code_snippet (str): The code to explain
        language (str): Programming language
        
    Returns:
        str: Explanation of the code
    """
    
    client = get_gemini_client()
    if not client:
        return "Please configure your Gemini API key in ai_assistant.py"
    
    try:
        prompt = f"""Explain this {language} code in the context of data science:

```{language}
{code_snippet}
```

Provide:
1. What the code does
2. Key concepts used
3. Practical application in data science
"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"Error explaining code: {str(e)}"

def get_concept_explanation(concept, detail_level="medium"):
    """
    Get detailed explanation of a data science concept
    
    Args:
        concept (str): The concept to explain
        detail_level (str): "brief", "medium", or "detailed"
        
    Returns:
        str: Explanation of the concept
    """
    
    client = get_gemini_client()
    if not client:
        return "Please configure your Gemini API key in ai_assistant.py"
    
    try:
        detail_instructions = {
            "brief": "Provide a concise 2-3 sentence explanation.",
            "medium": "Provide a clear explanation with an example.",
            "detailed": "Provide a comprehensive explanation with examples, use cases, and best practices."
        }
        
        prompt = f"""Explain the data science concept: {concept}

{detail_instructions.get(detail_level, detail_instructions["medium"])}

Include:
- Core definition
- Practical applications
- When to use it
"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"Error explaining concept: {str(e)}"

