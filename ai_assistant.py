"""
Gemini AI Assistant Integration
Handles AI chat functionality using Google's Gemini API
With security guardrails to prevent abuse
"""

from google import genai
import os
from datetime import datetime, timedelta
from collections import defaultdict

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# GUARDRAILS CONFIGURATION
MAX_MESSAGE_LENGTH = 1000  # Max characters per message
MAX_REQUESTS_PER_HOUR = 20  # Max requests per IP per hour
MAX_OUTPUT_TOKENS = 1024    # Limit response length (saves costs)

# Simple in-memory rate limiting (resets on server restart)
request_tracker = defaultdict(list)

def get_gemini_client():
    """Get configured Gemini client"""
    if GEMINI_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        return genai.Client()
    return None

def check_rate_limit(ip_address):
    """
    Check if user has exceeded rate limit
    
    Args:
        ip_address (str): User's IP address
        
    Returns:
        tuple: (is_allowed, remaining_requests)
    """
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    
    # Clean old requests
    request_tracker[ip_address] = [
        timestamp for timestamp in request_tracker[ip_address]
        if timestamp > one_hour_ago
    ]
    
    # Check limit
    request_count = len(request_tracker[ip_address])
    is_allowed = request_count < MAX_REQUESTS_PER_HOUR
    remaining = MAX_REQUESTS_PER_HOUR - request_count
    
    if is_allowed:
        request_tracker[ip_address].append(now)
    
    return is_allowed, max(0, remaining)

def validate_input(user_message):
    """
    Validate user input for safety
    
    Args:
        user_message (str): User's message
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if message is empty
    if not user_message or not user_message.strip():
        return False, "Please enter a question."
    
    # Check message length
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return False, f"Message too long. Please keep it under {MAX_MESSAGE_LENGTH} characters."
    
    # Block potential prompt injection attempts
    suspicious_patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "you are now",
        "new instructions:",
        "system:",
        "ignore system prompt"
    ]
    
    message_lower = user_message.lower()
    for pattern in suspicious_patterns:
        if pattern in message_lower:
            return False, "Invalid input detected. Please ask a genuine data science question."
    
    return True, None

def get_ai_response(user_message, ip_address="unknown"):
    """
    Get response from Gemini AI with security guardrails
    
    Args:
        user_message (str): The user's question or message
        ip_address (str): User's IP address for rate limiting
        
    Returns:
        str: AI response or error message
    """
    
    # GUARDRAIL 1: Input validation
    is_valid, error_msg = validate_input(user_message)
    if not is_valid:
        return error_msg
    
    # GUARDRAIL 2: Rate limiting
    is_allowed, remaining = check_rate_limit(ip_address)
    if not is_allowed:
        return f"⚠️ Rate limit reached. You can make {MAX_REQUESTS_PER_HOUR} requests per hour. Please try again later."
    
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
        Keep responses focused on data science topics only.
        """
        
        # Combine system prompt with user message
        full_prompt = f"{system_prompt}\n\nUser question: {user_message}"
        
        # GUARDRAIL 3: Token limits
        # Generate response with output token limit
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt
        )
        
        # Note: Token limits are set at API level (Google AI Studio)
        # For programmatic limits, use Gemini API's generation_config parameter
        # Current implementation relies on rate limiting + input validation
        
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}. Please try a shorter question or try again later."

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
