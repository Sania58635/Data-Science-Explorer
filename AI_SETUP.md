# AI Assistant Setup Guide

## 🤖 Gemini AI Integration

Your Data Science Encyclopedia now includes an AI chat assistant powered by Google's Gemini API, displayed in a beautiful side panel just like Cursor!

## 🔑 Getting Your Gemini API Key

1. **Visit Google AI Studio:**
   Go to: https://makersuite.google.com/app/apikey

2. **Sign in with your Google account**

3. **Create an API key:**
   - Click "Get API key"
   - Click "Create API key in new project" (or select existing project)
   - Copy your API key

4. **Add the API key to your project:**
   Open `ai_assistant.py` and replace the placeholder:
   ```python
   GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
   ```

## 📦 Install Required Package

```bash
# Activate your virtual environment
source venv/bin/activate  # or ./venv/bin/activate

# Install Gemini package (latest version)
pip install google-genai
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## ✨ Features

### AI Chat Panel
- **Location:** Right side of the screen
- **Toggle:** Click the sparkly ✨ button in the bottom-right corner
- **Keyboard Shortcut:** Cmd+Enter (Mac) or Ctrl+Enter (Windows) to send messages

### What the AI Can Help With:
- 🧠 **Machine Learning concepts** and algorithm explanations
- 📊 **Statistics & Probability** questions
- 🧮 **Mathematics** for data science
- 🧼 **Data cleaning** best practices
- 📈 **Visualization** recommendations
- 🔮 **Deep Learning** and NLP guidance
- 💻 **Code examples** and debugging
- 📚 **Concept explanations** at any detail level

### Example Questions to Ask:
- "Explain gradient descent in simple terms"
- "What's the difference between L1 and L2 regularization?"
- "How do I handle missing data in my dataset?"
- "Show me how to create a scatter plot with seaborn"
- "What is backpropagation?"
- "When should I use Random Forest vs XGBoost?"

## 🎨 UI Features

### Beautiful Design:
- **Glassmorphism** - Frosted glass effect with blur
- **Smooth animations** - Messages slide in gracefully
- **Gradient accents** - Blue-to-teal gradient branding
- **Responsive** - Works on desktop and mobile

### Message Formatting:
- **Code blocks** - Syntax highlighted
- **Bold text** - For emphasis
- **Line breaks** - Properly formatted
- **Lists** - Bulleted and numbered

### User Experience:
- **Real-time typing** - See responses as they generate
- **Message history** - Scroll through conversation
- **Auto-scroll** - Always see latest messages
- **Loading indicators** - Know when AI is thinking

## 🔒 Security Notes

- **Keep your API key private** - Don't commit it to Git
- **Add to .gitignore:**
  ```bash
  echo "ai_assistant.py" >> .gitignore
  ```
  Or use environment variables (better practice)

## 🚀 Advanced: Environment Variables (Recommended)

Instead of hardcoding your API key, use environment variables:

1. Create a `.env` file:
   ```bash
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. Update `ai_assistant.py`:
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
   ```

3. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

4. Add to .gitignore:
   ```bash
   echo ".env" >> .gitignore
   ```

## 🆓 Pricing

Gemini API has a **generous free tier**:
- 60 requests per minute
- 1,500 requests per day
- Perfect for learning and development

## 🐛 Troubleshooting

### "Please configure your Gemini API key"
- Check that you've added your API key to `ai_assistant.py`
- Make sure it's not still set to `"YOUR_GEMINI_API_KEY_HERE"`

### "Module not found: google.genai"
```bash
pip install google-genai
```

### API Rate Limit
- Free tier limits: 60 requests/minute
- Wait a moment and try again
- Consider upgrading for higher limits

### Connection Errors
- Check your internet connection
- Verify your API key is valid
- Ensure Google AI services are accessible in your region

## 💡 Tips

1. **Be specific** - The more context you provide, the better the response
2. **Ask follow-ups** - Build on previous answers
3. **Request examples** - Ask for code snippets or practical applications
4. **Adjust detail** - Ask for "simple", "detailed", or "technical" explanations

## 📚 Documentation

- **Gemini API Docs:** https://ai.google.dev/docs
- **Python SDK:** https://github.com/google/generative-ai-python
- **Pricing:** https://ai.google.dev/pricing

---

Built with ❤️ by Sania Panchal

