# Deploying to Vercel

## ✅ Your app is ready for Vercel deployment!

### Prerequisites
1. Install Vercel CLI: `npm install -g vercel`
2. Sign up at [vercel.com](https://vercel.com)

### Deployment Steps

#### Option 1: Deploy via Vercel CLI (Recommended)
```bash
# Navigate to project directory
cd /Users/satvikpanchal/data_science_encyclopedia

# Login to Vercel (first time only)
vercel login

# Deploy
vercel
```

#### Option 2: Deploy via GitHub
1. Push your code to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your GitHub repository
4. Vercel will auto-detect the Flask app and deploy

### Environment Variables
If you want to keep your Gemini API key secure:

1. In Vercel dashboard, go to: **Settings → Environment Variables**
2. Add: `GEMINI_API_KEY` = `your_api_key_here`
3. Update `ai_assistant.py` to read from environment:
   ```python
   GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "fallback_key")
   ```

### Important Notes

⚠️ **Vercel Limitations for Flask:**
- Vercel uses serverless functions (cold starts may occur)
- Maximum execution time: 10 seconds (Hobby plan) / 60 seconds (Pro)
- File system is read-only (can't generate plots at runtime)
- Better for stateless apps

### Alternative Deployment Options

If you encounter issues with Vercel, consider these Flask-friendly alternatives:

#### 1. **Render** (Recommended for Flask)
- Free tier available
- Better for long-running Flask apps
- Easy deployment: https://render.com
- Command: `gunicorn app:app`

#### 2. **Railway**
- Great Flask support
- Free tier with $5 credit/month
- https://railway.app

#### 3. **PythonAnywhere**
- Designed for Python web apps
- Free tier available
- https://pythonanywhere.com

#### 4. **Heroku**
- Classic choice for Flask
- Free tier (with limitations)
- https://heroku.com

### For Render Deployment (Recommended)

Create a `render.yaml`:
```yaml
services:
  - type: web
    name: data-science-encyclopedia
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
```

Add to `requirements.txt`:
```
gunicorn==21.2.0
```

## 🚀 Quick Start

```bash
# Deploy to Vercel
vercel

# Or deploy to Render (if you prefer)
# 1. Push to GitHub
# 2. Connect at render.com
# 3. It will auto-deploy
```

## 📝 Notes

- All images in `/static/images/` will be deployed
- The AI Assistant will work if GEMINI_API_KEY is set
- Static files are automatically served
- Your app will get a URL like: `your-app.vercel.app`

Good luck with deployment! 🎉

