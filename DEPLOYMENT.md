# Resume Parser Deployment Guide

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements-production.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Copy and configure environment
cp .env.production .env

# Run service
python resume_parser_production.py
```

### Using Shell Script (Linux/Mac)
```bash
chmod +x start.sh
./start.sh
```

### Using Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

---

## 🌐 Free Deployment Options

### 1. **Render.com** (Recommended)
**Free Tier:** 750 hours/month

#### Steps:
1. Push code to GitHub
2. Go to [Render.com](https://render.com)
3. Click **New** → **Web Service**
4. Connect your repository
5. Configure:
   - **Name:** `resume-parser`
   - **Environment:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements-production.txt && python -m spacy download en_core_web_sm
     ```
   - **Start Command:**
     ```bash
     gunicorn resume_parser_production:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
     ```
6. Add environment variables:
   ```
   FLASK_ENV=production
   FRONTEND_URL=https://your-frontend.vercel.app
   BACKEND_URL=https://your-backend.render.com
   ```

#### Auto-deploy Configuration:
Create `render.yaml`:
```yaml
services:
  - type: web
    name: resume-parser
    env: python
    buildCommand: "pip install -r requirements-production.txt && python -m spacy download en_core_web_sm"
    startCommand: "gunicorn resume_parser_production:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120"
    envVars:
      - key: FLASK_ENV
        value: production
      - key: FRONTEND_URL
        sync: false
      - key: BACKEND_URL
        sync: false
```

---

### 2. **Railway.app**
**Free Tier:** $5 credit/month

#### Steps:
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

Or use the web interface:
1. Go to [Railway.app](https://railway.app)
2. Click **New Project** → **Deploy from GitHub**
3. Select repository
4. Railway auto-detects Python and Procfile

---

### 3. **Fly.io**
**Free Tier:** 3 shared VMs

#### Steps:
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Launch:
   ```bash
   fly launch
   ```
4. Configure `fly.toml` (auto-generated):
   ```toml
   app = "resume-parser"
   
   [build]
   
   [env]
   PORT = "8080"
   FLASK_ENV = "production"
   
   [[services]]
   http_checks = []
   internal_port = 8080
   processes = ["app"]
   protocol = "tcp"
   
   [[services.ports]]
   force_https = true
   handlers = ["http"]
   port = 80
   
   [[services.ports]]
   handlers = ["tls", "http"]
   port = 443
   ```
5. Deploy: `fly deploy`

---

### 4. **PythonAnywhere**
**Free Tier:** 1 web app

#### Steps:
1. Sign up at [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Go to **Web** → **Add new web app**
3. Choose **Flask**
4. Upload files or use bash console:
   ```bash
   git clone https://github.com/your-repo/resume-parser
   cd resume-parser
   pip install -r requirements-production.txt --user
   python -m spacy download en_core_web_sm
   ```
5. Configure WSGI file:
   ```python
   import sys
   path = '/home/yourusername/resume-parser'
   if path not in sys.path:
       sys.path.append(path)
   
   from resume_parser_production import app as application
   ```

---

### 5. **Heroku** (Paid plans only now)
If you have Heroku credits:
```bash
heroku create resume-parser-app
git push heroku main
heroku config:set FLASK_ENV=production
heroku config:set FRONTEND_URL=https://your-frontend.com
```

---

## 🔗 Integration with Job Portal

### Backend Integration (Express.js)
```javascript
// backend/utils/resumeParser.js
import axios from 'axios';

const PARSER_URL = process.env.RESUME_PARSER_URL || 'http://localhost:5001';

export async function parseResume(filePath) {
  try {
    const FormData = require('form-data');
    const fs = require('fs');
    
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    const response = await axios.post(`${PARSER_URL}/parse`, form, {
      headers: form.getHeaders(),
      timeout: 30000
    });
    
    return response.data;
  } catch (error) {
    console.error('Resume parsing failed:', error);
    return { success: false, error: error.message };
  }
}
```

### Frontend Integration (React)
```javascript
// frontend/src/utils/resumeParser.js
const PARSER_URL = process.env.REACT_APP_RESUME_PARSER_URL || 'http://localhost:5001';

export async function uploadAndParseResume(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${PARSER_URL}/parse`, {
      method: 'POST',
      body: formData
    });
    
    return await response.json();
  } catch (error) {
    console.error('Parse error:', error);
    return { success: false, error: error.message };
  }
}
```

---

## 📋 Environment Variables

### Required
- `PORT`: Service port (default: 5001)
- `FLASK_ENV`: Environment (production/development)

### Optional
- `FRONTEND_URL`: Frontend URL for CORS
- `BACKEND_URL`: Backend URL for CORS
- `MAX_FILE_SIZE_MB`: Max upload size (default: 10)
- `ENABLE_CACHE`: Enable caching (default: true)
- `CACHE_MAX_SIZE`: Max cache entries (default: 100)
- `LOG_LEVEL`: Logging level (default: INFO)

---

## 🧪 Testing

### Health Check
```bash
curl http://localhost:5001/health
```

### Parse Resume
```bash
curl -X POST http://localhost:5001/parse \
  -F "file=@/path/to/resume.pdf"
```

### Clear Cache
```bash
curl -X POST http://localhost:5001/cache/clear
```

---

## 🔒 Security Best Practices

1. **Set rate limiting** (add Flask-Limiter)
2. **Add API key authentication**
3. **Validate file types server-side**
4. **Scan uploads for malware**
5. **Use HTTPS in production**
6. **Set proper CORS origins**
7. **Implement request size limits**

---

## 📊 Monitoring

### Health Endpoints
- `/` - Basic status
- `/health` - Detailed health check
- `/cache/stats` - Cache statistics

### Logs
```bash
# Docker
docker-compose logs -f

# Railway
railway logs

# Render
View in dashboard
```

---

## 🐛 Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Memory Issues
Reduce workers in Procfile:
```
--workers 1
```

### Timeout Errors
Increase timeout:
```
--timeout 180
```

### CORS Errors
Update `.env`:
```
FRONTEND_URL=https://your-actual-frontend.com
```

---

## 📈 Performance Tips

1. **Enable caching** - Reuse parsed results
2. **Use CDN** - For static assets
3. **Optimize workers** - Based on available RAM
4. **Add Redis** - For distributed caching
5. **Implement queuing** - For async processing

---

## 🎯 Next Steps

1. Deploy to Render.com (free tier)
2. Update backend `.env` with parser URL
3. Update frontend `.env` with parser URL
4. Test integration end-to-end
5. Monitor performance and logs

---

## 📚 Resources

- [Render Documentation](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app)
- [Flask-CORS Documentation](https://flask-cors.readthedocs.io)
- [Gunicorn Documentation](https://docs.gunicorn.org)
