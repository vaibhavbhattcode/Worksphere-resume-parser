"""
Production-Ready Resume Parser Service
Optimized for deployment with job portal integration
"""

import os
import re
import json
import spacy
import logging
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import PyPDF2
import pdfplumber
from docx import Document
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import lru_cache
from rapidfuzz import fuzz, process as rf_process
import dateutil.parser as dparser

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    # Flask settings
    DEBUG = os.getenv('FLASK_ENV', 'production') == 'development'
    PORT = int(os.getenv('PORT', 5001))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # File upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './temp')
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc'}
    
    # CORS settings
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')
    
    # Cache settings
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', 100))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS for job portal integration
CORS(app, resources={
    r"/*": {
        "origins": [Config.FRONTEND_URL, Config.BACKEND_URL, "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Create upload folder
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ============================================================================
# SPACY MODEL LOADING
# ============================================================================

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ spaCy model loaded successfully")
except OSError:
    logger.error("❌ spaCy model not found. Install: python -m spacy download en_core_web_sm")
    nlp = None

# ============================================================================
# CACHE IMPLEMENTATION
# ============================================================================

_PARSE_CACHE: Dict[str, Dict] = {}

def _hash_file(path: str) -> str:
    """Generate SHA256 hash of file for caching"""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_from_cache(file_hash: str) -> Optional[Dict]:
    """Get parsed result from cache"""
    if not Config.ENABLE_CACHE:
        return None
    return _PARSE_CACHE.get(file_hash)

def save_to_cache(file_hash: str, result: Dict):
    """Save parsed result to cache with size limit"""
    if not Config.ENABLE_CACHE:
        return
    
    if len(_PARSE_CACHE) >= Config.CACHE_MAX_SIZE:
        # Remove oldest entry
        _PARSE_CACHE.pop(next(iter(_PARSE_CACHE)))
    
    _PARSE_CACHE[file_hash] = result
    logger.info(f"📦 Cached result for {file_hash[:8]}... (cache size: {len(_PARSE_CACHE)})")

# ============================================================================
# RESUME PARSER CLASS
# ============================================================================

class ResumeParser:
    def __init__(self):
        self.non_cities = self._load_non_cities()
        self.job_titles = self._load_job_titles()
        self.skills_db = self._load_skills()
        self.industries = self._load_industries()
        
    def _load_non_cities(self) -> set:
        """Load common words that aren't cities"""
        return set([
            # Indian states
            "india", "maharashtra", "uttar pradesh", "madhya pradesh", "karnataka", 
            "tamil nadu", "gujarat", "rajasthan", "punjab", "haryana", "bihar",
            # US states
            "usa", "california", "texas", "florida", "new york", "illinois",
            # Tech terms
            "react", "node", "python", "javascript", "java", "developer", "engineer",
            "mern", "full stack", "remote", "onsite", "hybrid", "state", "country"
        ])
    
    def _load_job_titles(self) -> List[str]:
        """Load common job titles for better matching"""
        return [
            "software engineer", "software developer", "full stack developer",
            "frontend developer", "backend developer", "devops engineer",
            "data scientist", "data analyst", "machine learning engineer",
            "product manager", "project manager", "business analyst",
            "ui/ux designer", "graphic designer", "web developer",
            "mobile developer", "qa engineer", "test engineer"
        ]
    
    def _load_skills(self) -> Dict[str, List[str]]:
        """Load categorized technical skills"""
        return {
            "programming": ["python", "javascript", "java", "c++", "c#", "ruby", "go", "rust"],
            "frontend": ["react", "angular", "vue", "html", "css", "tailwind", "bootstrap"],
            "backend": ["node.js", "express", "django", "flask", "spring", "asp.net"],
            "database": ["mongodb", "mysql", "postgresql", "redis", "elasticsearch"],
            "devops": ["docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "git"],
            "data": ["pandas", "numpy", "tensorflow", "pytorch", "spark", "tableau"]
        }
    
    def _load_industries(self) -> List[str]:
        """Load industry keywords"""
        return [
            "technology", "healthcare", "finance", "education", "manufacturing",
            "retail", "consulting", "marketing", "real estate", "transportation"
        ]
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF (pdfplumber first, PyPDF2 fallback)"""
        # Try pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([
                    page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                    for page in pdf.pages
                ])
                if len(text.strip()) >= 50:
                    return text
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            logger.error(f"PyPDF2 failed: {e}")
            raise
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume"""
        if not nlp:
            return None
        
        doc = nlp(text[:1000])  # Check first 1000 chars
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        return None
    
    def extract_email(self, text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return list(set(re.findall(pattern, text, re.IGNORECASE)))
    
    def extract_phone(self, text: str) -> List[str]:
        """Extract phone numbers"""
        patterns = [
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US/International
            r'(\+?\d{2,3}[-.\s]?)?\d{10}',  # Indian format
            r'(\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{7}'  # Alternative format
        ]
        
        phones = []
        for pattern in patterns:
            phones.extend(re.findall(pattern, text))
        
        # Clean and deduplicate
        cleaned = []
        for phone in phones:
            if isinstance(phone, tuple):
                phone = ''.join(phone)
            phone = re.sub(r'[^\d+]', '', str(phone))
            if 10 <= len(phone) <= 15:
                cleaned.append(phone)
        
        return list(set(cleaned))
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract location with improved city detection"""
        if not nlp:
            return None
        
        doc = nlp(text[:2000])
        locations = []
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                loc = ent.text.strip().lower()
                if loc not in self.non_cities and len(loc) > 2:
                    locations.append(ent.text.strip())
        
        return locations[0] if locations else None
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills"""
        text_lower = text.lower()
        found_skills = set()
        
        for category, skills in self.skills_db.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.add(skill)
        
        return sorted(list(found_skills))
    
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract years of experience"""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
            r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|as)',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([int(m) for m in matches])
        
        if years:
            return {
                "years": max(years),
                "confidence": "high" if len(years) > 1 else "medium"
            }
        
        return {"years": 0, "confidence": "low"}
    
    def extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education details"""
        degrees = [
            "phd", "ph.d", "doctorate", "master", "mba", "m.s", "m.tech", "m.sc",
            "bachelor", "b.tech", "b.e", "b.sc", "b.com", "bca", "mca",
            "diploma", "associate"
        ]
        
        education = []
        text_lower = text.lower()
        
        for degree in degrees:
            if degree in text_lower:
                education.append({
                    "degree": degree.upper(),
                    "confidence": "medium"
                })
        
        return education[:3]  # Return top 3
    
    def extract_job_title(self, text: str) -> Optional[str]:
        """Extract most likely job title using fuzzy matching"""
        text_lower = text[:1000].lower()
        
        # Find best matching job title
        matches = rf_process.extract(
            text_lower,
            self.job_titles,
            scorer=fuzz.partial_ratio,
            limit=1
        )
        
        if matches and matches[0][1] > 70:  # 70% confidence threshold
            return matches[0][0].title()
        
        return None
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Main parsing function"""
        start_time = datetime.now()
        
        try:
            # Check cache
            file_hash = _hash_file(file_path)
            cached = get_from_cache(file_hash)
            if cached:
                logger.info(f"✅ Cache hit for {file_hash[:8]}...")
                return cached
            
            # Extract text
            text = self.extract_text(file_path)
            
            if not text or len(text.strip()) < 50:
                raise ValueError("Insufficient text extracted from resume")
            
            # Parse all fields
            result = {
                "success": True,
                "data": {
                    "name": self.extract_name(text),
                    "email": self.extract_email(text),
                    "phone": self.extract_phone(text),
                    "location": self.extract_location(text),
                    "skills": self.extract_skills(text),
                    "experience": self.extract_experience(text),
                    "education": self.extract_education(text),
                    "job_title": self.extract_job_title(text),
                    "raw_text_length": len(text),
                    "parsed_at": datetime.now().isoformat()
                },
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000)
            }
            
            # Cache result
            save_to_cache(file_hash, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000)
            }

# ============================================================================
# GLOBAL PARSER INSTANCE
# ============================================================================

parser = ResumeParser()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "success": True,
        "service": "Resume Parser API",
        "version": "1.0.0",
        "status": "running",
        "spacy_loaded": nlp is not None,
        "cache_size": len(_PARSE_CACHE),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        "success": True,
        "status": "healthy",
        "checks": {
            "spacy_model": nlp is not None,
            "upload_folder": os.path.exists(Config.UPLOAD_FOLDER),
            "cache_enabled": Config.ENABLE_CACHE,
            "cache_entries": len(_PARSE_CACHE)
        },
        "config": {
            "max_file_size_mb": Config.MAX_FILE_SIZE / (1024 * 1024),
            "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
            "debug_mode": Config.DEBUG
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/parse', methods=['POST'])
def parse_resume():
    """Parse uploaded resume file"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        if not parser:
            return jsonify({
                "success": False,
                "error": "Parser not initialized"
            }), 500
        
        # Validate file extension
        filename = secure_filename(file.filename)
        if not any(filename.lower().endswith(f'.{ext}') for ext in Config.ALLOWED_EXTENSIONS):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Save file temporarily
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > Config.MAX_FILE_SIZE:
            os.remove(temp_path)
            return jsonify({
                "success": False,
                "error": f"File too large. Max size: {Config.MAX_FILE_SIZE / (1024*1024)}MB"
            }), 400
        
        # Parse resume
        result = parser.parse(temp_path)
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify(result), 200 if result.get("success") else 500
        
    except Exception as e:
        logger.error(f"Parse endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear parsing cache (admin endpoint)"""
    global _PARSE_CACHE
    size_before = len(_PARSE_CACHE)
    _PARSE_CACHE.clear()
    
    return jsonify({
        "success": True,
        "message": f"Cache cleared. Removed {size_before} entries",
        "cache_size": len(_PARSE_CACHE)
    })

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    return jsonify({
        "success": True,
        "cache_enabled": Config.ENABLE_CACHE,
        "cache_size": len(_PARSE_CACHE),
        "cache_max_size": Config.CACHE_MAX_SIZE,
        "cache_keys": list(_PARSE_CACHE.keys())[:10]  # Show first 10
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Starting Resume Parser Service")
    logger.info("=" * 60)
    logger.info(f"Environment: {Config.DEBUG and 'Development' or 'Production'}")
    logger.info(f"Host: {Config.HOST}")
    logger.info(f"Port: {Config.PORT}")
    logger.info(f"Max File Size: {Config.MAX_FILE_SIZE / (1024*1024)}MB")
    logger.info(f"Cache Enabled: {Config.ENABLE_CACHE}")
    logger.info(f"CORS Origins: {Config.FRONTEND_URL}, {Config.BACKEND_URL}")
    logger.info("=" * 60)
    
    if not nlp:
        logger.error("⚠️  spaCy model not loaded. Run: python -m spacy download en_core_web_sm")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
