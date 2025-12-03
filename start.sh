#!/bin/bash
# Production startup script

echo "🚀 Starting Resume Parser Service..."

# Install spaCy model if not present
python -m spacy download en_core_web_sm --quiet 2>/dev/null || echo "✓ spaCy model already installed"

# Start the service
if [ "$FLASK_ENV" = "development" ]; then
    echo "📍 Running in DEVELOPMENT mode"
    python resume_parser_production.py
else
    echo "📍 Running in PRODUCTION mode with Gunicorn"
    gunicorn resume_parser_production:app \
        --bind 0.0.0.0:${PORT:-5001} \
        --workers ${WORKERS:-2} \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        --log-level info
fi
