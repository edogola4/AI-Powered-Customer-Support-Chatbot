#!/bin/bash

# Create directory structure
mkdir -p data/intents
mkdir -p models
mkdir -p src/nlp
mkdir -p src/api
mkdir -p src/utils
mkdir -p templates

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment
$PYTHON -m venv venv

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Failed to create virtual environment."
    exit 1
fi

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Download NLTK data
$PYTHON -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Train the model
$PYTHON train_model.py

echo "Setup complete. Run 'python app.py' to start the application."