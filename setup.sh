#!/bin/bash

# Initialize Git repository if it doesn't exist
if [ ! -d ".git" ]; then
  echo "Initializing Git repository..."
  git init
  
  # Configure Git to use the existing repository if needed
  echo "Configuring Git to use the jasonhall1985 repository..."
  git remote add origin https://github.com/jasonhall1985/icu-lipreading-mvp.git || true
fi

# Add all files to Git
git add .
git commit -m "Initial commit: ICU-lipreading MVP setup" || true

# Check for Python version
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
fi

echo "Using Python command: $PYTHON_CMD"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv || {
  echo "Failed to create virtual environment with venv. Trying virtualenv..."
  pip install virtualenv || pip3 install virtualenv
  $PYTHON_CMD -m virtualenv venv
}

# Activate virtual environment and install dependencies
echo "Activating virtual environment..."
source venv/bin/activate || {
  echo "Failed to activate virtual environment. Please check your Python installation."
  exit 1
}

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models

# Download and prepare the LipNet model
echo "Downloading and preparing the LipNet model..."
$PYTHON_CMD download_model.py

echo "Setup complete! Virtual environment created and dependencies installed."
echo "The LipNet model has been downloaded and prepared for use."
echo "To activate the virtual environment in the future, run: source venv/bin/activate"
