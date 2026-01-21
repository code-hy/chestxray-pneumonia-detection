#!/bin/bash

# Exit on any error
set -e

echo "📥 Downloading Chest X-ray (Pneumonia) dataset from Kaggle..."

# Create data directory if it doesn't exist
mkdir -p data

# Check if kaggle.json exists in ~/.kaggle
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "❌ Error: kaggle.json not found in ~/.kaggle/"
    echo "👉 Please follow these steps:"
    echo "   1. Go to https://www.kaggle.com/settings/account"
    echo "   2. Scroll to 'API' → Click 'Create New API Token'"
    echo "   3. Move the downloaded 'kaggle.json' to ~/.kaggle/kaggle.json"
    echo "   4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Ensure kaggle.json has correct permissions
chmod 600 "$HOME/.kaggle/kaggle.json"

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/

# Unzip and clean up
echo "📦 Extracting dataset..."
unzip -q data/chest-xray-pneumonia.zip -d data/
rm data/chest-xray-pneumonia.zip

# Verify structure
if [ ! -d "data/chest_xray" ]; then
    echo "❌ Unexpected dataset structure. Expected 'data/chest_xray/'"
    echo "   Contents of data/:" 
    ls data/
    exit 1
fi

echo "✅ Dataset successfully downloaded to: data/chest_xray/"
echo "📁 Structure:"
tree data/chest_xray/ --dirsfirst -L 2