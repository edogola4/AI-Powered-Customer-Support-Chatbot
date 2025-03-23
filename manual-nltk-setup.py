import os
import ssl
import nltk
import sys

# Print current environment info
print(f"Python version: {sys.version}")
print(f"NLTK version: {nltk.__version__}")

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
    print(f"Created NLTK data directory: {nltk_data_dir}")
else:
    print(f"NLTK data directory already exists: {nltk_data_dir}")

# Create corpora directory if it doesn't exist
corpora_dir = os.path.join(nltk_data_dir, 'corpora')
if not os.path.exists(corpora_dir):
    os.makedirs(corpora_dir)
    print(f"Created corpora directory: {corpora_dir}")
else:
    print(f"Corpora directory already exists: {corpora_dir}")

# Create tokenizers directory if it doesn't exist
tokenizers_dir = os.path.join(nltk_data_dir, 'tokenizers')
if not os.path.exists(tokenizers_dir):
    os.makedirs(tokenizers_dir)
    print(f"Created tokenizers directory: {tokenizers_dir}")
else:
    print(f"Tokenizers directory already exists: {tokenizers_dir}")

print("\nTo complete setup, download these files manually:")
print("1. punkt - https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip")
print("   Extract and place in:", os.path.join(tokenizers_dir, 'punkt'))
print("2. wordnet - https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip")
print("   Extract and place in:", os.path.join(corpora_dir, 'wordnet'))
print("\nAfter downloading and extracting these files, your app should work.")

# Test if NLTK can find the data directories
print("\nChecking NLTK data path:")
print(nltk.data.path)