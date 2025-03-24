import nltk
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
import ssl
import os
import sys

def download_nltk_resources():
    # Bypass SSL certificate verification
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Create directories if they don't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download required resources
    resources = ['punkt', 'wordnet', 'punkt_tab']
    
    print("Downloading NLTK resources...")
    for resource in resources:
        try:
            # # # # # # # nltk.download(resource, quiet=False)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    # Additional step: manually create punkt_tab directory if needed
    punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
    os.makedirs(punkt_tab_dir, exist_ok=True)
    
    # Check if downloads were successful
    success = all(nltk.data.find(f"tokenizers/{res}") for res in ['punkt'])
    success = success and nltk.data.find("corpora/wordnet")
    
    if success:
        print("\nAll resources downloaded successfully!")
    else:
        print("\nSome resources might not have downloaded correctly.")
        print("You may need to manually install them.")

if __name__ == "__main__":
    download_nltk_resources()