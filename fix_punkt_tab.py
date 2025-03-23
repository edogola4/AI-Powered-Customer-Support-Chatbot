import os
import nltk

# Ensure the punkt_tab directory exists
home_dir = os.path.expanduser('~')
nltk_data_dir = os.path.join(home_dir, 'nltk_data')
punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
os.makedirs(punkt_tab_dir, exist_ok=True)

# Create an empty file in the directory
with open(os.path.join(punkt_tab_dir, 'punkt.data'), 'w') as f:
    f.write('')

print(f"Created punkt_tab directory at {punkt_tab_dir}")
print("Try running your script again.")