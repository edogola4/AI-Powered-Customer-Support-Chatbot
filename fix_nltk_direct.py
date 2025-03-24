import os

# Files to fix
nltk_init_file = './venv/lib/python3.12/site-packages/nltk/__init__.py'
preprocessing_file = './src/nlp/preprocessing.py'

# Fix the NLTK __init__.py file
if os.path.exists(nltk_init_file):
    try:
        with open(nltk_init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'nltk.data.path.append(' in content:
            # Replace any problematic circular references
            fixed_content = content.replace(
                'nltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))', 
                '# Set the NLTK data path\nfrom nltk import data\ndata.path.append(os.path.expanduser(\'~/nltk_data\'))'
            )
            
            # Also comment out any nltk.download calls
            fixed_content = fixed_content.replace('nltk.download(', '# nltk.download(')
            
            with open(nltk_init_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed NLTK __init__.py file")
        else:
            print(f"No changes needed for NLTK __init__.py")
    except Exception as e:
        print(f"Error fixing NLTK init file: {str(e)}")

# Fix the preprocessing.py file
if os.path.exists(preprocessing_file):
    try:
        with open(preprocessing_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add the data path line if it doesn't already exist
        if 'nltk.data.path.append(' not in content:
            # If there's an import nltk line, add after it
            if 'import nltk' in content:
                modified_content = content.replace(
                    'import nltk', 
                    'import nltk\nimport os\n\n# Set NLTK data path\nnltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))'
                )
            else:
                # Otherwise add at the top of the file
                modified_content = 'import os\nimport nltk\n\n# Set NLTK data path\nnltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))\n\n' + content
            
            # Comment out any nltk.download calls
            modified_content = modified_content.replace('nltk.download(', '# nltk.download(')
            
            with open(preprocessing_file, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"Fixed preprocessing.py file")
        else:
            print(f"No changes needed for preprocessing.py")
    except Exception as e:
        print(f"Error fixing preprocessing file: {str(e)}")

print("Done fixing NLTK files.")