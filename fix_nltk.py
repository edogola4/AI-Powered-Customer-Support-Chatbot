import os

def fix_nltk_downloads(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    # Try to read with UTF-8 encoding first
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'nltk.download' in content:
                        print(f"Fixing file: {filepath}")
                        
                        # Add data path line after imports
                        if 'import nltk
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))' in content:
                            modified_content = content.replace('import nltk
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))', 'import nltk
import os

# Set NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))\nimport os\n\n# Set NLTK data path\nnltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))')
                        else:
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if 'nltk' in line and 'import' in line:
                                    lines.insert(i+1, 'import os\n\n# Set NLTK data path\nnltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))')
                                    break
                            modified_content = '\n'.join(lines)
                        
                        # Comment out nltk.download lines
                        modified_content = modified_content.replace('# nltk.download(', '# # nltk.download(')
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(modified_content)
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a more forgiving encoding or skip the file
                    print(f"Skipping file due to encoding issues: {filepath}")
                except Exception as e:
                    print(f"Error processing file {filepath}: {str(e)}")

if __name__ == "__main__":
    fix_nltk_downloads('.')
    print("Done fixing NLTK download calls.")