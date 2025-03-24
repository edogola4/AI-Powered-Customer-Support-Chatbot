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
                        
                        # Special handling for __init__.py files
                        if os.path.basename(filepath) == '__init__.py' and 'nltk.download' in content:
                            # For __init__.py files, just comment out download calls
                            modified_content = content.replace('nltk.download(', '# nltk.download(')
                        else:
                            # Add data path line after imports for non-init files
                            if 'import nltk' in content:
                                modified_content = content.replace('import nltk', 'import nltk\nimport os\n\n# Set NLTK data path\nnltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))')
                            else:
                                lines = content.split('\n')
                                import_section_end = 0
                                
                                # Find the end of the import section
                                for i, line in enumerate(lines):
                                    if 'import' in line:
                                        import_section_end = i
                                
                                # Add the data path setting after imports
                                if import_section_end > 0:
                                    lines.insert(import_section_end + 1, 'import os\n\n# Set NLTK data path\nnltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))')
                                modified_content = '\n'.join(lines)
                            
                            # Comment out nltk.download lines
                            modified_content = modified_content.replace('nltk.download(', '# nltk.download(')
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(modified_content)
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a more forgiving encoding or skip the file
                    print(f"Skipping file due to encoding issues: {filepath}")
                except Exception as e:
                    print(f"Error processing file {filepath}: {str(e)}")

# Function to fix specific NLTK __init__.py issue if it exists
def fix_nltk_init_file():
    nltk_init_file = './venv/lib/python3.12/site-packages/nltk/__init__.py'
    
    try:
        if os.path.exists(nltk_init_file):
            with open(nltk_init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if our modification created an error
            if 'nltk.data.path.append' in content:
                # Fix the circular reference
                fixed_content = content.replace(
                    'nltk.data.path.append(os.path.expanduser(\'~/nltk_data\'))', 
                    'from nltk import data\ndata.path.append(os.path.expanduser(\'~/nltk_data\'))'
                )
                
                with open(nltk_init_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"Fixed circular reference in {nltk_init_file}")
    except Exception as e:
        print(f"Error fixing NLTK init file: {str(e)}")

if __name__ == "__main__":
    fix_nltk_downloads('.')
    fix_nltk_init_file()  # Run this additional fix for the __init__.py issue
    print("Done fixing NLTK download calls.")