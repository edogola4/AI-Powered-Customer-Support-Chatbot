import nltk

try:
    # Test punkt
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Punkt tokenizer loaded successfully!")
    
    # Test simple tokenization
    sample = "Hello world. This is a test."
    print(f"Tokenized sample: {tokenizer.tokenize(sample)}")
    
    # Test wordnet
    from nltk.corpus import wordnet
    synonyms = wordnet.synsets("good")
    print(f"Wordnet loaded successfully! Found {len(synonyms)} synonyms for 'good'")
    if synonyms:
        print(f"Example synonym: {synonyms[0].lemma_names()[0]}")
    
    print("\nNLTK data installation verified successfully!")
except Exception as e:
    print(f"Error: {e}")