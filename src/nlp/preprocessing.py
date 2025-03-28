import json
import pickle
import numpy as np
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
from nltk.stem import WordNetLemmatizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

# Download necessary NLTK data
# # # # nltk.download('punkt')
# # # # nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, intents_file):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open(intents_file).read())
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        self.vocab = None
        self.max_sequence_length = 20
        
    def preprocess(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word in the pattern
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                # Add documents to corpus
                self.documents.append((word_list, intent['tag']))
                # Add to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Lemmatize and lower each word, removing duplicates
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        # Create vocabulary using torchtext
        def yield_tokens():
            for word in self.words:
                yield [word]
        
        self.vocab = build_vocab_from_iterator(yield_tokens(), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        # Save data
        pickle.dump(self.words, open('models/words.pkl', 'wb'))
        pickle.dump(self.classes, open('models/classes.pkl', 'wb'))
        pickle.dump(self.vocab, open('models/vocab.pkl', 'wb'))
        
        return self.words, self.classes, self.documents
    
    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        
        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            # Create bag of words
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            # Create output row
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])
        
        # Shuffle features
        np.random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Create train and test lists
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        return np.array(train_x), np.array(train_y)
    
    def prepare_input(self, sentence):
        # Tokenize pattern
        sentence_words = nltk.word_tokenize(sentence)
        # Lemmatize each word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        
        # Create bag of words
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, word in enumerate(self.words):
                if word == s:
                    bag[i] = 1
        
        return np.array([bag])
    
    # New method to convert to PyTorch tensor
    def text_to_tensor(self, sentence):
        # Process input sentence
        processed_input = self.prepare_input(sentence)
        # Convert numpy array to PyTorch tensor
        return torch.tensor(processed_input, dtype=torch.float32)