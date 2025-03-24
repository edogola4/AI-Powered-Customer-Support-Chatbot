import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
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
from nltk.stem import WordNetLemmatizer
import random

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    # # # # # nltk.download('wordnet')
    # # # # # nltk.download('punkt')

class TextPreprocessor:
    """A standalone text preprocessor that mimics the original functionality"""
    def __init__(self, intents_file):
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents()
        self.words = []
        self.classes = []
        self.documents = []
        
    def load_intents(self):
        """Load the intents from a JSON file"""
        try:
            with open(self.intents_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            # Create a simple sample dataset if file doesn't exist
            print(f"Warning: {self.intents_file} not found. Using sample data instead.")
            return {
                "intents": [
                    {
                        "tag": "greeting",
                        "patterns": ["Hi", "Hello", "Hey", "How are you", "Good day"],
                        "responses": ["Hello!", "Hi there!", "How can I help you?"]
                    },
                    {
                        "tag": "goodbye",
                        "patterns": ["Bye", "See you later", "Goodbye", "I'm leaving"],
                        "responses": ["Goodbye!", "See you later", "Have a nice day"]
                    },
                    {
                        "tag": "thanks",
                        "patterns": ["Thank you", "Thanks", "That's helpful"],
                        "responses": ["You're welcome!", "Any time!", "My pleasure"]
                    },
                    {
                        "tag": "help",
                        "patterns": ["I need help", "Can you help me", "Support"],
                        "responses": ["How can I help?", "What do you need help with?"]
                    }
                ]
            }
    
    def clean_text(self, text):
        """Clean and tokenize text"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize
        words = nltk.word_tokenize(text)
        # Lemmatize each word
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def preprocess(self):
        """Process the intents data"""
        for intent in self.intents["intents"]:
            tag = intent["tag"]
            # Add tag to classes if not present
            if tag not in self.classes:
                self.classes.append(tag)
            
            # Process each pattern
            for pattern in intent["patterns"]:
                word_list = self.clean_text(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, tag))
        
        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(self.classes)
        
        print(f"Number of documents: {len(self.documents)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Number of unique lemmatized words: {len(self.words)}")
        
        return self.words, self.classes, self.documents
    
    def create_training_data(self):
        """Create the training data"""
        training = []
        
        # Create an empty array for output
        output_empty = [0] * len(self.classes)
        
        # Create training set, bag of words for each sentence
        for doc in self.documents:
            # Initialize bag of words
            bag = [0] * len(self.words)
            word_patterns = doc[0]
            
            # Create bag of words array
            for word in word_patterns:
                bag[self.words.index(word)] = 1
            
            # Output is '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        # Shuffle the training data
        random.shuffle(training)
        
        # Split into X and y values
        train_x = [item[0] for item in training]
        train_y = [item[1] for item in training]
        
        return train_x, train_y

class IntentClassifierNN(nn.Module):
    """PyTorch implementation of the intent classifier neural network"""
    def __init__(self, input_size, output_size, layer_config):
        super(IntentClassifierNN, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, layer_config[0][0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(layer_config[0][1]))
        
        # Hidden layers
        for i in range(1, len(layer_config)):
            layers.append(nn.Linear(layer_config[i-1][0], layer_config[i][0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(layer_config[i][1]))
            
        # Output layer
        layers.append(nn.Linear(layer_config[-1][0], output_size))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    """Train the PyTorch model"""
    # Check for MPS (Apple Silicon) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    model.to(device)
    model.train()
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, label_indices = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == label_indices).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return history

def evaluate_model(model, data_loader):
    """Evaluate the model accuracy"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            _, label_indices = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == label_indices).sum().item()
    
    accuracy = correct / total
    avg_loss = loss_sum / len(data_loader)
    
    return avg_loss, accuracy

def get_model_size(model):
    """Get the size of the PyTorch model in MB"""
    os.makedirs('models', exist_ok=True)
    model_path = 'models/temp_model.pt'
    torch.save(model.state_dict(), model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return size_mb

def optimize_model():
    """Optimize the model and evaluate its performance"""
    print("Loading model and data...")
    
    # Make directories if they don't exist
    os.makedirs('data/intents', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    intents_file = 'data/intents/intents.json'
    
    # Load the preprocessor and prepare data
    preprocessor = TextPreprocessor(intents_file)
    words, classes, documents = preprocessor.preprocess()
    train_x, train_y = preprocessor.create_training_data()
    
    # Convert to PyTorch tensors
    train_x_tensor = torch.FloatTensor(train_x)
    train_y_tensor = torch.FloatTensor(train_y)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)
    
    # Test different model configurations
    layer_configs = [
        [(128, 0.5), (64, 0.5)],              # Original
        [(256, 0.5), (128, 0.5)],             # Larger
        [(128, 0.5), (64, 0.5), (32, 0.5)],   # Deeper
        [(64, 0.3), (32, 0.3)]                # Smaller with less dropout
    ]
    
    results = []
    
    for i, config in enumerate(layer_configs):
        print(f"\nTesting configuration {i+1}:")
        for layer in config:
            print(f"- Linear({layer[0]}) with Dropout({layer[1]})")
        
        # Build model with this configuration
        model = IntentClassifierNN(len(words), len(classes), config)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Train model
        start_time = time.time()
        history = train_model(model, train_loader, criterion, optimizer, epochs=100)
        end_time = time.time()
        
        # Evaluate model
        loss, accuracy = evaluate_model(model, train_loader)
        training_time = end_time - start_time
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Get model size
        model_size = get_model_size(model)
        print(f"Model size: {model_size:.2f} MB")
        
        results.append({
            'config': config,
            'accuracy': accuracy,
            'loss': loss,
            'training_time': training_time,
            'model_size': model_size,
            'history': {'loss': history['loss'], 'accuracy': history['accuracy']}
        })
    
    # Find the best model
    best_model_idx = np.argmax([r['accuracy'] for r in results])
    best_model = results[best_model_idx]
    
    print("\n=== Best Model Configuration ===")
    for layer in best_model['config']:
        print(f"- Linear({layer[0]}) with Dropout({layer[1]})")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Loss: {best_model['loss']:.4f}")
    print(f"Training time: {best_model['training_time']:.2f} seconds")
    print(f"Model size: {best_model['model_size']:.2f} MB")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    accuracies = [r['accuracy'] for r in results]
    plt.bar(range(len(results)), accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xticks(range(len(results)), [f"Config {i+1}" for i in range(len(results))])
    plt.ylim(0.8, 1.0)  # Adjust as needed
    
    # Training time comparison
    plt.subplot(2, 2, 2)
    times = [r['training_time'] for r in results]
    plt.bar(range(len(results)), times)
    plt.title('Training Time Comparison')
    plt.xticks(range(len(results)), [f"Config {i+1}" for i in range(len(results))])
    
    # Model size comparison
    plt.subplot(2, 2, 3)
    sizes = [r['model_size'] for r in results]
    plt.bar(range(len(results)), sizes)
    plt.title('Model Size Comparison')
    plt.xticks(range(len(results)), [f"Config {i+1}" for i in range(len(results))])
    
    # Learning curve of best model
    plt.subplot(2, 2, 4)
    epochs = range(1, len(best_model['history']['accuracy']) + 1)
    plt.plot(epochs, best_model['history']['accuracy'])
    plt.plot(epochs, best_model['history']['loss'])
    plt.title('Best Model Learning Curve')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('model_optimization_results.png')
    plt.close()
    
    # Save the best model configuration
    with open('models/best_config.json', 'w') as f:
        json.dump({
            'config': [[int(layer[0]), float(layer[1])] for layer in best_model['config']],
            'accuracy': float(best_model['accuracy']),
            'loss': float(best_model['loss']),
            'words': words,
            'classes': classes
        }, f)
    
    # Save the best model
    best_model_config = best_model['config']
    best_model_obj = IntentClassifierNN(len(words), len(classes), best_model_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(best_model_obj.parameters(), lr=0.01, momentum=0.9)
    train_model(best_model_obj, train_loader, criterion, optimizer, epochs=100)
    torch.save(best_model_obj, 'models/best_model.pt')
    
    print("Optimization complete. Results saved to model_optimization_results.png")

if __name__ == "__main__":
    # Make sure the models directory exists
    os.makedirs('models', exist_ok=True)
    optimize_model()