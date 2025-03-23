import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os

class IntentModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super(IntentModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.words = None
        self.classes = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self, input_shape, output_shape):
        # Creating the model with the same architecture as the original
        self.model = IntentModel(
            input_size=input_shape,
            hidden_size1=128,
            hidden_size2=64,
            output_size=output_shape
        ).to(self.device)
        return self.model
    
    def train(self, train_x, train_y, epochs=200, batch_size=5):
        # Convert numpy arrays to PyTorch tensors
        X = torch.FloatTensor(train_x).to(self.device)
        y = torch.FloatTensor(train_y).to(self.device)
        
        # Define loss function and optimizer (similar to the original SGD setup)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # Training loop
        history = {'loss': [], 'accuracy': []}
        for epoch in range(epochs):
            # Process in batches
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Calculate epoch metrics
            with torch.no_grad():
                outputs = self.model(X)
                loss = criterion(outputs, y)
                _, predicted = torch.max(outputs, 1)
                _, targets = torch.max(y, 1)
                correct = (predicted == targets).sum().item()
                accuracy = correct / len(X)
                
                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy)
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
                    
        # Save the model
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), 'models/intent_model.pth')
        return history
    
    def load_model(self):
        # Load model structure
        self.words = pickle.load(open('models/words.pkl', 'rb'))
        self.classes = pickle.load(open('models/classes.pkl', 'rb'))
        
        try:
            self.vocab = pickle.load(open('models/vocab.pkl', 'rb'))
        except:
            print("Vocab file not found, continuing without it.")
        
        # Create model with correct dimensions
        input_shape = len(self.words)
        output_shape = len(self.classes)
        self.build_model(input_shape, output_shape)
        
        # Load trained weights
        self.model.load_state_dict(torch.load('models/intent_model.pth'))
        self.model.eval()
    
    def predict_class(self, sentence, preprocessor, error_threshold=0.25):
        # Generate probabilities from the model
        input_data = preprocessor.prepare_input(sentence)
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        # Get the predictions
        with torch.no_grad():
            results = self.model(input_tensor)[0]
        
        # Convert tensor to numpy for further processing
        results = results.cpu().numpy()
        
        # Filter out predictions below threshold
        results = [[i, r] for i, r in enumerate(results) if r > error_threshold]
        
        # Sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
            
        return return_list