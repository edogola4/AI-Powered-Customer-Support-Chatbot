from src.nlp.preprocessing import TextPreprocessor
from src.nlp.model import IntentClassifier

def main():
    # Initialize preprocessor
    preprocessor = TextPreprocessor('data/intents/intents.json')
    
    # Preprocess data
    words, classes, documents = preprocessor.preprocess()
    print(f"Unique lemmatized words: {len(words)}")
    print(f"Classes: {classes}")
    print(f"Documents: {len(documents)}")
    
    # Create training data
    train_x, train_y = preprocessor.create_training_data()
    print(f"Training data created - X: {train_x.shape}, Y: {train_y.shape}")
    
    # Build and train model
    classifier = IntentClassifier()
    model = classifier.build_model(len(words), len(classes))
    history = classifier.train(train_x, train_y, epochs=200)
    
    print("Model trained and saved!")
    
if __name__ == "__main__":
    main()