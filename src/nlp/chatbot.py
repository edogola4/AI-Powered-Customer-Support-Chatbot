from src.nlp.preprocessing import TextPreprocessor
from src.nlp.model import IntentClassifier
from src.nlp.response import ResponseGenerator

class Chatbot:
    def __init__(self, intents_file='data/intents/intents.json'):
        # Initialize the text preprocessor
        self.preprocessor = TextPreprocessor(intents_file)
        self.preprocessor.preprocess()
        
        # Initialize the intent classifier
        self.classifier = IntentClassifier()
        
        try:
            # Load the trained model
            self.classifier.load_model()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please run train_model.py first")
            
        # Initialize the response generator
        self.response_generator = ResponseGenerator(intents_file)
        
    def get_response(self, message):
        """
        Process the user message and return an appropriate response
        """
        # Predict the intent
        intents = self.classifier.predict_class(message, self.preprocessor)
        
        # Generate a response
        response = self.response_generator.get_response(intents, message)
        
        return {
            'response': response,
            'intents': intents
        }