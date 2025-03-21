from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        # Initialize the sentiment analysis pipeline
        self.sentiment_pipeline = pipeline("sentiment-analysis")
    
    def analyze(self, text):
        """
        Analyze the sentiment of the given text
        Returns: {'label': 'POSITIVE/NEGATIVE', 'score': float}
        """
        result = self.sentiment_pipeline(text)[0]
        return result