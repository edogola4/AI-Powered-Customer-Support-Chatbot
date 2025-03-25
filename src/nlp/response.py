import random
import json
import numpy as np
from src.nlp.sentiment import SentimentAnalyzer
 
class ResponseGenerator:
     def __init__(self, intents_file):
         self.intents = json.loads(open(intents_file).read())
         self.sentiment_analyzer = SentimentAnalyzer()
         
     def get_response(self, intents_list, message_text):
         """Generate a response based on predicted intent"""
         if not intents_list:
             return "I'm not sure I understand. Could you rephrase that?"
         
         tag = intents_list[0]['intent']
         intent_data = None
         
         # Find the matching intent
         for intent in self.intents['intents']:
             if intent['tag'] == tag:
                 intent_data = intent
                 break
         
         if not intent_data:
             return "I'm having trouble understanding. Could you try again?"
         
         # Analyze sentiment to customize response
         sentiment = self.sentiment_analyzer.analyze(message_text)
         
         # Select response based on intent and sentiment
         responses = intent_data['responses']
         
         # Customize response based on sentiment if needed
         if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.75:
             # For highly negative sentiment, add empathy
             return f"I understand you might be frustrated. {random.choice(responses)}"
         
         return random.choice(responses)