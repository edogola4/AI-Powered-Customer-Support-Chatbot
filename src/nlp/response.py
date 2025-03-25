import random
import json
import numpy as np
from src.nlp.sentiment import SentimentAnalyzer
from nltk.corpus import wordnet

def get_synonyms(word):
    """Return a list of synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def augment_text(text, replacement_prob=0.5):
    """
    Generate an augmented version of the text by replacing words with their synonyms.
    Each word is replaced with a random synonym with probability replacement_prob.
    """
    words = text.split()
    augmented_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and random.random() < replacement_prob:
            augmented_words.append(random.choice(synonyms))
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)

class ResponseGenerator:
    def __init__(self, intents_file):
        # Load intents from the JSON file
        self.intents = json.loads(open(intents_file).read())
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # For each intent, augment the available responses to enrich the dataset.
        for intent in self.intents['intents']:
            original_responses = intent.get('responses', [])
            augmented_responses = []
            for response in original_responses:
                # Generate a couple of augmented variations for each response.
                augmented_responses.append(augment_text(response))
                augmented_responses.append(augment_text(response))
            # Combine the original responses with the new augmented ones
            intent['responses'] = list(set(original_responses + augmented_responses))
        
    def get_response(self, intents_list, message_text):
        """Generate a response based on predicted intent and sentiment analysis."""
        if not intents_list:
            return "I'm not sure I understand. Could you rephrase that?"
        
        tag = intents_list[0]['intent']
        intent_data = None
        
        # Find the matching intent from the loaded intents
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                intent_data = intent
                break
        
        if not intent_data:
            return "I'm having trouble understanding. Could you try again?"
        
        # Analyze sentiment to potentially adjust the response tone
        sentiment = self.sentiment_analyzer.analyze(message_text)
        responses = intent_data['responses']
        
        # If the sentiment is highly negative, prepend an empathetic message
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.75:
            return f"I understand you might be frustrated. {random.choice(responses)}"
        
        return random.choice(responses)
