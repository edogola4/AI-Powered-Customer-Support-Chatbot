from flask import Blueprint, request, jsonify
from src.nlp.chatbot import Chatbot

# Initialize Blueprint
api = Blueprint('api', __name__)

# Initialize chatbot
chatbot = Chatbot()

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Chatbot API is running'
    })

@api.route('/chat', methods=['POST'])
def chat():
    """Process chat messages"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({
            'error': 'Invalid request. Please provide a message.'
        }), 400
    
    user_message = data['message']
    user_id = data.get('user_id', 'guest')
    
    # Get response from chatbot
    result = chatbot.get_response(user_message)
    
    return jsonify({
        'response': result['response'],
        'user_id': user_id,
        'intents': result['intents'],
        'timestamp': request.args.get('timestamp', 0)
    })