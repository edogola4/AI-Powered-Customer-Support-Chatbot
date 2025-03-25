# src/api/chat.py
from flask import Blueprint, request, jsonify
import time

chat_api = Blueprint('chat_api', __name__, url_prefix='/api/chat')

@chat_api.route('', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    user_id = data.get('user_id', '')
    
    # Simple intent matching (replace with real NLP logic as needed)
    if 'order' in message.lower():
        response_text = "I'll help you track your order. Could you provide your order number?"
        intent = "order_status"
        probability = "0.92"
    elif 'hello' in message.lower():
        response_text = "Hello! How can I help you today?"
        intent = "greeting"
        probability = "0.95"
    elif 'bye' in message.lower():
        response_text = "Goodbye! Have a great day!"
        intent = "goodbye"
        probability = "0.95"
    else:
        response_text = "I'm not sure how to respond to that."
        intent = "unknown"
        probability = "0.50"
    
    timestamp = int(time.time())
    
    return jsonify({
        "response": response_text,
        "user_id": user_id,
        "intents": [
            {"intent": intent, "probability": probability}
        ],
        "timestamp": timestamp
    })
