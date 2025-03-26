# src/api/recommendations.py
from flask import Blueprint, request, jsonify

recommendations_api = Blueprint('recommendations_api', __name__, url_prefix='/api/recommendations')

@recommendations_api.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id', '')
    
    # Dummy recommendation data (integrate your recommendation logic here)
    recommendations = [
        {
            "id": "PROD004",
            "name": "Bluetooth Speaker",
            "price": 79.99,
            "rating": 4.7,
            "reason": "Based on your interest in Electronics"
        },
        {
            "id": "PROD002",
            "name": "Smartphone Case",
            "price": 19.99,
            "rating": 4.2,
            "reason": "Customers also bought"
        }
    ]
    
    return jsonify({
        "success": True,
        "recommendations": recommendations
    })
