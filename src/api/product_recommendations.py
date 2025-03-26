from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd

# Initialize Blueprint
recommendations_api = Blueprint('recommendations_api', __name__, url_prefix='/api/product_recommendations')
#orders_api = Blueprint('orders_api', __name__, url_prefix='/api/orders')


# Mock product database
PRODUCTS = {
    'PROD001': {'name': 'Wireless Headphones', 'category': 'Electronics', 'price': 99.99, 'rating': 4.5},
    'PROD002': {'name': 'Smartphone Case', 'category': 'Accessories', 'price': 19.99, 'rating': 4.2},
    'PROD003': {'name': 'Screen Protector', 'category': 'Accessories', 'price': 9.99, 'rating': 3.9},
    'PROD004': {'name': 'Bluetooth Speaker', 'category': 'Electronics', 'price': 79.99, 'rating': 4.7},
    'PROD005': {'name': 'Wireless Charger', 'category': 'Electronics', 'price': 29.99, 'rating': 4.0},
    'PROD006': {'name': 'USB-C Cable', 'category': 'Accessories', 'price': 12.99, 'rating': 4.3}
}

# Mock purchase history
PURCHASE_HISTORY = {
    'CUST001': ['PROD001', 'PROD005'],
    'CUST002': ['PROD002', 'PROD003']
}

@recommendations_api.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Get product recommendations based on user ID or product ID"""
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'Invalid request'
        }), 400
    
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    
    recommendations = []
    
    if user_id and user_id in PURCHASE_HISTORY:
        # User-based recommendations (collaborative filtering concept)
        recommendations = get_user_recommendations(user_id)
    elif product_id and product_id in PRODUCTS:
        # Item-based recommendations
        recommendations = get_product_recommendations(product_id)
    else:
        # General recommendations
        recommendations = get_popular_products()
    
    return jsonify({
        'success': True,
        'recommendations': recommendations
    })

def get_user_recommendations(user_id):
    """Get recommendations based on user purchase history"""
    purchased_products = PURCHASE_HISTORY.get(user_id, [])
    
    # Get products from the same categories
    categories = set()
    for prod_id in purchased_products:
        if prod_id in PRODUCTS:
            categories.add(PRODUCTS[prod_id]['category'])
    
    recommendations = []
    for prod_id, product in PRODUCTS.items():
        if prod_id not in purchased_products and product['category'] in categories:
            recommendations.append({
                'id': prod_id,
                'name': product['name'],
                'price': product['price'],
                'rating': product['rating'],
                'reason': f"Based on your interest in {product['category']}"
            })
    
    return sorted(recommendations, key=lambda x: x['rating'], reverse=True)[:3]

def get_product_recommendations(product_id):
    """Get recommendations based on a specific product"""
    product = PRODUCTS.get(product_id)
    
    if not product:
        return []
    
    category = product['category']
    price_range = (product['price'] * 0.7, product['price'] * 1.3)
    
    recommendations = []
    for prod_id, prod in PRODUCTS.items():
        if (prod_id != product_id and 
            prod['category'] == category and 
            price_range[0] <= prod['price'] <= price_range[1]):
            recommendations.append({
                'id': prod_id,
                'name': prod['name'],
                'price': prod['price'],
                'rating': prod['rating'],
                'reason': "Customers also bought"
            })
    
    return sorted(recommendations, key=lambda x: x['rating'], reverse=True)[:3]

def get_popular_products():
    """Get generally popular products"""
    products = [{'id': prod_id, **product} for prod_id, product in PRODUCTS.items()]
    return sorted(products, key=lambda x: x['rating'], reverse=True)[:3]
