from flask import Blueprint, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Blueprint
order_api = Blueprint('order_api', __name__)

# Mock order database (would be a real database in production)
ORDERS = {
    'ORD12345': {
        'status': 'shipped',
        'customer_id': 'CUST001',
        'items': [
            {'id': 'PROD001', 'name': 'Wireless Headphones', 'quantity': 1},
        ],
        'shipping_address': '123 Main St, Anytown, USA',
        'tracking_number': 'TRK987654321',
        'estimated_delivery': '2025-03-25'
    },
    'ORD12346': {
        'status': 'processing',
        'customer_id': 'CUST002',
        'items': [
            {'id': 'PROD002', 'name': 'Smartphone Case', 'quantity': 1},
            {'id': 'PROD003', 'name': 'Screen Protector', 'quantity': 2}
        ],
        'shipping_address': '456 Oak St, Othertown, USA',
        'tracking_number': None,
        'estimated_delivery': '2025-03-28'
    }
}

@order_api.route('/order/<order_id>', methods=['GET'])
def get_order(order_id):
    """Get order details by ID"""
    if order_id in ORDERS:
        return jsonify({
            'success': True,
            'order': ORDERS[order_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Order not found'
        }), 404

@order_api.route('/tracking/<tracking_number>', methods=['GET'])
def track_shipment(tracking_number):
    """Track shipment by tracking number (mock)"""
    # In a real app, this would call a shipping carrier's API
    tracking_info = {
        'TRK987654321': {
            'status': 'in transit',
            'location': 'Distribution Center, Chicago, IL',
            'last_update': '2025-03-22T08:30:00',
            'estimated_delivery': '2025-03-25'
        }
    }
    
    if tracking_number in tracking_info:
        return jsonify({
            'success': True,
            'tracking_info': tracking_info[tracking_number]
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Tracking information not found'
        }), 404