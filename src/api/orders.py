# src/api/orders.py
from flask import Blueprint, jsonify

orders_api = Blueprint('orders_api', __name__, url_prefix='/api/orders')

@orders_api.route('/order/<order_id>', methods=['GET'])
def get_order(order_id):
    # Dummy order data (replace with your actual data source)
    order = {
        "status": "shipped",
        "customer_id": "CUST001",
        "items": [
            {"id": "PROD001", "name": "Wireless Headphones", "quantity": 1}
        ],
        "shipping_address": "123 Main St, Anytown, USA",
        "tracking_number": "TRK987654321",
        "estimated_delivery": "2025-03-25"
    }
    return jsonify({
        "success": True,
        "order": order
    })
