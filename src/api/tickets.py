# src/api/tickets.py
from flask import Blueprint, request, jsonify
import random

tickets_api = Blueprint('tickets_api', __name__, url_prefix='/api/tickets')

@tickets_api.route('/create', methods=['POST'])
def create_ticket():
    data = request.get_json()
    customer_id = data.get('customer_id', '')
    issue_type = data.get('issue_type', '')
    description = data.get('description', '')
    
    # Simulate ticket creation with a random ticket ID
    ticket_id = "TICKET-" + str(random.randint(1000, 9999))
    
    return jsonify({
        "success": True,
        "ticket_id": ticket_id,
        "message": f"Ticket created for customer {customer_id}"
    })
