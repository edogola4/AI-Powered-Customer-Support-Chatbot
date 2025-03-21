from flask import Blueprint, request, jsonify
from src.utils.crm_integration import CRMIntegration

# Initialize Blueprint
ticket_api = Blueprint('ticket_api', __name__)

# Initialize CRM integration
crm = CRMIntegration()

@ticket_api.route('/create', methods=['POST'])
def create_ticket():
    """Create a support ticket"""
    data = request.get_json()
    
    if not data or 'customer_id' not in data or 'issue_type' not in data or 'description' not in data:
        return jsonify({
            'error': 'Invalid request. Please provide customer_id, issue_type, and description.'
        }), 400
    
    result = crm.create_ticket(
        data['customer_id'],
        data['issue_type'],
        data['description']
    )
    
    return jsonify(result)

@ticket_api.route('/update/<ticket_id>', methods=['PUT'])
def update_ticket(ticket_id):
    """Update a support ticket"""
    data = request.get_json()
    
    if not data or 'status' not in data:
        return jsonify({
            'error': 'Invalid request. Please provide status.'
        }), 400
    
    notes = data.get('notes')
    result = crm.update_ticket(ticket_id, data['status'], notes)
    
    return jsonify(result)

@ticket_api.route('/escalate', methods=['POST'])
def escalate_to_agent():
    """Escalate conversation to human agent"""
    data = request.get_json()
    
    if not data or 'customer_id' not in data or 'conversation' not in data:
        return jsonify({
            'error': 'Invalid request. Please provide customer_id and conversation.'
        }), 400
    
    # Create a ticket for human follow-up
    ticket_result = crm.create_ticket(
        data['customer_id'],
        'escalation',
        f"Customer chat escalated to human agent. Conversation: {data['conversation']}"
    )
    
    return jsonify({
        'success': True,
        'ticket_id': ticket_result['ticket_id'],
        'message': 'Conversation escalated to human agent'
    })