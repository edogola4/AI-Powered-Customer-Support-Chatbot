import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CRMIntegration:
    def __init__(self):
        # In a real application, these would be environment variables
        self.api_key = os.getenv('CRM_API_KEY', 'dummy_key')
        self.base_url = os.getenv('CRM_BASE_URL', 'https://api.example-crm.com')
    
    def get_customer_info(self, customer_id):
        """
        Get customer information from CRM
        In a real application, this would make an API call to the CRM system
        """
        # Mock implementation
        customers = {
            'CUST001': {
                'id': 'CUST001',
                'name': 'Bran Don',
                'email': 'bran.don@example.com',
                'phone': '123-456-7890',
                'membership_level': 'Gold',
                'last_purchase_date': '2025-03-15'
            },
            'CUST002': {
                'id': 'CUST002',
                'name': 'Jane Smith',
                'email': 'jane.smith@example.com',
                'phone': '987-654-3210',
                'membership_level': 'Silver',
                'last_purchase_date': '2025-03-10'
            }
        }
        
        return customers.get(customer_id, {})
    
    def create_ticket(self, customer_id, issue_type, description):
        """Create a support ticket in the CRM system"""
        # In a real application, this would make an API call to create a ticket
        
        # Mock implementation
        ticket_id = f"TICKET-{hash(description) % 10000}"
        
        return {
            'success': True,
            'ticket_id': ticket_id,
            'message': f"Ticket created for customer {customer_id}"
        }
    
    def update_ticket(self, ticket_id, status, notes=None):
        """Update a support ticket in the CRM system"""
        # In a real application, this would make an API call to update a ticket
        
        # Mock implementation
        return {
            'success': True,
            'ticket_id': ticket_id,
            'message': f"Ticket {ticket_id} updated to status: {status}"
        }