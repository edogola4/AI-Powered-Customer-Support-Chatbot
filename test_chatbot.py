import requests
import json
import time
import random

def test_chat_api():
    """Test the chat API endpoint"""
    url = "http://localhost:5000/api/chat"
    
    # Test messages
    test_messages = [
        "Hello",
        "How do I track my order?",
        "I need help with a return",
        "What products do you recommend?",
        "Can I speak to a human?",
        "Thank you for your help"
    ]
    
    user_id = f"test_user_{random.randint(1000, 9999)}"
    
    print("Testing Chat API...")
    
    for message in test_messages:
        print(f"\nSending: '{message}'")
        
        start_time = time.time()
        response = requests.post(
            url,
            json={"message": message, "user_id": user_id}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: '{data['response']}'")
            print(f"Intents: {data['intents']}")
            print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    print("\nChat API test completed.")

def test_order_api():
    """Test the order tracking API"""
    url = "http://localhost:5000/api/orders/order/ORD12345"
    
    print("\nTesting Order API...")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Order status: {data['order']['status']}")
        print(f"Tracking number: {data['order']['tracking_number']}")
        print(f"Estimated delivery: {data['order']['estimated_delivery']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("Order API test completed.")

def test_recommendation_api():
    """Test the recommendation API"""
    url = "http://localhost:5000/api/recommendations/recommend"
    
    print("\nTesting Recommendation API...")
    
    response = requests.post(
        url,
        json={"user_id": "CUST001"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Number of recommendations: {len(data['recommendations'])}")
        for i, recommendation in enumerate(data['recommendations']):
            print(f"Recommendation {i+1}: {recommendation['name']} (${recommendation['price']})")
            print(f"Reason: {recommendation['reason']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("Recommendation API test completed.")

def test_ticket_api():
    """Test the ticket management API"""
    url = "http://localhost:5000/api/tickets/create"
    
    print("\nTesting Ticket API...")
    
    response = requests.post(
        url,
        json={
            "customer_id": "CUST001",
            "issue_type": "return",
            "description": "Customer wants to return a defective product"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Ticket created: {data['ticket_id']}")
        print(f"Message: {data['message']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("Ticket API test completed.")

if __name__ == "__main__":
    test_chat_api()
    test_order_api()
    test_recommendation_api()
    test_ticket_api()