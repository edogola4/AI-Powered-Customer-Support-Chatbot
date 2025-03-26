import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from flask_cors import CORS
from src.api.routes import api
from src.api.order_tracking import order_api
from src.api.product_recommendations import recommendations_api
from src.api.ticket_management import ticket_api

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(order_api, url_prefix='/api/orders')
    app.register_blueprint(recommendations_api, url_prefix='/api/recommendations')
    app.register_blueprint(ticket_api, url_prefix='/api/tickets')
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/health')
    def health():
        """Health check endpoint for monitoring"""
        return {
            'status': 'healthy',
            'version': '1.0.0'
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)