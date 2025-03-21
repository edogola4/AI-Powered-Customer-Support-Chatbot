from flask import Flask, render_template
from flask_cors import CORS
from src.api.routes import api
from src.api.order_tracking import order_api
from src.api.product_recommendations import recommendation_api
import os

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(order_api, url_prefix='/api/orders')
    app.register_blueprint(recommendation_api, url_prefix='/api/recommendations')
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))