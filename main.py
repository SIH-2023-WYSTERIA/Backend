from datetime import timedelta
from flask import Flask , request
from flask_jwt_extended import JWTManager
from flask_cors import CORS
import os
from dotenv import load_dotenv
from routes.public_routes import public_bp
from routes.private_routes import private_bp

from services import Finetune

load_dotenv()
app = Flask(__name__)
CORS(app,origins='*')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  # Replace with your secret key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=365) 
jwt = JWTManager(app)

# Register the public and private blueprints
app.register_blueprint(public_bp)
app.register_blueprint(private_bp)

@app.after_request
def after_request(response):
    if request.endpoint == 'private.send_conversation':
        Finetune()
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
