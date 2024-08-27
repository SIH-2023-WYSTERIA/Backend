from datetime import timedelta
import subprocess
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
subprocess.run(["huggingface-cli", "login", "--token", os.getenv('HUGGINGFACE_TOKEN')])
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=365) 
jwt = JWTManager(app)

# Register the public and private blueprints
app.register_blueprint(public_bp)
app.register_blueprint(private_bp)


if __name__ == '__main__':
    # certfile = './ssl_keys/cert.pem'
    # keyfile = './ssl_keys/key.pem'

    # For https
    # Passphrase is "sih2023"
    # app.run(host='0.0.0.0', port=5000,ssl_context=(certfile, keyfile))

    # For http
    app.run(host='0.0.0.0', port=8080)
