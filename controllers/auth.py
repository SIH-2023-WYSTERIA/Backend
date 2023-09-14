from flask import jsonify, request
from flask.views import MethodView
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from dependencies.db import MongoDB


class BaseAuthAPI(MethodView,MongoDB):
    def __init__(self):
        super().__init__()

class Register(BaseAuthAPI):
    def post(self):
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        role = data.get('role')

        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400

        # Check if the username already exists
        if self.db.users.find_one({'username': username}):
            return jsonify({'message': 'Username already exists'}), 400

        # Hash the password
        hashed_password = generate_password_hash(password, method='sha256')

        # Insert user data into the database
        user_id = self.db.users.insert_one({'username': username, 'password': hashed_password, 'role':role}).inserted_id

        return jsonify({'message': 'User registered successfully', 'user_id': str(user_id)}), 201

class Login(BaseAuthAPI):
    def post(self):
        data = request.get_json()

        username = data.get("username")
        password = data.get("password")

        user = self.db.users.find_one({'username': username})


        if not user or not check_password_hash(user['password'], password):
            return jsonify({'message': 'Invalid username or password'}), 401

        if check_password_hash(user["password"],password):
            # Generate a JWT token with user identity and role
            access_token = create_access_token(
                identity={"username": user["username"], "role": user["role"]}
            )
            return jsonify({"access_token": access_token}), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401
