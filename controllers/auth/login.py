from flask import request, jsonify, MethodView
from flask_jwt_extended import create_access_token
from passlib.hash import pbkdf2_sha256 as sha256


class Login(MethodView):
    def post():
        data = request.get_json()

        username = data.get("username")
        password = data.get("password")

        if username not in users:
            return jsonify({"message": "User not found"}), 404

        user = users[username]

        if sha256.verify(password, user["password"]):
            # Generate a JWT token with user identity and role
            access_token = create_access_token(
                identity={"username": user["username"], "role": user["role"]}
            )
            return jsonify({"access_token": access_token}), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401
