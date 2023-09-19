import uuid
from flask import jsonify, request
from flask.views import MethodView
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from dependencies.db import MongoDB
import re

def checkEmailFormat(s):
    pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    if re.match(pat,s):
        return True
    else:
        return False


class BaseAuthAPI(MethodView,MongoDB):
    def __init__(self):
        super().__init__()

class RegisterCompany(BaseAuthAPI):
    def post(self):
        data = request.get_json()
        company_name = data.get('company_name')
        company_registration_id = data.get('company_registration_id')
        password = data.get('password')
        role = 'admin'

        if not company_name or not company_registration_id or not password:
            return jsonify({'message': 'Missing field'}), 400

        # Check if the username already exists
        if self.db.companies.find_one({'company_registration_id': company_registration_id}):
            return jsonify({'message': 'Company already exists'}), 400

        # Hash the password
        hashed_password = generate_password_hash(password, method='sha256')

        # Generate Company Id
        company_id = str(uuid.uuid4())

        # Insert user data into the database
        _ = self.db.companies.insert_one({'company_name': company_name, 
                                                'company_registration_id':company_registration_id,
                                                'company_id':company_id,
                                                'password': hashed_password, 
                                                'role':role}).inserted_id

        return jsonify({'message': 'Company registered successfully', 'company_id': company_id}), 201
    
class RegisterEmployee(BaseAuthAPI):
    def post(self):
        data = request.get_json()
        employee_email = data.get('employee_email')
        company_id = data.get('company_id')
        password = data.get('password')
        role = 'employee'

        if not employee_email or not password or not company_id:
            return jsonify({'message': 'Username and password and company id are required'}), 400
        
        if not checkEmailFormat(employee_email):
            return jsonify({'message': 'Email format invalid'}), 400

        # Check if the username already exists
        if self.db.employees.find_one({'employee_email': employee_email,'company_id':company_id}):
            return jsonify({'message': 'employee_email already exists'}), 400
        
        # Check if company_id is valid
        if not self.db.companies.find_one({'company_id':company_id}):
            return jsonify({'message': 'company_id does not exist'}), 400

        # Hash the password
        hashed_password = generate_password_hash(password, method='sha256')

        # Insert user data into the database
        user_id = self.db.employees.insert_one({'employee_email': employee_email,
                                                'company_id':company_id , 
                                                'password': hashed_password, 
                                                'score':0,
                                                'num_conversations':0,
                                                'cumulative_score':0,
                                                'average_sentiment':'Neutral',
                                                'role':role}).inserted_id

        return jsonify({'message': 'Employee registered successfully', 'employee_email': employee_email}), 201


class CompanyLogin(BaseAuthAPI):
    def post(self):
        data = request.get_json()

        company_id = data.get("company_id")
        password = data.get("password")

        company = self.db.companies.find_one({'company_id': company_id})


        if not company or not check_password_hash(company['password'], password):
            return jsonify({'message': 'Invalid username or password'}), 401

        if check_password_hash(company["password"],password):
            # Generate a JWT token with user identity and role
            access_token = create_access_token(
                identity={"company_id": company["company_id"], "role": company["role"]}
            )
            return jsonify({"access_token": access_token}), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401

class EmployeeLogin(BaseAuthAPI):
    def post(self):
        data = request.get_json()

        employee_email = data.get("employee_email")
        password = data.get("password")

        employee = self.db.employees.find_one({'employee_email': employee_email})

        print(employee)
        if not employee or not check_password_hash(employee['password'], password):
            return jsonify({'message': 'Invalid email or password'}), 401

        if check_password_hash(employee["password"],password):
            # Generate a JWT token with user identity and role
            access_token = create_access_token(
                identity={"employee_email": employee["employee_email"],
                          'company_id': employee["company_id"],
                          "role": employee["role"]}
            )
            return jsonify({"access_token": access_token}), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401