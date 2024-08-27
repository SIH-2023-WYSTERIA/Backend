from flask.views import MethodView
from flask import request, jsonify

class AdminAPI(MethodView):
    def get_admin(self):
        # Extract admin details from the request body instead of JWT
        data = request.get_json()
        return {
            'company_id': data.get('company_id'),
            'admin_email': data.get('admin_email')
        }

    def get(self):
        return "Admin GET endpoint"

    def post(self):
        return "Admin POST endpoint"
    
class EmployeeAPI(MethodView):
    def get_employee(self):
        # Extract employee details from the request body instead of JWT
        data = request.get_json()
        return {
            'employee_email': data.get('employee_email'),
            'company_id': data.get('company_id')
        }
    
    def get(self):
        return "Employee GET endpoint"

    def post(self):
        return "Employee POST endpoint"
