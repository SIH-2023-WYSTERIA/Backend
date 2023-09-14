from flask.views import MethodView
from flask import request
from dependencies import token_required

class AdminAPI(MethodView):
    decorators = [token_required(role='admin')] 

    def get_admin(self):
        return request.environ['decoded_jwt']

    def get(self):
        return "Admin GET endpoint"

    def post(self):
        return "Admin POST endpoint"
    
class EmployeeAPI(MethodView):
    decorators = [token_required(role='employee')]  
    
    def get_employee(self):
        return request.environ['decoded_jwt']
    
    def get(self):
        return "Employee GET endpoint"

    def post(self):
        return "Employee POST endpoint"