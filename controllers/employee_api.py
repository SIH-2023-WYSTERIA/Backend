from flask import MethodView
from dependencies import token_required

class EmployeeAPI(MethodView):
    decorators = [token_required(role='employee')]  

    def get(self):
        return "Employee GET endpoint"

    def post(self):
        return "Employee POST endpoint"