from flask import MethodView
from dependencies import token_required

class AdminAPI(MethodView):
    decorators = [token_required(role='admin')] 

    def get(self):
        return "Admin GET endpoint"

    def post(self):
        return "Admin POST endpoint"