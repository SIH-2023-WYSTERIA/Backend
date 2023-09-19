from flask import jsonify, request
from flask.views import MethodView

from dependencies.db import MongoDB


class GetEmployee(MethodView,MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        employee_email = request.args.get("employee_email")
        company_id = request.args.get("company_id")

        # Create a filter dictionary based on the provided parameters
        filter_dict = {}
        if employee_email:
            filter_dict["employee_email"] = employee_email
        if company_id:
            filter_dict["company_id"] = company_id

        employees = list(self.db.employees.find(filter_dict))
        employees_json = [
            {"employee_email": emp["employee_email"], 
             "company_id": emp["company_id"], 
             "score": emp["score"],
             "num_conversations":emp['num_conversations'],
             "average_sentiment": emp['average_sentiment']}
            for emp in employees
        ]
        
        if not employees:
            return jsonify({"message": "Employee or Company does not exist"}), 400

        if(company_id):
            return jsonify({"employees": employees_json}), 200
        
        else:
            return jsonify({"employee": employees_json}), 200