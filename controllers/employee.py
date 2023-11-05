from datetime import datetime
from bson import ObjectId
from flask import jsonify, request
from .base import EmployeeAPI, AdminAPI

from dependencies.db import MongoDB

# TODO get employees implement and check
class GetEmployee(AdminAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        employee_email = request.args.get("employee_email")
        company_id = self.get_admin()["company_id"]

        # Create a filter dictionary based on the provided parameters
        filter_dict = {}
        filter_dict["company_id"] = company_id
        if employee_email:
            filter_dict["employee_email"] = employee_email
            

        employees = list(self.db.employees.find(filter_dict).sort([("_id", -1)]))
        employees_json = [
            {
                "employee_email": emp["employee_email"],
                "company_id": emp["company_id"],
                "score": emp["score"],
                "num_conversations": emp["num_conversations"],
                "num_positive_conversations": emp["num_positive_conversations"],
                "num_neutral_conversations": emp["num_neutral_conversations"],
                "num_negative_conversations": emp["num_negative_conversations"],
                "average_sentiment": emp["average_sentiment"],
            }
            for emp in employees
        ]

        if not employees:
            return jsonify({"message": "Employee or Company does not exist"}), 400

        if company_id:
            return jsonify({"employees": employees_json}), 200

        else:
            return jsonify({"employee": employees_json}), 200

class GetEmployeeStats(AdminAPI,MongoDB):
    def __init__(self):
        MongoDB.__init__(self)
    
    def get(self):
        company_id = self.get_admin()['company_id']
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        # Convert the date strings to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Query the database to retrieve conversations within the specified date range
        conversations = list(self.db.conversations.find({
            'company_id':company_id,
            '_id': {
                '$gte': ObjectId.from_datetime(start_date),
                '$lte': ObjectId.from_datetime(end_date)
            }
        }))

        num_positive = 0
        num_neutral = 0
        num_negative = 0
        for conv in conversations:
            if conv['inference']['sentiment'] == 'Positive':
                num_positive += 1
            if conv['inference']['sentiment'] == 'Neutral':
                num_neutral += 1
            if conv['inference']['sentiment'] == 'Negative':
                num_negative += 1
        
        # Convert the MongoDB documents to a list of dictionaries
        result = {
            'num_positive':num_positive,
            'num_negative':num_negative,
            'num_neutral':num_neutral,
            'total_num':num_positive + num_neutral + num_negative
        }

        return jsonify({'conversations': result})


class GetEmployeeByEmail(EmployeeAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        employee = self.get_employee()
        employee_email = employee["employee_email"]
        company_id = employee["company_id"]

        emp = self.db.employees.find_one({"employee_email": employee_email,"company_id":company_id})
        employee_json = {
            "employee_email": emp["employee_email"],
            "company_id": emp["company_id"],
            "score": emp["score"],
            "num_conversations": emp["num_conversations"],
            "num_positive_conversations": emp["num_positive_conversations"],
            "num_neutral_conversations": emp["num_neutral_conversations"],
            "num_negative_conversations": emp["num_negative_conversations"],
            "average_sentiment": emp["average_sentiment"],
        }

        if not employee:
            return jsonify({"message": "Employee does not exist"}), 400

        return jsonify({"employee": employee_json}), 200

