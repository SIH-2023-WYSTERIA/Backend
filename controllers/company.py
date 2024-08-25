from datetime import datetime
from bson import ObjectId
from flask import jsonify, request
from flask.views import MethodView
from controllers.base import AdminAPI
from dependencies.db import MongoDB

class GetStats(AdminAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        data = request.get_json()
        company_id = data.get("company_id")

        if not company_id:
            return jsonify({"message": "Company ID is required"}), 400

        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        if not start_date_str or not end_date_str:
            return jsonify({"message": "Start date and end date are required"}), 400

        # Convert the date strings to datetime objects
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({"message": "Invalid date format, should be YYYY-MM-DD"}), 400

        # Query the database to retrieve conversations within the specified date range
        conversations = list(self.db.conversations.find({
            'company_id': company_id,
            '_id': {
                '$gte': ObjectId.from_datetime(start_date),
                '$lte': ObjectId.from_datetime(end_date)
            }
        }))

        num_positive, num_neutral, num_negative = 0, 0, 0
        for conv in conversations:
            sentiment = conv['inference']['sentiment']
            if sentiment == 'Positive':
                num_positive += 1
            elif sentiment == 'Neutral':
                num_neutral += 1
            elif sentiment == 'Negative':
                num_negative += 1

        # Convert the MongoDB documents to a list of dictionaries
        result = {
            'num_positive': num_positive,
            'num_negative': num_negative,
            'num_neutral': num_neutral,
            'total_num': num_positive + num_neutral + num_negative
        }

        return jsonify({'conversations': result})
