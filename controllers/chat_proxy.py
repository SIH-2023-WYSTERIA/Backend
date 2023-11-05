from flask import jsonify, request
from flask.views import MethodView
import requests
from dependencies.db import MongoDB

URL = "https://2112-35-224-121-60.ngrok-free.app/chat"

class Chat(MethodView):
    def post(self):
        data = request.get_json()
        query = data.get("query")
        print(query)

        response = requests.post(URL, json={"query": query})
        print(response.text)

        # Access the JSON content of the response
        response_data = response.json()

        # Return a valid JSON response
        return jsonify({'answer': response_data.get('answer', 'N/A')})

class ChatFinetune(MethodView,MongoDB):
    def post(self):
        data = request.get_json()
        conversations = data.get("conversations")
        rating = data.get("rating")
        _ = self.db.chat_finetune.insert_one({
            "conversations":conversations,
            "rating":rating
        }).inserted_id

        return jsonify({"message": "updated chat finetuning data"}), 200
