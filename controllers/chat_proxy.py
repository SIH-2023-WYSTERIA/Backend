from flask import jsonify, request
from flask.views import MethodView
import requests
from dependencies.db import MongoDB

URL = "http://0c7e-34-125-145-147.ngrok-free.app/chat"

class Chat(MethodView):
    def post(self):
        data = request.get_json()
        query = data.get("query")
        print(query)

        response = requests.post(URL, json={"query": query})
        print(response.text)

        response_data = response.json()
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
