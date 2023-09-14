from flask import jsonify
from .base import AdminAPI
from dependencies import S3, MongoDB

ALLOWED_EXTENSIONS = ["mp3", "wav"]


class GetAllConversations(AdminAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        # Query the database to retrieve all conversations
        conversations = list(self.db.conversations.find({}))
        # Transform MongoDB documents to a list of dictionaries (JSON serializable)
        conversations_json = [
            {"username": conv["username"], "stream_url": conv["stream_url"],"inference":conv.get("inference", {})}
            for conv in conversations
        ]

        return jsonify({"conversations": conversations_json}), 200