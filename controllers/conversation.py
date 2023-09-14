import uuid
import os
import tempfile
from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity
from werkzeug.utils import secure_filename
from .base import AdminAPI, EmployeeAPI
from dependencies import S3, MongoDB
from services import Model_Inference

ALLOWED_EXTENSIONS = ["mp3", "wav"]

from time import perf_counter


class SendConversation(EmployeeAPI, S3, MongoDB):
    def __init__(self):
        S3.__init__(self)
        MongoDB.__init__(self)

    def allowed_file(self, filename):
        return (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        )

    def post(self):
        start_time = perf_counter()
        # Check if a file was uploaded in the request
        if "file" not in request.files:
            return jsonify({"message": "No file part"}), 400

        file = request.files["file"]

        # Check if the file is empty
        if file.filename == "":
            return jsonify({"message": "No selected file"}), 400

        # Check if the file has a valid extension
        if not self.allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "message": "Invalid file extension. Only .wav or .mp3 files are allowed"
                    }
                ),
                400,
            )

        # Create a temporary directory to store the file
        temp_dir = tempfile.mkdtemp()


        try:
            # Securely save the file with a random name to avoid conflicts
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(temp_dir, filename)

            # Save the uploaded file to the temporary location
            file.save(file_path)

            # Get model inference using the file path
            inference = Model_Inference(file_path)  # Ensure Model_Inference accepts a file path

            # Upload the file using the inherited S3Manager
            success, error = self.upload_file(file, filename)

            if not success:
                return jsonify({'message': 'Failed to upload file to S3', 'error': error}), 500

            # Optionally, you can perform further processing on the uploaded file here
            user = self.get_employee()
            username = user['username']
            stream_url = self.generate_presigned_url(filename)

            _ = self.db.conversations.insert_one({'username': username, 'stream_url': stream_url, 'inference': inference}).inserted_id

            # Return the result of inference along with a success message
            return jsonify({'message': 'File uploaded and processed successfully', 'inference': inference}), 200

        finally:
            # Delete the temporary file and directory after processing
            os.remove(file_path)
            os.rmdir(temp_dir)
            latency = perf_counter() - start_time
            print(latency)


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