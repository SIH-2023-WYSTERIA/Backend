import uuid
from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity
from werkzeug.utils import secure_filename
from .base import AdminAPI, EmployeeAPI
from dependencies import S3,MongoDB

ALLOWED_EXTENSIONS = ['mp3','wav']
    
class SendConversation(EmployeeAPI,S3,MongoDB):
    def __init__(self):
        S3.__init__(self)
        MongoDB.__init__(self)

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def post(self):
        # Check if a file was uploaded in the request
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'}), 400

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        # Check if the file has a valid extension
        if not self.allowed_file(file.filename):
            return jsonify({'message': 'Invalid file extension. Only .wav files are allowed'}), 400

        # Securely save the file with a random name to avoid conflicts
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        # Upload the file using the inherited S3Manager
        success, error = self.upload_file(file, filename)
        if not success:
            return jsonify({'message': 'Failed to upload file to S3', 'error': error}), 500

        # Optionally, you can perform further processing on the uploaded file here
        user = self.get_employee()
        username = user['username']
        s3_file_key = filename
        _ = self.db.conversations.insert_one({'username': username, 's3_file_key': s3_file_key}).inserted_id
        return jsonify({'message': 'File uploaded successfully'}) ,200
    
class GetAllConversations(AdminAPI,MongoDB):
    def __init__(self):
        MongoDB.__init__(self)
    
    def get(self):
        # Query the database to retrieve all conversations
        conversations = list(self.db.conversations.find({}))
        # Transform MongoDB documents to a list of dictionaries (JSON serializable)
        conversations_json = [{"username": conv["username"], "s3_file_key": conv["s3_file_key"]} for conv in conversations]

        return jsonify({"conversations": conversations_json}), 200