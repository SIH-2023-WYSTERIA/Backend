import uuid
import os
import tempfile
from flask import jsonify, request
from flask.views import MethodView
from flask_jwt_extended import get_jwt_identity
from werkzeug.utils import secure_filename
from .base import AdminAPI, EmployeeAPI
from dependencies import S3, MongoDB
from services import Model_Inference,Update_Employee_Stats

ALLOWED_EXTENSIONS = ["mp3", "wav"]



class SendConversation(EmployeeAPI, S3, MongoDB):
    def __init__(self):
        S3.__init__(self)
        MongoDB.__init__(self)

    def allowed_file(self, filename):
        return (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        )
    
    # Function to get the next available index
    def get_next_index(self):
        last_document = self.db.conversations.find_one(sort=[("index", -1)])
        if last_document:
            return last_document['index'] + 1
        return 1  # If no documents exist, start from 1

    def post(self):
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

            # Upload the file using the inherited S3Manager
            success, error = self.upload_file(file_path, filename)

            if not success:
                return jsonify({'message': 'Failed to upload file to S3', 'error': error}), 500

            # Generate the S3 URL for the uploaded file
            s3_url = self.generate_presigned_url(filename)

            employee = self.get_employee()
            employee_email = employee['employee_email']
            company_id = employee['company_id']
            index = self.get_next_index()
            inference = Model_Inference(file_path)  # Ensure Model_Inference accepts a file path

            _ = self.db.conversations.insert_one({'employee_email': employee_email, 
                                                  'company_id' : company_id,
                                                  'stream_url': s3_url, 
                                                  'index':index,
                                                  'inference': inference}).inserted_id

            Update_Employee_Stats(employee_email,inference["score"],inference["sentiment"])
            return jsonify({'message': 'File uploaded and processed successfully'}), 200

        finally:
            # Delete the temporary file and directory after processing
            os.remove(file_path)
            os.rmdir(temp_dir)


class GetAllConversations(AdminAPI,MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        company_id = self.get_admin()["company_id"]
        employee_email = request.args.get("employee_email")
        sentiment = request.args.get("sentiment")

        # Create a filter dictionary based on the provided parameters
        filter_dict = {}
        filter_dict["company_id"] = company_id
        
        if employee_email:
            filter_dict["employee_email"] = employee_email
        if sentiment:
            filter_dict["inference.sentiment"] = sentiment  # Adjust the field name as per your MongoDB schema

        # Query the database to retrieve all conversations
        conversations = list(self.db.conversations.find(filter_dict).sort([("_id", -1)]))
        # Transform MongoDB documents to a list of dictionaries (JSON serializable)
        conversations_json = [
            {"employee_email": conv["employee_email"], 
             "company_id": conv["company_id"], 
             "stream_url": conv["stream_url"],
             "inference":conv.get("inference", {})}
            for conv in conversations
        ]

        return jsonify({"conversations": conversations_json}), 200
    
class GetConversationsByEmail(EmployeeAPI,MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        employee_email = self.get_employee()["employee_email"]
        sentiment = request.args.get("sentiment")

        # Create a filter dictionary based on the provided parameters
        filter_dict = {}
        if employee_email:
            filter_dict["employee_email"] = employee_email
        if sentiment:
            filter_dict["inference.sentiment"] = sentiment  # Adjust the field name as per your MongoDB schema

        # Query the database to retrieve all conversations
        conversations = list(self.db.conversations.find(filter_dict).sort([("_id", -1)]))
        # Transform MongoDB documents to a list of dictionaries (JSON serializable)
        conversations_json = [
            {"employee_email": conv["employee_email"], 
             "company_id": conv["company_id"], 
             "stream_url": conv["stream_url"],
             "inference":conv.get("inference", {})}
            for conv in conversations
        ]

        return jsonify({"conversations": conversations_json}), 200