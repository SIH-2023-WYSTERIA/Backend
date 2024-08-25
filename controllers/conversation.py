import datetime
import uuid
import os
import tempfile
from bson import ObjectId
from flask import jsonify, request
from flask.views import MethodView
from werkzeug.utils import secure_filename
from .base import AdminAPI, EmployeeAPI
from dependencies import S3, MongoDB
from services import Model_Inference, Update_Employee_Stats, Finetune
import pytz

ALLOWED_EXTENSIONS = ["mp3", "wav"]
ALLOWD_FINETUNE_TAGS = [
    "generic",
    "hotel_booking",
    "car_rental",
    "event_ticketing",
    "cruise_booking",
    "train_ticketing",
    "bus_reservation",
    "vacation_rentals",
    "tour_booking",
    "restaurant_reservation",
    "conference_booking",
    "adventure_tours",
    "travel_package",
    "sightseeing_tours",
    "vacation_planning",
    "theme_park_tickets",
]

# Define the Indian time zone
indian_tz = pytz.timezone("Asia/Kolkata")

class SendConversation(EmployeeAPI, S3, MongoDB):
    def __init__(self):
        S3.__init__(self)
        MongoDB.__init__(self)

    def allowed_file(self, filename):
        return (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        )

    def get_next_index(self):
        last_document = self.db.conversations.find_one(sort=[("index", -1)])
        if last_document:
            return last_document["index"] + 1
        return 1  # If no documents exist, start from 1

    def post(self):
        # Extract employee details from the request body
        data = request.get_json()
        employee_email = data.get("employee_email")
        company_id = data.get("company_id")

        if not employee_email or not company_id:
            return jsonify({"message": "Employee email and company ID are required"}), 400

        # Check if a file was uploaded in the request
        if "file" not in request.files:
            return jsonify({"message": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"message": "No selected file"}), 400

        if not self.allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "message": "Invalid file extension. Only .wav or .mp3 files are allowed"
                    }
                ),
                400,
            )

        temp_dir = tempfile.mkdtemp()

        try:
            filename = str(uuid.uuid4()) + "." + file.filename.rsplit(".", 1)[1].lower()
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            success, error = self.upload_file(file_path, filename)

            if not success:
                return (
                    jsonify({"message": "Failed to upload file to S3", "error": error}),
                    500,
                )

            s3_url = self.generate_presigned_url(filename)
            index = self.get_next_index()
            inference = Model_Inference(file_path)

            _ = self.db.conversations.insert_one(
                {
                    "employee_email": employee_email,
                    "company_id": company_id,
                    "stream_url": s3_url,
                    "index": index,
                    "date": str(datetime.datetime.now(indian_tz).date()),
                    "time": str(datetime.datetime.now(indian_tz).time().strftime("%H:%M:%S")),
                    "inference": inference,
                }
            ).inserted_id

            Update_Employee_Stats(employee_email, inference["score"], inference["sentiment"])
            return jsonify({"message": "File uploaded and processed successfully"}), 200

        finally:
            os.remove(file_path)
            os.rmdir(temp_dir)

class GetAllConversations(AdminAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        data = request.get_json()
        company_id = data.get("company_id")

        if not company_id:
            return jsonify({"message": "Company ID is required"}), 400

        employee_email = request.args.get("employee_email")
        sentiment = request.args.get("sentiment")

        filter_dict = {"company_id": company_id}
        if employee_email:
            filter_dict["employee_email"] = employee_email
        if sentiment:
            filter_dict["inference.sentiment"] = sentiment

        conversations = list(self.db.conversations.find(filter_dict).sort([("_id", -1)]))
        conversations_json = [
            {
                "employee_email": conv["employee_email"],
                "company_id": conv["company_id"],
                "stream_url": conv["stream_url"],
                "inference": conv.get("inference", {}),
                "date": conv["date"],
                "time": conv["time"],
            }
            for conv in conversations
        ]

        return jsonify({"conversations": conversations_json}), 200

class GetConversationsByEmail(EmployeeAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get(self):
        data = request.get_json()
        employee_email = data.get("employee_email")

        if not employee_email:
            return jsonify({"message": "Employee email is required"}), 400

        sentiment = request.args.get("sentiment")

        filter_dict = {"employee_email": employee_email}
        if sentiment:
            filter_dict["inference.sentiment"] = sentiment

        conversations = list(self.db.conversations.find(filter_dict).sort([("_id", -1)]))
        conversations_json = [
            {
                "employee_email": conv["employee_email"],
                "company_id": conv["company_id"],
                "stream_url": conv["stream_url"],
                "inference": conv.get("inference", {}),
            }
            for conv in conversations
        ]

        return jsonify({"conversations": conversations_json}), 200

class SendFinetuneData(EmployeeAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def get_next_index(self):
        last_document = self.db.finetuning_data.find_one(sort=[("index", -1)])
        if last_document:
            return last_document["index"] + 1
        return 1  # If no documents exist, start from 1

    def post(self):
        data = request.get_json()
        employee_email = data.get("employee_email")
        company_id = data.get("company_id")

        if not employee_email or not company_id:
            return jsonify({"message": "Employee email and company ID are required"}), 400

        conversation_summary = data.get("conversation_summary")
        corrected_sentiment = data.get("corrected_sentiment")
        tags = data.get("tags", [])

        if not tags:
            tags = ["generic"]

        for tag in tags:
            if tag not in ALLOWD_FINETUNE_TAGS:
                return jsonify({"message": "given tag is invalid"}), 400

        if not conversation_summary or not corrected_sentiment:
            return jsonify({"message": "conversation and sentiment is required"}), 400

        _ = self.db.finetuning_data.insert_one(
            {
                "employee_email": employee_email,
                "company_id": company_id,
                "index": self.get_next_index(),
                "date": str(datetime.datetime.now(indian_tz).date()),
                "time": str(datetime.datetime.now(indian_tz).time().strftime("%H:%M:%S")),
                "conversation_summary": conversation_summary,
                "corrected_sentiment": corrected_sentiment,
                "tags": tags,
            }
        ).inserted_id

        return jsonify({"message": "updated finetuning data"}), 200

class GetFinetuneData(AdminAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def post(self):
        data = request.get_json()
        company_id = data.get("company_id")

        if not company_id:
            return jsonify({"message": "Company ID is required"}), 400

        tags = data.get("tags", [])
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")

        num = request.args.get("num")
        if not num:
            num = 10000
        else:
            num = int(num)

        if not tags:
            tags = ALLOWD_FINETUNE_TAGS
        for tag in tags:
            if tag not in ALLOWD_FINETUNE_TAGS:
                return jsonify({"message": "given tag is invalid"}), 400

        filter_dict = {"company_id": company_id, "tags": {"$in": tags}}

        if start_date_str and end_date_str:
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
            filter_dict["_id"] = {
                '$gte': ObjectId.from_datetime(start_date),
                '$lte': ObjectId.from_datetime(end_date)
            }

        filtered_documents = list(self.db.finetuning_data.find(filter_dict).sort('_id', -1).limit(num))

        finetune_data = [
            {
                "employee_email": doc["employee_email"],
                "date": doc["date"],
                "text": doc["conversation_summary"],
                "corrected_label": doc["corrected_sentiment"],
                "tags": doc.get("tags", []),
            }
            for doc in filtered_documents
        ]

        return jsonify({"finetune_data": finetune_data}), 200

class FinetuneData(AdminAPI, MongoDB):
    def __init__(self):
        MongoDB.__init__(self)

    def post(self):
        data = request.get_json()
        company_id = data.get("company_id")

        if not company_id:
            return jsonify({"message": "Company ID is required"}), 400

        tags = data.get("tags", [])
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")

        num = request.args.get("num")
        if not num:
            num = 10000
        else:
            num = int(num)

        if not tags:
            tags = ALLOWD_FINETUNE_TAGS
        for tag in tags:
            if tag not in ALLOWD_FINETUNE_TAGS:
                return jsonify({"message": "given tag is invalid"}), 400

        filter_dict = {"company_id": company_id, "tags": {"$in": tags}}

        if start_date_str and end_date_str:
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
            filter_dict["_id"] = {
                '$gte': ObjectId.from_datetime(start_date),
                '$lte': ObjectId.from_datetime(end_date)
            }

        filtered_documents = list(self.db.finetuning_data.find(filter_dict).sort('_id', -1).limit(num))
        if len(filtered_documents) < 1000:
            return jsonify({"error":"number of data rows must be greater than 1000"}), 400

        try:
            Finetune(filtered_documents)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        return jsonify({"message": "finetuned successfully"}), 200
