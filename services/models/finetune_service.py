from dependencies import MongoDB
from .model_service import finetune
import csv

client = MongoDB()

threshold = 1000

def Finetune():
    max_index = client.db.conversations.find_one({}, sort=[("index", -1)])["index"]
    if(max_index and max_index % threshold != 0):
        return
    documents = client.db.conversations.find().sort("index", -1).limit(threshold)
    csv_file_path = write_to_csv(documents)
    finetune(csv_file_path)

def write_to_csv(documents):
    csv_file_path = 'finetune.csv'
    # Create and open the CSV file for writing
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row (column names) to the CSV file
        csv_writer.writerow(["text", "label"])  # Replace with your desired column names

        # Iterate through the top thousand documents and write each document to the CSV file
        for document in documents:
            # Extract the relevant data from the document and format it as needed
            text = document.get("summary", "")  # Replace with the actual field names
            label = document.get("sentiment", "")  # Replace with the actual field names

            # Write the data to the CSV file
            csv_writer.writerow([text, label])
    return csv_file_path