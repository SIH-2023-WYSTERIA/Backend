import os
import tempfile
import uuid
from dependencies.db import MongoDB

from .model_service import finetune
import csv

client = MongoDB()

threshold = 1000

def Finetune(documents):
    csv_file_path = write_to_csv(documents)
    finetune(csv_file_path)

def write_to_csv(documents):
    temp_dir = tempfile.mkdtemp()
    filename = str(uuid.uuid4()) + '.csv' 
    csv_file_path = os.path.join(temp_dir, filename)
    # Create and open the CSV file for writing
    try:
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            # Write the header row (column names) to the CSV file
            csv_writer.writerow(["text", "label"])  # Replace with your desired column names

            # Iterate through the top thousand documents and write each document to the CSV file
            for document in documents:
                text = document["text"]  
                label = document["corrected_label"]  

                # Write the data to the CSV file
                csv_writer.writerow([text, label])

        absolute_csv_file_path = os.path.abspath(csv_file_path)
        print(absolute_csv_file_path)
        return absolute_csv_file_path
    finally:
        print("returned file")

