import os
import boto3
from dotenv import load_dotenv

load_dotenv()


class S3:
    def __init__(self):
        self.s3 = boto3.client(
            service_name="s3",
            region_name=os.getenv("AWS_REGION_NAME"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def upload_file(self, file, filename):
        try:
            self.s3.upload_fileobj(file, os.getenv('S3_BUCKET_NAME'), filename)
        except Exception as e:
            return False, str(e)
        return True, None
    

if __name__ == "__main__":
    s3 = S3()

