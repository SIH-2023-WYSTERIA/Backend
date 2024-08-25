import os
import boto3
from dotenv import load_dotenv

load_dotenv()


class S3:
    def __init__(self):
        self.s3 = boto3.client(
            service_name="s3",
            region_name=os.getenv("AWS_BUCKET_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        )

    def upload_file(self, file_path, filename):
        try:
            with open(file_path, 'rb') as file:
                self.s3.upload_fileobj(file, os.getenv('AWS_BUCKET_NAME'), filename)
        except Exception as e:
            return False, str(e)
        return True, None
    
    def generate_presigned_url(self, file_key):
        # Generate a pre-signed URL for the audio file
        presigned_url = self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': os.getenv('AWS_BUCKET_NAME'), 'Key': file_key},
            ExpiresIn=31536000  # Set an expiration time for the URL (e.g., 1 hour)
        )
        return presigned_url
    

if __name__ == "__main__":
    s3 = S3()

