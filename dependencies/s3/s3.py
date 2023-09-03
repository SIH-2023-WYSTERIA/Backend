import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.resource(
    service_name='s3',
    region_name=os.getenv('AWS_REGION_NAME'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def list_buckets():
    # Print out bucket names
    for bucket in s3.buckets.all():
        print(bucket.name)

def list_objects(bucket_name):
    response = s3.list_objects_v2(Bucket=bucket_name)
    objects = [obj['Key'] for obj in response['Contents']]
    return objects

if __name__ == '__main__':
    list_buckets()
