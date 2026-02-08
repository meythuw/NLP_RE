import os
import boto3
import yaml
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

def get_s3_client():
    try:
        return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY"),
        region_name="us-east-1",   
    )

    except Exception as e:
        raise ValueError("Something was wrong with MiniO", e)
    
    
def upload_file(file_path: str, object_name: str, bucket: str = 'nlp-ie'):
    s3 = get_s3_client()
    bucket = bucket

    s3.upload_file(
        Filename=file_path,
        Bucket=bucket,
        Key=object_name
    )
    print(f"Uploaded {file_path} → s3://{bucket}/{object_name}")


def read_yaml_from_minio(key: str, bucket: str = 'nlp-ie') -> dict:
    s3 = get_s3_client()
    bucket = bucket

    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read()

    return yaml.safe_load(content)

# upload_file(file_path="/Users/nhatlan/Documents/nlp/config.yml", object_name='label-config.yml')

# print(read_yaml_from_minio(key="label-config.yml"))