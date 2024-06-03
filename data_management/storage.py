from typing import List, Dict
from pathlib import Path
import os, uuid
import boto3
from botocore.exceptions import ClientError


class S3StorageManager:
    def __init__(self):
        self._user_session_id: str = str(uuid.uuid4())
        self._s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv("AWS_REGION")
        )
        self._bucket_name: str = "intro-to-dsde-training-dt-app-s3".lower()  # Use your predefined bucket name here
        self._user_prefix: str = f"user-{self._user_session_id}/"  # Ensuring a trailing slash for folder simulation

    def upload_data_to_s3(self, local_data: str):
        object_name: str = f"{self._user_prefix}/{self._user_session_id}_data_file"
        self._s3_client.put_object(Bucket=self._bucket_name, Key=object_name, Body=local_data)
        return object_name

    def retrieve_file_from_s3(self, object_name: str):
        try:
            response = self._s3_client.get_object(Bucket=self._bucket_name, Key=object_name)
            data = response['Body'].read()
            return data
        except ClientError as e:
            print(f"Error retrieving the object {object_name}: {e}")
            return None

    def delete_user_data(self):
        # List all objects under the user's prefix and delete them
        response = self._s3_client.list_objects_v2(Bucket=self._bucket_name, Prefix=self._user_prefix)
        if 'Contents' in response:
            self._s3_client.delete_objects(
                Bucket=self._bucket_name,
                Delete={
                    'Objects': [{'Key': obj['Key']} for obj in response['Contents']]
                }
            )
