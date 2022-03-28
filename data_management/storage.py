from typing import List, Dict
from pathlib import Path
import os, uuid
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError


class BlobStorageManager:
    def __init__(self):
        self._user_session_id: str = str(uuid.uuid4())
        self._blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
            os.getenv("AZURE_BLOB_CONNECTION_STRING")
        )
        self._user_container_name: str = f"user-{self._user_session_id}-container"
        self._container_client: ContainerClient = None

    def _create_user_container(self) -> None:
        try:
            self._container_client = self._blob_service_client.create_container(
                self._user_container_name
            )
        except ResourceExistsError:
            print("The container already exists.")

    def upload_data_to_blob_container(self, local_data: str):
        self._create_user_container()
        blob_file_name: str = f"{self._user_session_id}_data_file"
        self._container_client.upload_blob(name=blob_file_name, data=local_data, overwrite=True)
        return blob_file_name

    def retrieve_file_from_blob_container(self, blob_file_name: str):
        data = self._container_client.download_blob(blob_file_name).readall()
        return data

    def delete_user_blob_container(self):
        self._container_client.delete_container()
