import numpy as np
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import google.auth
import os

storage_client = storage.Client()
bucket_name = "stylefinder-clothes"
destination_folder = "imagetwo"
bucket = storage_client.bucket(bucket_name)


blobs = bucket.list_blobs()

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for blob in blobs:

    destination_file_name = os.path.join(destination_folder,blob.name)
    os.makedirs(os.path.dirname(destination_file_name),exist_ok=True)
    blob.download_to_filename(destination_file_name)