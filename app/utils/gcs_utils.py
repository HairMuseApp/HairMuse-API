from google.cloud import storage

# Initialize Google Cloud Storage client
storage_client = storage.Client()

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Download a file from Google Cloud Storage.
    
    Parameters:
        bucket_name (str): Name of the GCS bucket.
        source_blob_name (str): Path to the file in GCS.
        destination_file_name (str): Local path where the file will be saved.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")
