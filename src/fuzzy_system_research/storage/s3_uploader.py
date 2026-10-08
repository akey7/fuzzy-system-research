import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.client import Config


class S3Uploader:
    """
    Manages uploads of optimization and modeling data to S3.
    """
    
    def __init__(self):
        """
        Instantiate class and acquire values from environment variables.
        """
        load_dotenv()
        self.endpoint_url = os.getenv("FSF_FRONT_END_BUCKET_ENDPOINT")
        self.aws_access_key_id = os.getenv("FSF_FRONT_END_BUCKET_KEY_ID")
        self.aws_secret_access_key = os.getenv("FSF_FRONT_END_BUCKET_RWDELETE")
        self.region_name = os.getenv("FSF_FRONT_END_BUCKET_REGION")

    def upload_file(self, local_filename, bucket_name, remote_filename):
        """
        Uploads a file to S3. Prints status to the console.

        Parameters
        ----------
        local_filename : str
            Local path to file.

        bucket_name : str
            Name of bucket to upload to

        remote_filename : str
            Filename to upload to on the remote bucket.
        """
        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        try:
            client.upload_file(
                local_filename,
                bucket_name,
                remote_filename,
            )
            print(f"Upload of {bucket_name}/{remote_filename} successful!")
        except FileNotFoundError:
            print("The file was not found")
        except NoCredentialsError:
            print("Credentials not available")
