import os
from dotenv import load_dotenv
import boto3
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

    def upload_file(
        self, local_filename: str, bucket_name: str, remote_filename: str
    ) -> None:
        """
        Uploads a file to S3. Prints a message after a successful upload.

        Errors are not caught: any failure propagates to the caller, so a
        failed upload stops the run instead of being reported as a success.

        Parameters
        ----------
        local_filename : str
            Local path to file.

        bucket_name : str
            Name of bucket to upload to

        remote_filename : str
            Filename to upload to on the remote bucket.

        Raises
        ------
        FileNotFoundError
            If local_filename does not exist.

        botocore.exceptions.NoCredentialsError
            If no credentials are available.

        boto3.exceptions.S3UploadFailedError, botocore.exceptions.ClientError
            If S3 rejects the upload (e.g. a wrong bucket name or
            insufficient permissions).
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
        client.upload_file(
            local_filename,
            bucket_name,
            remote_filename,
        )
        print(f"Upload of {bucket_name}/{remote_filename} successful!")
