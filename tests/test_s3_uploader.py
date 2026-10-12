"""
Tests for fuzzy_system_research.storage.s3_uploader.S3Uploader.

boto3 is replaced with an in-memory fake and load_dotenv is disabled, so no
network call is made and the real .env file is never read.
"""

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from fuzzy_system_research.storage import s3_uploader as s3_module
from fuzzy_system_research.storage.s3_uploader import S3Uploader


class FakeClient:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, local_filename, bucket_name, remote_filename):
        self.calls.append((local_filename, bucket_name, remote_filename))
        if self.error is not None:
            raise self.error


@pytest.fixture
def fake_client(monkeypatch):
    """Patch boto3 so S3Uploader gets a FakeClient; returns a setter for errors."""
    holder = {"client": FakeClient()}

    class FakeSession:
        def client(self, *args, **kwargs):
            return holder["client"]

    monkeypatch.setattr(s3_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(s3_module.boto3.session, "Session", FakeSession)
    return holder


def test_successful_upload_passes_arguments_and_prints(fake_client, capsys):
    S3Uploader().upload_file("local.csv", "bucket", "2026-10-12/all_forecasts.csv")
    assert fake_client["client"].calls == [
        ("local.csv", "bucket", "2026-10-12/all_forecasts.csv")
    ]
    assert "bucket/2026-10-12/all_forecasts.csv successful" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing.csv"),
        NoCredentialsError(),
        ClientError({"Error": {"Code": "NoSuchBucket", "Message": "nope"}}, "PutObject"),
    ],
    ids=["file-not-found", "no-credentials", "client-error"],
)
def test_upload_errors_propagate_without_success_message(fake_client, capsys, error):
    fake_client["client"] = FakeClient(error=error)
    with pytest.raises(type(error)):
        S3Uploader().upload_file("local.csv", "bucket", "key")
    assert "successful" not in capsys.readouterr().out
