import logging
from typing import List
import boto3
from botocore.client import Config
import os
from multiprocessing import Lock

lock = Lock()
logger = logging.getLogger()

def load_s3_file(s3_session, s3_bucket, s3_path, local_path):
    if not os.path.exists(local_path):
        logger.info(f"  Downloading file from bucket {s3_bucket} path {s3_path} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_session.Bucket(s3_bucket).download_file(s3_path, local_path)
        logger.info(f"  Downloaded file from bucket {s3_bucket} path {s3_path} to {local_path}!")
        return True

    return False

def load_s3_mlconfig_file(
                        inputs: List,
                        aws_endpoint_url: str,
                        aws_access_key_id: str,
                        aws_secret_access_key: str,
                        aws_bucket:str):
    lock.acquire()

    logger.info("Loading files from S3")
    try:
        s3_session = boto3.resource("s3",
                            endpoint_url=aws_endpoint_url,
                            config=Config(signature_version='s3v4'),
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)
        if inputs is not None:
            for file_path in inputs:
                s3_path = file_path
                load_s3_file(s3_session=s3_session, s3_bucket=aws_bucket, s3_path=s3_path, local_path=file_path)
    finally:
        lock.release()

    logger.info("Done Loading files from S3")
