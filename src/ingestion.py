import asyncio
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import structlog

from .config import settings
from .logger import get_logger, log_function_call, log_function_result, log_security_event


class S3SecurityError(Exception):
    pass


class S3IngestionService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self._validate_credentials()
        self._setup_s3_client()
        
    def _validate_credentials(self) -> None:
        log_function_call(self.logger, "_validate_credentials")
        
        if not all([
            settings.aws_access_key_id,
            settings.aws_secret_access_key,
            settings.s3_bucket_name
        ]):
            error = S3SecurityError("Missing required AWS credentials or bucket name")
            log_security_event(
                "missing_credentials",
                {"service": "S3IngestionService"},
                "ERROR"
            )
            log_function_result(self.logger, "_validate_credentials", error=error)
            raise error
            
        log_function_result(self.logger, "_validate_credentials")
    
    def _setup_s3_client(self) -> None:
        log_function_call(self.logger, "_setup_s3_client")
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            
            # Test credentials by listing bucket
            self.s3_client.head_bucket(Bucket=settings.s3_bucket_name)
            
            self.logger.info("S3 client initialized successfully", bucket=settings.s3_bucket_name)
            log_function_result(self.logger, "_setup_s3_client")
            
        except NoCredentialsError as e:
            error = S3SecurityError(f"Invalid AWS credentials: {str(e)}")
            log_security_event(
                "invalid_credentials",
                {"service": "S3IngestionService", "error": str(e)},
                "ERROR"
            )
            log_function_result(self.logger, "_setup_s3_client", error=error)
            raise error
            
        except ClientError as e:
            if e.response['Error']['Code'] == '403':
                error = S3SecurityError(f"Access denied to bucket {settings.s3_bucket_name}")
                log_security_event(
                    "bucket_access_denied",
                    {"bucket": settings.s3_bucket_name, "error": str(e)},
                    "ERROR"
                )
            else:
                error = S3SecurityError(f"S3 client error: {str(e)}")
            log_function_result(self.logger, "_setup_s3_client", error=error)
            raise error
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        log_function_call(self.logger, "_sanitize_filename", filename=filename[:50])
        
        # Remove any path components
        safe_name = os.path.basename(filename)
        
        # Only allow specific file extensions
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        _, ext = os.path.splitext(safe_name.lower())
        
        if ext not in allowed_extensions:
            error = S3SecurityError(f"File extension {ext} not allowed")
            log_security_event(
                "invalid_file_extension",
                {"filename": filename, "extension": ext},
                "WARNING"
            )
            log_function_result(self.logger, "_sanitize_filename", error=error)
            raise error
        
        # Remove potentially dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        sanitized = ''.join(c for c in safe_name if c in safe_chars)
        
        if not sanitized or sanitized != safe_name:
            self.logger.warning("Filename sanitized", original=safe_name, sanitized=sanitized)
        
        log_function_result(self.logger, "_sanitize_filename", result=sanitized)
        return sanitized
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for integrity checking"""
        log_function_call(self.logger, "_calculate_file_hash", file_path=file_path)
        
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            file_hash = hash_md5.hexdigest()
            log_function_result(self.logger, "_calculate_file_hash", result=file_hash[:16])
            return file_hash
            
        except Exception as e:
            log_function_result(self.logger, "_calculate_file_hash", error=e)
            raise
    
    async def list_s3_objects(self, prefix: str = "", max_objects: int = 1000) -> List[Dict]:
        """List objects in S3 bucket with security validation"""
        log_function_call(self.logger, "list_s3_objects", prefix=prefix, max_objects=max_objects)
        
        # Validate max_objects to prevent resource exhaustion
        if max_objects > 10000:
            error = S3SecurityError("max_objects cannot exceed 10000")
            log_security_event(
                "resource_limit_exceeded",
                {"max_objects": max_objects, "limit": 10000},
                "WARNING"
            )
            log_function_result(self.logger, "list_s3_objects", error=error)
            raise error
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=settings.s3_bucket_name,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_objects}
            )
            
            objects = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Only include PDF files
                        if obj['Key'].lower().endswith('.pdf'):
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'],
                                'etag': obj['ETag'].strip('"')
                            })
            
            self.logger.info("Listed S3 objects", count=len(objects), prefix=prefix)
            log_function_result(self.logger, "list_s3_objects", result=f"{len(objects)} objects")
            return objects
            
        except ClientError as e:
            error = S3SecurityError(f"Failed to list S3 objects: {str(e)}")
            log_function_result(self.logger, "list_s3_objects", error=error)
            raise error
    
    async def download_file(self, s3_key: str, local_path: str) -> Tuple[str, str]:
        """Download file from S3 with security validation"""
        log_function_call(self.logger, "download_file", s3_key=s3_key, local_path=local_path[:50])
        
        # Validate and sanitize the S3 key
        if not s3_key or '..' in s3_key or s3_key.startswith('/'):
            error = S3SecurityError(f"Invalid S3 key: {s3_key}")
            log_security_event(
                "invalid_s3_key",
                {"s3_key": s3_key},
                "WARNING"
            )
            log_function_result(self.logger, "download_file", error=error)
            raise error
        
        # Ensure local path is within allowed directory
        local_path = os.path.abspath(local_path)
        allowed_base = os.path.abspath("data/raw")
        if not local_path.startswith(allowed_base):
            error = S3SecurityError(f"Local path outside allowed directory: {local_path}")
            log_security_event(
                "path_traversal_attempt",
                {"local_path": local_path, "allowed_base": allowed_base},
                "ERROR"
            )
            log_function_result(self.logger, "download_file", error=error)
            raise error
        
        try:
            # Get object metadata first for validation
            response = self.s3_client.head_object(Bucket=settings.s3_bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            
            # Prevent downloading excessively large files (>100MB)
            max_file_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_file_size:
                error = S3SecurityError(f"File too large: {file_size} bytes (max: {max_file_size})")
                log_security_event(
                    "file_size_exceeded",
                    {"s3_key": s3_key, "size": file_size, "max_size": max_file_size},
                    "WARNING"
                )
                log_function_result(self.logger, "download_file", error=error)
                raise error
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            self.s3_client.download_file(settings.s3_bucket_name, s3_key, local_path)
            
            # Calculate hash for integrity verification
            file_hash = self._calculate_file_hash(local_path)
            
            self.logger.info(
                "File downloaded successfully",
                s3_key=s3_key,
                local_path=local_path,
                size=file_size,
                hash=file_hash[:16]
            )
            
            log_function_result(self.logger, "download_file", result=f"Downloaded {file_size} bytes")
            return local_path, file_hash
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                error = S3SecurityError(f"File not found: {s3_key}")
            else:
                error = S3SecurityError(f"Failed to download file: {str(e)}")
            
            log_function_result(self.logger, "download_file", error=error)
            raise error
        except Exception as e:
            error = S3SecurityError(f"Unexpected error downloading file: {str(e)}")
            log_function_result(self.logger, "download_file", error=error)
            raise error
    
    async def sync_bucket_files(self, force_redownload: bool = False) -> List[Dict]:
        """Sync all PDF files from S3 bucket"""
        log_function_call(self.logger, "sync_bucket_files", force_redownload=force_redownload)
        
        try:
            # List all PDF objects in bucket
            s3_objects = await self.list_s3_objects()
            
            downloaded_files = []
            for obj in s3_objects:
                s3_key = obj['key']
                filename = self._sanitize_filename(os.path.basename(s3_key))
                local_path = os.path.join("data", "raw", filename)
                
                # Check if file already exists and is up to date
                should_download = force_redownload
                if not should_download and os.path.exists(local_path):
                    # Compare file modification times or ETags
                    local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path))
                    s3_mtime = obj['last_modified'].replace(tzinfo=None)
                    should_download = s3_mtime > local_mtime
                
                if should_download:
                    try:
                        file_path, file_hash = await self.download_file(s3_key, local_path)
                        downloaded_files.append({
                            's3_key': s3_key,
                            'local_path': file_path,
                            'size': obj['size'],
                            'hash': file_hash,
                            'last_modified': obj['last_modified']
                        })
                    except S3SecurityError as e:
                        self.logger.error("Failed to download file", s3_key=s3_key, error=str(e))
                        continue
                else:
                    self.logger.debug("File up to date, skipping", s3_key=s3_key)
            
            self.logger.info("Bucket sync completed", downloaded_count=len(downloaded_files), total_objects=len(s3_objects))
            log_function_result(self.logger, "sync_bucket_files", result=f"Downloaded {len(downloaded_files)} files")
            return downloaded_files
            
        except Exception as e:
            log_function_result(self.logger, "sync_bucket_files", error=e)
            raise


# Convenience function for external use
async def sync_s3_documents(force_redownload: bool = False) -> List[Dict]:
    """Sync documents from S3 - main entry point"""
    service = S3IngestionService()
    return await service.sync_bucket_files(force_redownload)