import os
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # AWS Configuration
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""
    
    # Database Configuration
    database_url: str = ""
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    
    # Railway Database Variables (alternative)
    pghost: str = ""
    pgport: int = 5432
    pgdatabase: str = ""
    pguser: str = ""
    pgpassword: str = ""
    
    # OpenAI Configuration
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4-turbo"
    
    # Security Configuration
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    admin_api_key: str = ""
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600
    
    # Application Configuration
    debug: bool = False
    log_level: str = "INFO"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_results: int = 10
    embedding_dimension: int = 3072
    
    # CORS Configuration
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"
    allowed_methods: str = "GET,POST,PUT,DELETE"
    allowed_headers: str = "*"
    
    def get_cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(',')]
    
    def get_cors_methods(self) -> List[str]:
        return [method.strip() for method in self.allowed_methods.split(',')]
    
    def get_database_url(self) -> str:
        """Get properly formatted database URL, converting postgres:// to postgresql://"""
        if self.database_url:
            # Convert deprecated postgres:// to postgresql://
            if self.database_url.startswith('postgres://'):
                return self.database_url.replace('postgres://', 'postgresql://', 1)
            return self.database_url
        return ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()