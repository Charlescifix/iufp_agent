from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 20.0
    openai_max_retries: int = 2
    query_embedding_cache_size: int = 256
    
    # Security Configuration
    api_key_header: str = "X-API-Key"
    admin_api_key: str = ""
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600
    # Number of reverse proxies in front of the app (Railway edge = 1); 0 = use socket IP
    trusted_proxy_hops: int = 1

    # Application Configuration
    debug: bool = False
    log_level: str = "INFO"
    # ~1000 chars keeps table rows with their labels (500 split figures from what they describe)
    max_chunk_size: int = 1000
    chunk_overlap: int = 150
    max_retrieval_results: int = 5
    # Minimum cosine similarity for a chunk to be used as context. Measured on the IUFP corpus:
    # in-scope questions scored 0.43-0.71, out-of-scope 0.07-0.35.
    min_relevance_score: float = 0.38
    max_output_tokens: int = 200
    embedding_dimension: int = 1536
    response_cache_ttl_seconds: int = 180
    response_cache_max_entries: int = 200
    healthcheck_cache_ttl_seconds: int = 30
    
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
    
    # Unknown keys (e.g. a temporary variable in .env) are ignored rather than stopping startup
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

settings = Settings()
