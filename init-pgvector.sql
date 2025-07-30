-- Initialize pgvector extension for the IUFP RAG system
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify pgvector installation
SELECT 'pgvector extension created successfully' AS status;
SELECT version() AS postgresql_version;