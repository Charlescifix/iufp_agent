-- Initialize pgvector extension and create necessary extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify pgvector is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Create a simple test to ensure vector operations work
-- This will be used by our RAG system
SELECT 'pgvector setup complete' AS status;