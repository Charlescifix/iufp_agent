#!/bin/bash

# Set default port if PORT is not set
if [ -z "$PORT" ]; then
    PORT=8000
fi

echo "Starting IUFP Chat API on port $PORT"
echo "Environment: PORT=$PORT"

# Start the application
exec uvicorn src.chat_api:app --host 0.0.0.0 --port "$PORT" --workers 1