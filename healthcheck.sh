#!/bin/bash

# Health check script that uses the same PORT logic as start.sh
if [ -z "$PORT" ]; then
    PORT=8000
fi

curl -f http://localhost:$PORT/health