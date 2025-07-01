#!/bin/bash

# Activate virtual environment if it exists
if [ -d "antenv" ]; then
  source antenv/bin/activate
fi

# Start the FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000
