#!/bin/bash

echo "Starting Ollama server..."
ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 1
done

ollama pull llama3.2
ollama pull llama3-groq-tool-use:8b
ollama pull hengwen/watt-tool-8B

wait $SERVE_PID