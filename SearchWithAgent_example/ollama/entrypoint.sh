#!/bin/bash

echo "Starting Ollama server..."
ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 1
done

ollama pull llama3 
ollama pull llama3-groq-tool-use
ollama pull claude-3-5-sonnet-latest

wait $SERVE_PID