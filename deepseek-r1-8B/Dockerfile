# Use the base Ollama image
FROM ollama/ollama:latest

# Install curl
RUN apt-get update && apt-get install -y curl

# Set the environment variable for the model
ENV OLLAMA_MODEL=deepseek-r1:8b

# Expose the Ollama API port
EXPOSE 11434

# Start Ollama, pull the model, run it once with 'Hello' to initialize, then keep the container running
CMD /bin/sh -c "/bin/ollama serve & \
    while ! curl -s http://localhost:11434/api/tags > /dev/null; do sleep 1; done && \
    /bin/ollama pull $OLLAMA_MODEL && \
    /bin/ollama run $OLLAMA_MODEL 'Hello' && \
    tail -f /dev/null"
