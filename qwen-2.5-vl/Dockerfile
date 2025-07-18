# Use the base Ollama image
FROM vllm/vllm-openai:latest

# Install curl
RUN apt-get update && apt-get install -y curl

# Expose the Ollama API port
EXPOSE 8000

CMD ["vllm", "serve", "Qwen/Qwen2.5-VL-3B-Instruct", "--trust-remote-code", "--enable-prefix-caching"]
