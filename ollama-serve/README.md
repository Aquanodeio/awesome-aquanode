# Ollama Serve with API Gating

This Docker image provides a secure way to serve Ollama models with API authentication via nginx reverse proxy. It combines Ollama's powerful LLM serving capabilities with Bearer token authentication to control access to your models.

## Features

- **Ollama Integration**: Full Ollama functionality for serving large language models
- **API Authentication**: Bearer token-based authentication via nginx
- **Reverse Proxy**: nginx reverse proxy with Lua scripting for secure access control
- **Easy Configuration**: Simple environment variable configuration
- **Streaming Support**: Full support for streaming responses and WebSocket connections

## Usage

### Basic Usage

```sh
cd ollama-serve
docker build -t ollama-serve .
docker run -p 7860:7860 \
  -e AUTH_TOKEN=your_secret_token \
  -e OLLAMA_MODEL=llama2:7b \
  ollama-serve
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AUTH_TOKEN` | Bearer token for API authentication | Yes | - |
| `OLLAMA_MODEL` | Ollama model to download and serve | Yes | - |

### Example with Different Models

```sh
# Serve Llama 2 7B
docker run -p 7860:7860 \
  -e AUTH_TOKEN=my_secure_token_123 \
  -e OLLAMA_MODEL=llama2:7b \
  ollama-serve

# Serve Mistral 7B
docker run -p 7860:7860 \
  -e AUTH_TOKEN=my_secure_token_123 \
  -e OLLAMA_MODEL=mistral:7b \
  ollama-serve

# Serve CodeLlama
docker run -p 7860:7860 \
  -e AUTH_TOKEN=my_secure_token_123 \
  -e OLLAMA_MODEL=codellama:7b \
  ollama-serve
```

## API Usage

Once the container is running, you can interact with the Ollama API using the Bearer token for authentication.

### Making Requests

All requests must include the `Authorization` header with your Bearer token:

```bash
# Generate text completion
curl -X POST http://localhost:7860/api/generate \
  -H "Authorization: Bearer your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b",
    "prompt": "Why is the sky blue?",
    "stream": false
  }'

# List available models
curl -X GET http://localhost:7860/api/tags \
  -H "Authorization: Bearer your_secret_token"

# Chat completion
curl -X POST http://localhost:7860/api/chat \
  -H "Authorization: Bearer your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Streaming Requests

The proxy fully supports Ollama's streaming capabilities:

```bash
curl -X POST http://localhost:7860/api/generate \
  -H "Authorization: Bearer your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b",
    "prompt": "Write a short story about a robot",
    "stream": true
  }'
```

## Security Features

- **Bearer Token Authentication**: All requests require a valid Bearer token
- **nginx Reverse Proxy**: Secure proxy layer with Lua-based authentication
- **Environment Variable Security**: Tokens are passed via environment variables, not hardcoded
- **Error Handling**: Proper HTTP status codes and JSON error responses

### Authentication Responses

- **200**: Request successful with valid token
- **401**: Unauthorized - Missing or invalid authorization header
- **401**: Unauthorized - Invalid Bearer token format
- **401**: Unauthorized - Invalid token

## Docker Compose Example

```yaml
version: '3.8'
services:
  ollama-serve:
    build: .
    ports:
      - "7860:7860"
    environment:
      - AUTH_TOKEN=your_very_secure_token_here
      - OLLAMA_MODEL=llama2:7b
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

## Available Models

You can use any model available in the Ollama model library. Popular options include:

- `llama2:7b`, `llama2:13b`, `llama2:70b`
- `mistral:7b`
- `codellama:7b`, `codellama:13b`
- `vicuna:7b`, `vicuna:13b`
- `orca-mini:3b`, `orca-mini:7b`
- `phi:2.7b`

Check the [Ollama Model Library](https://ollama.ai/library) for the complete list.

## Troubleshooting

### Container Won't Start
- Ensure `AUTH_TOKEN` and `OLLAMA_MODEL` environment variables are set
- Check that port 7860 is not already in use
- Verify the model name is correct (check Ollama library)

### Authentication Errors
- Ensure the `Authorization` header is included in all requests
- Use the format: `Authorization: Bearer your_token`
- Verify the token matches the `AUTH_TOKEN` environment variable

### Model Download Issues
- Large models may take time to download on first run
- Ensure sufficient disk space for model storage
- Check internet connectivity for model downloads

## Architecture

```
Client Request
     ↓
nginx (Port 7860)
     ↓
Lua Auth Check
     ↓
Ollama (Port 11434)
     ↓
Model Response
```

The nginx reverse proxy authenticates requests using Lua scripting before forwarding them to the Ollama service running on the internal port 11434.

## License

This project is released under the [AGPL License](../LICENSE.md).
