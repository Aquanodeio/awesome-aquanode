# VLLM Serve with API Gating

This Docker image serves large language models using VLLM with Bearer token authentication via OpenResty reverse proxy.

## Features

- High-performance LLM serving with VLLM engine
- Bearer token authentication via OpenResty  
- OpenAI-compatible API endpoints
- Streaming support for real-time responses

## Usage

```sh
cd vllm-serve
docker build -t vllm-serve .
docker run -p 7860:7860 \
  -e AUTH_TOKEN=your_secret_token \
  -e VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf \
  vllm-serve
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AUTH_TOKEN` | Bearer token for API authentication | Yes |
| `VLLM_MODEL` | HuggingFace model name to serve | Yes |

## API Usage

All requests require the `Authorization: Bearer your_token` header.

### Chat Completions

```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Text Completions

```bash
curl -X POST http://localhost:7860/v1/completions \
  -H "Authorization: Bearer your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "The future of AI is",
    "max_tokens": 100
  }'
```

### Health Check

```bash
curl -X GET http://localhost:7860/health \
  -H "Authorization: Bearer your_secret_token"
```

## Supported Models

Popular VLLM-compatible models:
- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`  
- `codellama/CodeLlama-7b-Instruct-hf`
- `lmsys/vicuna-7b-v1.5`

Check [VLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for the complete list.
