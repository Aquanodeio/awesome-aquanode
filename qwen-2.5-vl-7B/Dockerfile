# Use the base VLLM image
FROM vllm/vllm-openai:latest

# Install curl and OpenResty
RUN apt-get update && apt-get install -y curl gnupg2 ca-certificates lsb-release && \
    curl -fsSL https://openresty.org/package/pubkey.gpg | apt-key add - && \
    echo "deb http://openresty.org/package/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/openresty.list && \
    apt-get update && apt-get install -y openresty

# Set the environment variable for the model
ENV VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# Create OpenResty configuration
RUN mkdir -p /usr/local/openresty/nginx/conf/conf.d

# Create the nginx site configuration
RUN cat > /usr/local/openresty/nginx/conf/conf.d/vllm-proxy.conf << 'EOF'
server {
    listen 7860;
    server_name _;
    
    location / {
        # Validate Bearer token using Lua
        access_by_lua_block {
            local auth_header = ngx.var.http_authorization
            local expected_token = os.getenv("AUTH_TOKEN")
            
            if not auth_header then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - No authorization header"}')
                ngx.exit(401)
            end
            
            local token = auth_header:match("Bearer%s+(.+)")
            if not token then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - Invalid Bearer token format"}')
                ngx.exit(401)
            end
            
            if not expected_token or token ~= expected_token then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - Invalid token"}')
                ngx.exit(401)
            end
        }
        
        # Proxy to VLLM
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Handle streaming responses
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Create main nginx.conf for OpenResty
RUN cat > /usr/local/openresty/nginx/conf/nginx.conf << 'EOF'
env AUTH_TOKEN;
user www-data;
worker_processes auto;
pid /var/run/openresty.pid;

events {
    worker_connections 1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    
    access_log /usr/local/openresty/nginx/logs/access.log;
    error_log /usr/local/openresty/nginx/logs/error.log;
    
    gzip on;
    
    include /usr/local/openresty/nginx/conf/conf.d/*.conf;
}
EOF

# Expose only the reverse proxy port
EXPOSE 7860

# Override the base image's ENTRYPOINT
ENTRYPOINT []

# Start script that runs both VLLM and OpenResty
CMD ["/bin/sh", "-c", "vllm serve $VLLM_MODEL --trust-remote-code --enable-prefix-caching --device cpu & while ! curl -s http://localhost:8000/health > /dev/null; do sleep 1; done && /usr/local/openresty/bin/openresty -g 'daemon off;' & tail -f /dev/null"]