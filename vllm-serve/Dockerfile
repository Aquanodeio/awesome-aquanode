# Use the base VLLM image
FROM vllm/vllm-openai:latest

# Install curl and OpenResty
RUN apt-get update && apt-get install -y curl gnupg2 ca-certificates lsb-release && \
    curl -fsSL https://openresty.org/package/pubkey.gpg | apt-key add - && \
    echo "deb http://openresty.org/package/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/openresty.list && \
    apt-get update && apt-get install -y openresty && \
    rm -rf /var/lib/apt/lists/*

# Create OpenResty configuration
RUN mkdir -p /usr/local/openresty/nginx/conf/conf.d

# Create the nginx site configuration
RUN cat > /usr/local/openresty/nginx/conf/conf.d/vllm-proxy.conf << 'EOF'
server {
    listen 7860 default_server;
    server_name _;
    
    location / {
        # Validate API key from query parameters using Lua
        access_by_lua_block {
            local expected_token = os.getenv("API_KEY")
            local args = ngx.req.get_uri_args()
            local token = args.api_key or args.key or args.token
            
            if not token then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - No API key provided. Include api_key parameter in query string"}')
                ngx.exit(401)
            end
            
            if not expected_token or token ~= expected_token then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - Invalid API key"}')
                ngx.exit(401)
            end
            
            -- Remove the api_key from query string before proxying
            args.api_key = nil
            args.key = nil
            args.token = nil
            local new_args = {}
            for k, v in pairs(args) do
                if type(v) == "table" then
                    for _, val in ipairs(v) do
                        table.insert(new_args, k .. "=" .. tostring(val))
                    end
                else
                    table.insert(new_args, k .. "=" .. tostring(v))
                end
            end
            
            if #new_args > 0 then
                ngx.var.args = table.concat(new_args, "&")
            else
                ngx.var.args = ""
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
env API_KEY;
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

# Create startup script
RUN cat > /start.sh << 'EOF'
#!/bin/bash
set -e

# Start VLLM in background
echo "Starting VLLM server..."
vllm serve $VLLM_MODEL --trust-remote-code --enable-prefix-caching &
VLLM_PID=$!

# Wait for VLLM to be ready with timeout
echo "Waiting for VLLM to be ready..."
timeout=120
while ! curl -s http://localhost:8000/health > /dev/null && [ $timeout -gt 0 ]; do 
    sleep 2
    timeout=$((timeout-2))
done

if [ $timeout -eq 0 ]; then
    echo "VLLM failed to start within 120 seconds"
    exit 1
fi

echo "VLLM is ready"

# Test OpenResty configuration
/usr/local/openresty/bin/openresty -t

# Start OpenResty in foreground
echo "Starting OpenResty proxy..."
echo "Setup complete. Server running on port 7860"
exec /usr/local/openresty/bin/openresty -g 'daemon off;'
EOF

RUN chmod +x /start.sh

# Expose only the reverse proxy port
EXPOSE 7860

# Override the base image's ENTRYPOINT
ENTRYPOINT []

# Use the startup script
CMD ["/start.sh"]
