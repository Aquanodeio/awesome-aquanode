# Use the base Ollama image
FROM ollama/ollama:latest

# Install curl, nginx, lua module, and lua-cjson
RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    libnginx-mod-http-lua \
    lua-cjson \
    && rm -rf /var/lib/apt/lists/*

# Create nginx configuration
RUN rm -f /etc/nginx/sites-enabled/default

# Create the nginx site configuration
RUN cat > /etc/nginx/sites-available/ollama-proxy << 'EOF'
server {
    listen 7860;
    server_name _;
    
    location / {
        # Validate API key using Lua
        access_by_lua_block {
            local expected_token = os.getenv("API_KEY")
            local token = nil
            
            -- Try to get token from query parameter
            local args = ngx.req.get_uri_args()
            token = args.api_key or args.key or args.token
            
            -- If not found in query params, try request body for POST requests
            if not token and ngx.var.request_method == "POST" then
                ngx.req.read_body()
                local body_data = ngx.req.get_body_data()
                if body_data then
                    local json = require "cjson"
                    local ok, body_table = pcall(json.decode, body_data)
                    if ok and body_table then
                        token = body_table.api_key or body_table.key or body_table.token
                        
                        -- Remove token from body and set modified body
                        if token then
                            body_table.api_key = nil
                            body_table.key = nil
                            body_table.token = nil
                            local new_body = json.encode(body_table)
                            ngx.req.set_body_data(new_body)
                        end
                    end
                end
            end
            
            if not token then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - No API key provided. Include api_key parameter in query string or request body"}')
                ngx.exit(401)
            end
            
            if not expected_token or token ~= expected_token then
                ngx.status = 401
                ngx.header.content_type = "application/json"
                ngx.say('{"error": "Unauthorized - Invalid API key"}')
                ngx.exit(401)
            end
            
            -- Remove the api_key from query string before proxying
            if ngx.req.get_uri_args().api_key or ngx.req.get_uri_args().key or ngx.req.get_uri_args().token then
                local args = ngx.req.get_uri_args()
                args.api_key = nil
                args.key = nil
                args.token = nil
                local new_args = {}
                for k, v in pairs(args) do
                    table.insert(new_args, k .. "=" .. tostring(v))
                end
                ngx.var.args = table.concat(new_args, "&")
            end
        }
        
        # Proxy to Ollama
        proxy_pass http://localhost:11434;
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
        
        # WebSocket support if needed
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Enable the site
RUN ln -s /etc/nginx/sites-available/ollama-proxy /etc/nginx/sites-enabled/

# Create nginx.conf to load lua module
RUN cat > /etc/nginx/nginx.conf << 'EOF'
env API_KEY;
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 768;
}

http {
    # Load Lua module and set package path
    lua_package_path "/usr/share/lua/5.1/?.lua;;";
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
    
    gzip on;
    
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOF

# Create startup script
RUN cat > /start.sh << 'EOF'
#!/bin/bash
set -e

# Start Ollama in background
echo "Starting Ollama server..."
/bin/ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do 
    sleep 1
done

# Pull the model if specified
if [ ! -z "$OLLAMA_MODEL" ]; then
    echo "Pulling model: $OLLAMA_MODEL"
    /bin/ollama pull $OLLAMA_MODEL
    
    # Test the model
    echo "Testing model..."
    /bin/ollama run $OLLAMA_MODEL 'Hello' || true
fi

# Start nginx
echo "Starting nginx..."
exec nginx -g 'daemon off;'

# Keep container running
echo "Setup complete. Server running on port 7860"
tail -f /dev/null
EOF

RUN chmod +x /start.sh

# Expose only the reverse proxy port
EXPOSE 7860

# Override the base image's ENTRYPOINT
ENTRYPOINT []

# Use the startup script
CMD ["/start.sh"]