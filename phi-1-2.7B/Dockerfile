# Use the base Ollama image
FROM ollama/ollama:latest

# Install curl, nginx, and lua module
RUN apt-get update && apt-get install -y curl nginx libnginx-mod-http-lua

# Set the environment variable for the model
ENV OLLAMA_MODEL=phi:2.7b

# Create nginx configuration
RUN rm -f /etc/nginx/sites-enabled/default

# Create the nginx site configuration
RUN cat > /etc/nginx/sites-available/ollama-proxy << 'EOF'
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
RUN cat > /etc/nginx/nginx.conf << EOF
env AUTH_TOKEN;
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

# Expose only the reverse proxy port
EXPOSE 7860

# Override the base image's ENTRYPOINT
ENTRYPOINT []

# Start script that runs both Ollama and nginx
CMD ["/bin/sh", "-c", "/bin/ollama serve & while ! curl -s http://localhost:11434/api/tags > /dev/null; do sleep 1; done && /bin/ollama pull $OLLAMA_MODEL && /bin/ollama run $OLLAMA_MODEL 'Hello' && nginx -g 'daemon off;' & tail -f /dev/null"]