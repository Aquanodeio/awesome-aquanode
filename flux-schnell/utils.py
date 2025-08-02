from fastapi import Query, HTTPException
import os

API_KEY = os.getenv("API_KEY", "auth_token")

def verify_api_key(api_key: str = Query(..., description="API key for authentication")):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    return True