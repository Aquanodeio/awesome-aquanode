from fastapi import Header, HTTPException
import os

API_KEY = os.getenv("API_KEY", "api_key")

def verify_api_key(authorization: str = Header(...)):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        api_key = authorization.split(" ")[1]
        if api_key != API_KEY:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
        
        return {"verify": "true"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")