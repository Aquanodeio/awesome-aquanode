import asyncio
import torch
import time
from fastapi import FastAPI, HTTPException, Form, Response, Header
from typing import Optional
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

from utils import verify_api_key

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # Comma-separated list of allowed origins

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allow all origins for simplicity, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize the pipeline globally
pipe = None
generation_lock = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")))  # Limit concurrent requests

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
def initialize_pipeline():
    global pipe
    if pipe is None:
        print("Loading the FLUX.1-dev pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        print("Pipeline loaded successfully!")


initialize_pipeline()

@app.get('/')
def read_root():
    return {"message": "Welcome to the FLUX.1-dev image generation API!"}

@app.post('/generate')
async def generate_image(
    prompt: str = Form(...),
    guidance_scale: Optional[float] = Form(3.5),
    seed: Optional[int] = Form(42),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024),
    num_inference_steps: Optional[int] = Form(28),
    authorization: str = Header(..., description="Bearer token for API key verification")
    ):
    
    try:

        verify_api_key(authorization)

        async with generation_lock:  # Ensure only one generation at a time
            torch.cuda.empty_cache()
            loop = asyncio.get_running_loop()
            
            def generate():
                
                pipe.scheduler._step_index = None  # type: ignore # Reset step index

                torch.cuda.empty_cache()
                torch.manual_seed(seed)

                with torch.inference_mode():
                    result = pipe(  # type: ignore
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps or 28,
                        guidance_scale=guidance_scale or 3.5
                    )
                    return result.images[0]  # type: ignore
            
            # Offload GPU work to thread (non-blocking)
            t1 = time.time()
            generated_image = await loop.run_in_executor(None, generate)
            t2 = time.time()
            
            print(f"Image generation took {t2 - t1:.2f} seconds")
            
            img_buffer = io.BytesIO()
            generated_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
        return Response(
                content=img_buffer.getvalue(),
                media_type="image/png"
            )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health_check():
    return {"status": "healthy", "model": "FLUX.1-dev"}
