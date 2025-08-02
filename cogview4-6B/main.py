import os
import asyncio
from diffusers.pipelines.cogview4.pipeline_cogview4 import CogView4Pipeline
import torch
import time
from fastapi import FastAPI, HTTPException, Form, Response, Depends
from typing import Optional
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware
from utils import verify_api_key

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # Comma-separated list of allowed origins
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))  # Limit concurrent requests
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

# Add async lock to prevent concurrent GPU operations
# stops VRAM outage
generation_lock = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def initialize_pipeline():
    global pipe
    if pipe is None:
        print("Loading the CogView pipeline...")
        pipe = CogView4Pipeline.from_pretrained(
            "THUDM/CogView4-6B", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()  # Enable CPU offloading for memory efficiency
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("Pipeline loaded successfully!")


initialize_pipeline()

@app.get('/')
def read_root():
    return {"message": "Welcome to the CogView4 image generation API!"}

@app.post('/generate')
async def generate_image(
    prompt: str = Form(...),
    guidance_scale: Optional[float] = Form(3.5),
    seed: Optional[int] = Form(42),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024),
    num_inference_steps: Optional[int] = Form(28),
    api_key: bool = Depends(verify_api_key)
    ):
    try:
        # Use async lock to ensure only one generation happens at a time
        async with generation_lock:
            torch.cuda.empty_cache()  # Clear GPU memory
            loop = asyncio.get_running_loop()
            
            def generate():
                pipe.scheduler._step_index = None  # type: ignore # Reset step index
                
                torch.cuda.empty_cache()  # Clear GPU memory
                torch.manual_seed(seed)
                with torch.inference_mode():
                    result = pipe(  # type: ignore
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps or 28,
                        guidance_scale=guidance_scale or 3.5
                    )
                torch.cuda.empty_cache()  # Clear GPU memory
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
    try:
        if pipe is None:
            initialize_pipeline()
            
        return {"status": "healthy", "model": "CogView4-6B"}
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service is not healthy")