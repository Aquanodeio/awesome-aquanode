import asyncio
import torch
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
from PIL import Image
import io
from utils import verify_api_key

port = int(os.getenv("PORT", "8000"))

app = FastAPI()

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

def initialize_pipeline():
    global pipe
    if pipe is None:
        print("Loading the FLUX.1-Kontext pipeline...")
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", 
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        print("Pipeline loaded successfully!")

initialize_pipeline()
generation_lock = asyncio.Lock()

@app.get('/')
def read_root():
    return {"message": "Welcome to the FLUX.1-Kontext image generation API!"}

@app.post('/generate')
async def generate_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    guidance_scale: Optional[float] = Form(3.5),
    seed: Optional[int] = Form(42),
    api_key: bool = Depends(verify_api_key),
):
    try:
        async with generation_lock:
            
            loop = asyncio.get_event_loop()

            def generate():
                pipe.scheduler._step_index = None  # type: ignore # Reset step index
                
                image_data = image.file.read()
                input_image = Image.open(io.BytesIO(image_data))
                
                # Convert to RGB if needed
                if input_image.mode != 'RGB':
                    input_image = input_image.convert('RGB')

                torch.manual_seed(seed)  # Set seed for reproducibility

                with torch.inference_mode():
                    result = pipe(  # type: ignore
                        image=input_image,
                        prompt=prompt,
                        guidance_scale=guidance_scale # type: ignore
                    )

                torch.cuda.empty_cache()  # Clear GPU memory
                return result.images[0]  # type: ignore
            
            # Offload GPU work to thread (non-blocking)
            t1 = time.time()
            generated_image = await loop.run_in_executor(None, generate)
            torch.cuda.empty_cache()
            t2 = time.time()
            
            print(f"Image generation took {t2 - t1:.2f} seconds")
            
            # Convert to buffer
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
        return {"status": "healthy", "model": "FLUX.1-Kontext"}
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")