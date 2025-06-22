import os
import uuid
import base64
import asyncio
from pathlib import Path
from typing import Optional

import torch
from diffusers import FluxPipeline
from huggingface_hub import login
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from brotli_asgi import BrotliMiddleware
from transformers.utils import logging as hf_logging

# ───── Config ─────
hf_logging.set_verbosity_error()

HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL_ID   = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
OUT_DIR    = Path(os.getenv("OUTPUT_DIR", "/dev/shm/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

device     = "cuda" if torch.cuda.is_available() else "cpu"
dtype      = torch.bfloat16 if device == "cuda" else torch.float32

BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 6))
BATCH_WAIT  = float(os.getenv("BATCH_WAIT", 0.05))
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT", 30))
MAX_QUEUE   = int(os.getenv("MAX_QUEUE", 256))

# ───── Model ─────
try:
    if HF_TOKEN:
        login(token=HF_TOKEN)

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID, token=HF_TOKEN, torch_dtype=dtype, use_safetensors=True
    ).to(device)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()

    # lower timesteps for speed
    pipe.scheduler.set_timesteps(28, device=device)

    if hasattr(torch, "compile") and hasattr(pipe, "unet"):
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

except Exception as e:
    raise RuntimeError(f"🚨 Failed to load FLUX model '{MODEL_ID}': {e}")

# ───── API ─────
app = FastAPI(title="FLUX Text-to-Image API", version="2.1.0")
app.add_middleware(BrotliMiddleware, quality=4)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PromptRequest(BaseModel):
    prompt: str
    return_base64: Optional[bool] = False

class Job:
    __slots__ = ("prompt", "b64", "future")
    def __init__(self, prompt: str, b64: bool):
        self.prompt, self.b64 = prompt, b64
        self.future = asyncio.get_event_loop().create_future()

_queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=MAX_QUEUE)

# ───── Batch worker ─────
async def batch_worker():
    while True:
        try:
            job = await asyncio.wait_for(_queue.get(), timeout=BATCH_WAIT)
        except asyncio.TimeoutError:
            continue

        batch = [job]
        for _ in range(BATCH_SIZE - 1):
            try:
                batch.append(_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        try:
            with torch.inference_mode():
                prompts = [j.prompt for j in batch]
                results = pipe(
                    prompts,
                    num_inference_steps=28,
                    output_type="pil",
                )
                images = results.images
        except Exception as e:
            for j in batch:
                if not j.future.done():
                    j.future.set_exception(RuntimeError(f"inference failure: {e}"))
        else:
            for img, j in zip(images, batch):
                try:
                    uid = uuid.uuid4().hex
                    fp  = OUT_DIR / f"{uid}.png"
                    img.save(fp)
                    payload = {"id": uid, "download_url": f"/download/{uid}"}
                    if j.b64:
                        payload["image_base64"] = base64.b64encode(fp.read_bytes()).decode()
                    if not j.future.done():
                        j.future.set_result(payload)
                except Exception as e:
                    if not j.future.done():
                        j.future.set_exception(RuntimeError(f"post-proc failure: {e}"))
        finally:
            for _ in batch:
                _queue.task_done()

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(batch_worker())
    try:
        with torch.inference_mode():
            pipe("warmup", num_inference_steps=5, output_type="pil")
    except:
        pass

@app.get("/", response_class=HTMLResponse)
def index(request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/generate")
async def generate(req: PromptRequest):
    job = Job(req.prompt, req.return_base64)
    try:
        _queue.put_nowait(job)
    except asyncio.QueueFull:
        raise HTTPException(429, "queue full — retry later")
    try:
        return await asyncio.wait_for(job.future, timeout=REQ_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(503, "generation timed out")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/download/{image_id}")
def download(image_id: str):
    fp = OUT_DIR / f"{image_id}.png"
    if not fp.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(fp, media_type="image/png", filename=f"{image_id}.png",
                        headers={"Cache-Control": "public, max-age=3600"})

