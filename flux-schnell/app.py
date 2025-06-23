#!/usr/bin/env python3
# ultra_flux_api.py — max-performance FLUX backend (full-VRAM)

import os, uuid, asyncio, base64, time, logging, io, contextlib, psutil
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
from diffusers import DiffusionPipeline, logging as dlog
from huggingface_hub import login
from huggingface_hub.utils import logging as hflog
from PIL import Image

# ────────── logging ──────────
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
dlog.set_verbosity_error()
hflog.set_verbosity_error()

# ────────── env / config ──────────
HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL_ID   = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
OUT_DIR    = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORCE_CPU  = os.getenv("FORCE_CPU", "0") == "1"
MAX_STEPS  = int(os.getenv("MAX_STEPS", "100"))
CLEAN_AFTER_SEC = int(os.getenv("CLEAN_AFTER_SEC", "900"))

# ────────── torch device ──────────
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "8")))  # plenty of CPU threads

logging.info("device=%s | dtype=%s | model=%s", device, dtype, MODEL_ID)

# ────────── model load ──────────
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)  # no helper → silence warning

try:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,   # still helpful for first load
    )
except Exception as e:
    logging.critical("Model load failed: %s", e, exc_info=True)
    raise SystemExit(1)

# ⬇️  **FULL VRAM LOAD** ; no cpu_offload, no sharded weights
if device == "cuda":
    pipe.to("cuda")
else:
    pipe.to("cpu")

pipe.scheduler.set_timesteps(28)
inference_lock = asyncio.Lock()

# ────────── FastAPI setup ──────────
app = FastAPI(title="FLUX T2I", version="1.4.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PromptRequest(BaseModel):
    prompt: str
    steps: Optional[int] = 28
    return_base64: bool = True
    @classmethod
    def validate(cls, v):
        if isinstance(v, dict) and v.get("steps", 28) > MAX_STEPS:
            raise ValidationError(f"steps must be ≤ {MAX_STEPS}")
        return super().validate(v)

# ────────── helpers ──────────
async def _delete_later(path: Path, delay: int = CLEAN_AFTER_SEC):
    await asyncio.sleep(delay)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()

def _mem_usage() -> str:
    p = psutil.Process()
    return f"{p.memory_info().rss / 1024**2:.1f} MiB RSS | {psutil.virtual_memory().percent:.1f}% sys"

def _gpu_mem_usage():
    if not torch.cuda.is_available():
        return "n/a"
    reserved = torch.cuda.memory_reserved() / 1e9
    tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{reserved:.2f}/{tot:.2f} GiB"

# ────────── routes ──────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz")
async def healthz():
    return {"status":"ok","device":device,"ram":_mem_usage(),"vram":_gpu_mem_usage()}

@app.post("/generate", status_code=status.HTTP_201_CREATED)
async def generate(req: PromptRequest, bg: BackgroundTasks):
    t0  = time.perf_counter()
    uid = uuid.uuid4().hex
    img_path = OUT_DIR / f"{uid}.png"

    async with inference_lock:                       # single-GPU safety
        try:
            pipe.scheduler.set_timesteps(req.steps)
            image = pipe(req.prompt, num_inference_steps=req.steps).images[0]
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            logging.error("GPU OOM")
            raise HTTPException(503, "GPU OOM — reduce steps/resolution") from oom
        except Exception as e:
            logging.error("Inference error: %s", e, exc_info=True)
            raise HTTPException(500, str(e)) from e

    # Write PNG + base64 encode
    image.save(img_path)
    with io.BytesIO() as buf:
        image.save(buf, "PNG")
        png_bytes = buf.getvalue()

    bg.add_task(_delete_later, img_path)

    resp = {
        "id": uid,
        "steps": req.steps,
        "elapsed_sec": round(time.perf_counter()-t0, 3),
        "download_url": f"/download/{uid}",
    }
    if req.return_base64:
        resp["image_base64"] = base64.b64encode(png_bytes).decode()

    logging.info("served %s | %.2fs | RAM=%s | VRAM=%s | prompt=%r",
                 uid, resp["elapsed_sec"], _mem_usage(), _gpu_mem_usage(), req.prompt[:80])
    return resp

@app.get("/download/{image_id}")
async def download(image_id: str):
    img_path = OUT_DIR / f"{image_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(img_path, media_type="image/png", filename=f"{image_id}.png")

