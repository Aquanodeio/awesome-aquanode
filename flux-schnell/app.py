import os, uuid, asyncio, base64
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from huggingface_hub import login

# ────────────────────────── config ──────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL_ID   = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
OUT_DIR    = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

# ────────────────────────── model ───────────────────────────
if HF_TOKEN:
    login(token=HF_TOKEN)

pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
pipe.enable_model_cpu_offload()          # handles .to(device) internally
pipe.scheduler.set_timesteps(28)         # avoid step-index bug

# ────────────────────────── fastapi ─────────────────────────
app = FastAPI(title="FLUX Text-to-Image API",
              description="Generate images via FLUX.1-schnell",
              version="1.0.1")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PromptRequest(BaseModel):
    prompt: str
    steps: Optional[int] = 28          # let caller override
    return_base64: Optional[bool] = False

async def _delete_after(path: Path, delay: int = 600) -> None:
    await asyncio.sleep(delay)
    try: path.unlink()
    except FileNotFoundError: pass

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz", tags=["system"])
async def healthz():
    return {"status": "ok"}

@app.post("/generate", tags=["inference"])
async def generate(req: PromptRequest, bg: BackgroundTasks):
    uid = uuid.uuid4().hex
    img_path = OUT_DIR / f"{uid}.png"

    try:
        pipe.scheduler.set_timesteps(req.steps)
        image = pipe(req.prompt, num_inference_steps=req.steps).images[0]
        image.save(img_path)

        bg.add_task(_delete_after, img_path)

        resp = {"id": uid, "download_url": f"/download/{uid}"}
        if req.return_base64:
            resp["image_base64"] = base64.b64encode(img_path.read_bytes()).decode()
        return resp

    except Exception as e:
        if img_path.exists():
            img_path.unlink()
        raise HTTPException(500, f"Image generation failed: {e}")

@app.get("/download/{image_id}", tags=["inference"])
async def download(image_id: str):
    img_path = OUT_DIR / f"{image_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(img_path, media_type="image/png",
                        filename=f"{image_id}.png",
                        headers={"Cache-Control": "public, max-age=3600"})

