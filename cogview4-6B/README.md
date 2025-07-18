# CogView4-6B

## Inference Requirements and Model Introduction

+ Resolution: Width and height must be between `512px` and `2048px`, divisible by `32`, and ensure the maximum number of
  pixels does not exceed `2^21` px.
+ Precision: BF16 / FP32 (FP16 is not supported as it will cause overflow resulting in completely black images)

## Usage

```sh
cd cogview4
docker build -t {image} .
docker run -p 7860:7860 {image}
```
**checkout /docs for swagger docs**

---
```shell
pip install diffusers
```

```python
from diffusers import CogView4Pipeline

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]

image.save("cogview4.png")
```

## License

This model is released under the [Apache 2.0 License](LICENSE).