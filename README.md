# Description
This repo contains the code for the custom dockerfiles used in aquanode.

Most probably all of them are going to be **Models** used for inference.

HF_TOKEN is primary requirement in these images while running. 

# Guidelines

0.16: means simple inference on bfloat16 or float16.
In case that model runs on f32[default precision on HF](which is rare in generative models, this 0.16 might mean that too. Although we will make sure f32 are in 0.32 tag).

And yeah I am gonna move this flux-schnell, but just not now.[we are using publicly available flux-schnell image drycoco/flux-schnell:0.0.3
