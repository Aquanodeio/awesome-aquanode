# Flux.1-Kontext-dev

## Usage
```sh
cd flu_kontext
docker build -t {image} .
docker run -p 7860:7860 {image}
```

**checkout /docs for swagger docs**

---


Image editing:
```py
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5
).images[0]
```

Flux Kontext comes with an integrity checker, which should be run after the image generation step. To run the safety checker, install the official repository from [black-forest-labs/flux](https://github.com/black-forest-labs/flux) and add the following code:

```python
import torch
import numpy as np
from flux.content_filters import PixtralContentFilter
integrity_checker = PixtralContentFilter(torch.device("cuda"))
image_ = np.array(image) / 255.0
image_ = 2 * image_ - 1
image_ = torch.from_numpy(image_).to("cuda", dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
if integrity_checker.test_image(image_):
raise ValueError("Your image has been flagged. Choose another prompt/image or try again.")
```

## Risks

Black Forest Labs is committed to the responsible development of generative AI technology. Prior to releasing FLUX.1 Kontext, we evaluated and mitigated a number of risks in our models and services, including the generation of unlawful content. We implemented a series of pre-release mitigations to help prevent misuse by third parties, with additional post-release mitigations to help address residual risks:
1. **Pre-training mitigation**. We filtered pre-training data for multiple categories of “not safe for work” (NSFW) content to help prevent a user generating unlawful content in response to text prompts or uploaded images.
2. **Post-training mitigation.** We have partnered with the Internet Watch Foundation, an independent nonprofit organization dedicated to preventing online abuse, to filter known child sexual abuse material (CSAM) from post-training data. Subsequently, we undertook multiple rounds of targeted fine-tuning to provide additional mitigation against potential abuse. By inhibiting certain behaviors and concepts in the trained model, these techniques can help to prevent a user generating synthetic CSAM or nonconsensual intimate imagery (NCII) from a text prompt, or transforming an uploaded image into synthetic CSAM or NCII.
3. **Pre-release evaluation.** Throughout this process, we conducted multiple internal and external third-party evaluations of model checkpoints to identify further opportunities for improvement. The third-party evaluations—which included 21 checkpoints of FLUX.1 Kontext [pro] and [dev]—focused on eliciting CSAM and NCII through adversarial testing with text-only prompts, as well as uploaded images with text prompts. Next, we conducted a final third-party evaluation of the proposed release checkpoints, focused on text-to-image and image-to-image CSAM and NCII generation. The final FLUX.1 Kontext [pro] (as offered through the FLUX API only) and FLUX.1 Kontext [dev] (released as an open-weight model) checkpoints demonstrated very high resilience against violative inputs, and FLUX.1 Kontext [dev] demonstrated higher resilience than other similar open-weight models across these risk categories.  Based on these findings, we approved the release of the FLUX.1 Kontext [pro] model via API, and the release of the FLUX.1 Kontext [dev] model as openly-available weights under a non-commercial license to support third-party research and development.
4. **Inference filters.** We are applying multiple filters to intercept text prompts, uploaded images, and output images on the FLUX API for FLUX.1 Kontext [pro]. Filters for CSAM and NCII are provided by Hive, a third-party provider, and cannot be adjusted or removed by developers. We provide filters for other categories of potentially harmful content, including gore, which can be adjusted by developers based on their specific risk profile. Additionally, the repository for the open FLUX.1 Kontext [dev] model includes filters for illegal or infringing content. Filters or manual review must be used with the model under the terms of the FLUX.1 [dev] Non-Commercial License. We may approach known deployers of the FLUX.1 Kontext [dev] model at random to verify that filters or manual review processes are in place.
5. **Content provenance.** The FLUX API applies cryptographically-signed metadata to output content to indicate that images were produced with our model. Our API implements the Coalition for Content Provenance and Authenticity (C2PA) standard for metadata.
6. **Policies.** Access to our API and use of our models are governed by our Developer Terms of Service, Usage Policy, and FLUX.1 [dev] Non-Commercial License, which prohibit the generation of unlawful content or the use of generated content for unlawful, defamatory, or abusive purposes. Developers and users must consent to these conditions to access the FLUX Kontext models.
7. **Monitoring.** We are monitoring for patterns of violative use after release, and may ban developers who we detect intentionally and repeatedly violate our policies via the FLUX API. Additionally, we provide a dedicated email address (safety@blackforestlabs.ai) to solicit feedback from the community. We maintain a reporting relationship with organizations such as the Internet Watch Foundation and the National Center for Missing and Exploited Children, and we welcome ongoing engagement with authorities, developers, and researchers to share intelligence about emerging risks and develop effective mitigations.


## License
This model falls under the [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

## Citation
```bib
@misc{labs2025flux1kontextflowmatching,
      title={FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space}, Add commentMore actions
      author={Black Forest Labs and Stephen Batifol and Andreas Blattmann and Frederic Boesel and Saksham Consul and Cyril Diagne and Tim Dockhorn and Jack English and Zion English and Patrick Esser and Sumith Kulal and Kyle Lacey and Yam Levi and Cheng Li and Dominik Lorenz and Jonas Müller and Dustin Podell and Robin Rombach and Harry Saini and Axel Sauer and Luke Smith},
      year={2025},
      eprint={2506.15742},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.15742},
}
```