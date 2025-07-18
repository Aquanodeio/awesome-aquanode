# Gemma

## Usage

```sh
cd gemma
docker build -t {image} .
docker run -p 11434:11434 {image}
```

**checkout /docs for swagger docs**

---

```shell
pip install transformers torch
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.bfloat16)

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## License

This model is released under the [Gemma License](https://ai.google.dev/gemma/terms).
