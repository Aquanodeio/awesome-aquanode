# Phi-3.5

## Usage

```sh
cd phi3.5
docker build -t {image} .
docker run -p 11434:11434 {image}
```


---

```shell
pip install transformers torch
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16)

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## License

This model is released under the [MIT License].
