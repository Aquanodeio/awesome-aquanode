# Qwen2.5-Coder

## Usage

```sh
cd qwen-2.5-coder-14B
docker build -t {image} .
docker run -p 7860:7860 {image}
```

---

```shell
pip install transformers torch
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype=torch.bfloat16)

prompt = "Write a Python function to calculate the factorial of a number."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## License

This model is released under the [Apache 2.0 License].
