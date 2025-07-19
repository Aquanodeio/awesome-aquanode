# LLaMA 3.3

## Usage

```sh
cd llama3.3-70B
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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", torch_dtype=torch.bfloat16)

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## License

This model is released under the [License](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE).
