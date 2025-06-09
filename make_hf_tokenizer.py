from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding
from transformers import GPT2TokenizerFast
import shutil

# todo: fix the vocab hole at idx 50256 thing

save_dir = "avey_tokenizer"

encoding = get_encoding("p50k_base")
convert_tiktoken_to_fast(encoding, save_dir)

tokenizer = GPT2TokenizerFast.from_pretrained(save_dir)
shutil.rmtree(save_dir)

tokenizer.save_pretrained(f"{save_dir}_base")
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>' + message['content'] }}{% elif message['role'] == 'system' %}{{ '<|system|>' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>'  + message['content'] }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"
tokenizer.additional_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
tokenizer.save_pretrained(f"{save_dir}_it")
