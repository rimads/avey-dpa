# based on https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from avey.modeling_avey import Avey

torch.set_float32_matmul_precision('high')	# tf32?

@register_model("avey")
class AveyEvalWrapper(HFLM):

	AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

	def __init__(self, pretrained, max_length=136000, batch_size=None, device="cuda"):
		LM.__init__(self)
		self._model = Avey.from_pretrained(pretrained, trust_remote_code=True).to(device)
		self.tokenizer = AutoTokenizer.from_pretrained("avey-ai/avey1-tokenizer-base", trust_remote_code=True)
		self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
		self.vocab_size = self.tokenizer.vocab_size
		self._batch_size = int(batch_size) if batch_size is not None else 64
		self._max_length = max_length
		self._device = torch.device(device)
		self.backend = "causal"
		self.add_bos_token = True
		self.logits_cache = False
		self.revision = None
		self.pretrained = pretrained
		self.peft = None
		self.delta = None
		self.truncation = False
		self.softmax_dtype = None

	@property
	def batch_size(self):
		return self._batch_size

	def _model_generate(self, context, max_length, stop=None, **generation_kwargs):
		output_ids = self._model.generate(
			context,
			max_length=max_length,
			**generation_kwargs
		)
		return output_ids

if __name__ == "__main__":
	cli_evaluate()
