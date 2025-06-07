from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from torchtune.modules import RotaryPositionalEmbeddings

from tpp.configuration_tpp import TPPConfig

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embed % config.n_head == 0
		if RotaryPositionalEmbeddings is None:
			raise ImportError("RotaryPositionalEmbeddings could not be imported from torchtune.")

		self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
		self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
		self.c_proj.NANOGPT_SCALE_INIT = 1
		self.n_head = config.n_head
		self.n_embed = config.n_embed
		self.head_dim = self.n_embed // self.n_head

		# --- Instantiate Torchtune RoPE ---
		self.rope = RotaryPositionalEmbeddings(
			dim=self.head_dim,
			max_seq_len=136000,
			base=int(config.rope_theta)
		)

	def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)

		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.n_embed, dim=2)

		# Reshape for multi-head attention: (B, nh, T, hs)
		k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
		q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
		v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

		q_for_rope = q.transpose(1, 2)
		k_for_rope = k.transpose(1, 2)

		q_rotated = self.rope(q_for_rope, input_pos=position_ids)
		k_rotated = self.rope(k_for_rope, input_pos=position_ids)

		# Transpose back to [b, n_h, s, h_d] for attention calculation
		q = q_rotated.transpose(1, 2)
		k = k_rotated.transpose(1, 2)

		y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(B, T, C)

		y = self.c_proj(y)
		return y

class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.hidden_size = config.n_embed
		self.intermediate_size = self.hidden_size * 4
		self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
		self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
		self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
		self.act_fn = nn.SiLU()
		self.config.pretraining_tp = 1

	def forward(self, x):
		slice = self.intermediate_size
		gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
		up_proj_slices = self.up_proj.weight.split(slice, dim=0)
		down_proj_slices = self.down_proj.weight.split(slice, dim=1)
		gate_proj = torch.cat(
			[F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
		)
		up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
		intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
		down_proj = [
			F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
		]
		down_proj = sum(down_proj)
		return down_proj

class Block(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.RMSNorm(config.n_embed)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.RMSNorm(config.n_embed)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class TPPForCausalLM(PreTrainedModel, GenerationMixin):
	config_class = TPPConfig

	def __init__(self, config):
		super().__init__(config)
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embed),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = nn.RMSNorm(config.n_embed),
		))

		# init params
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			std = 0.02
			if hasattr(module, 'NANOGPT_SCALE_INIT'):
				std *= (2 * self.config.n_layer) ** -0.5
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, input_ids, labels=None, **kwargs):
		# idx is of shape (B, T)
		B, T = input_ids.size()
		# forward the token and posisition embeddings
		tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (B, T, n_embed)
		x = tok_emb
		# forward the blocks of the transformer
		for block in self.transformer.h:
			x = block(x)
		# forward the final RMSNorm and the classifier
		x = self.transformer.ln_f(x)
		logits = F.linear(x, self.transformer.wte.weight)

		if labels is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
			return CausalLMOutput(logits=logits, loss=loss)
		return CausalLMOutput(logits=logits)
