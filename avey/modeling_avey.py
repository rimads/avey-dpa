import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from avey.configuration_avey import AveyConfig


class Contextualizer(nn.Module):
	def __init__(self, config: AveyConfig):
		super().__init__()
		self.spatial_proj = nn.Parameter(torch.empty(config.extended_len, config.extended_len))
		nn.init.xavier_normal_(self.spatial_proj)

	def cosim(self, embeddings: torch.Tensor) -> torch.Tensor:
		norm = torch.sqrt(torch.sum(embeddings ** 2, dim=-1, keepdim=True) + 1e-8)
		normalized = embeddings / norm
		cosim = torch.matmul(normalized, normalized.transpose(-1, -2))
		return cosim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x0, x1 = x.chunk(2, dim=-1)
		x0 = (torch.tril(self.spatial_proj) * self.cosim(x0)) @ x0
		output = x0 * x1
		return output


class NeuralContextualizerLayer(nn.Module):
	def __init__(self, config: AveyConfig):
		super().__init__()
		expanded_dim = config.d_embed * config.expansion_factor
		self.split_factor = [
			int(expanded_dim * config.context_proportion),
			int(expanded_dim * (1-config.context_proportion))
		]
		diff = expanded_dim - (self.split_factor[0] + self.split_factor[1])
		self.split_factor[1] += diff
		if self.split_factor[0] % 2 != 0:
			self.split_factor[0] += 1
			self.split_factor[1] -= 1

		self.enricher = nn.Linear(config.d_embed, expanded_dim)
		self.contextualizer = Contextualizer(config)
		proj_in_features = int(self.split_factor[0] / 2 + self.split_factor[1])
		self.fuser = nn.Linear(proj_in_features, config.d_embed)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_proj = F.relu(self.enricher(x))
		x_proj = x_proj * x_proj
		x0, x1 = x_proj.split(self.split_factor, dim=-1)
		x0 = self.contextualizer(x0)
		return self.fuser(torch.cat([x0, x1], dim=-1))


class AveyBlock(nn.Module):
	def __init__(self, config: AveyConfig):
		super().__init__()
		self.rms_norm = nn.RMSNorm(config.d_embed, eps=1e-10)
		self.ctxt = NeuralContextualizerLayer(config)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.ctxt(self.rms_norm(x))


class Avey(PreTrainedModel, GenerationMixin):
	config_class = AveyConfig

	def __init__(self, config):
		super().__init__(config)
		self.config = config

		self.wte = nn.Embedding(config.vocab_size, config.d_embed)
		nn.init.xavier_normal_(self.wte.weight)

		self.blocks = nn.ModuleList([AveyBlock(config) for _ in range(config.n_blocks)])
		self.ln_f = nn.RMSNorm(config.d_embed, eps=1e-10)

		self.generation_mode = False  # linear inference mode, disabled for training and by default
		self._tp_plan = {}  # something crashes in lm eval without this when running with accelerate

	def set_to_generation_mode(self):
		self.generation_mode = True

	def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
		x = self.wte(input_ids)
		B, T, E = x.shape

		# add padding to multiple of split size
		padded = False
		orig_T = T
		if T % self.config.chunk_size != 0:
			pad_length = self.config.chunk_size - (T % self.config.chunk_size)
			pad_tensor = torch.zeros(B, pad_length, E, device=x.device, dtype=x.dtype)
			x = torch.cat([x, pad_tensor], dim=1)
			T = x.shape[1]
			padded = True

		chunk_size = self.config.chunk_size
		k = self.config.k	# k is the number of chunks that is fed into the contextualizer (top-k + current chunk)
		target_length = self.config.extended_len
		N = T // chunk_size

		x_chunks = x.view(B, N, chunk_size, E)
		extended_chunks = []
		if self.generation_mode:
			# only contextualize the last chunk
			start = N - 1
			end = N
			N = 1
		else:
			start = 0
			end = N

		for i in range(start, end):
			cur_chunk = x_chunks[:, i]
			if i == 0:
				cat_chunks = cur_chunk
			else:
				prev_chunks = x_chunks[:, :i]
				cand_norm = prev_chunks / (prev_chunks.norm(dim=-1, keepdim=True) + 1e-8)
				cur_norm = cur_chunk / (cur_chunk.norm(dim=-1, keepdim=True) + 1e-8)
				cur_exp = cur_norm.unsqueeze(1)
				cand_trans = cand_norm.transpose(-1, -2)
				sims = torch.matmul(cur_exp, cand_trans)
				max_sims, _ = sims.max(dim=-1)
				sim_scores = max_sims.sum(dim=-1)
				num_to_select = min(i, k - 1)

				if num_to_select > 0:
					topk_scores, topk_indices = sim_scores.topk(num_to_select, dim=1)

					# Sort for temporal order
					sorted_topk_indices, sort_order = torch.sort(topk_indices, dim=1)
					sorted_topk_scores = torch.gather(topk_scores, 1, sort_order)

					max_score = sorted_topk_scores[:, :1]
					normalized_weights = sorted_topk_scores / (max_score + 1e-8)
					normalized_weights = normalized_weights.unsqueeze(-1).unsqueeze(-1)

					topk_exp = sorted_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, chunk_size, E)
					selected_candidates = torch.gather(prev_chunks, 1, topk_exp)

					weighted_candidates = selected_candidates * normalized_weights
					weighted_candidates = weighted_candidates.reshape(B, num_to_select * chunk_size, E)

					cat_chunks = torch.cat([weighted_candidates, cur_chunk], dim=1)
				else:
					cat_chunks = cur_chunk

			current_len = cat_chunks.shape[1]
			if current_len < target_length:
				pad_len = target_length - current_len
				pad_tensor = torch.zeros(B, pad_len, E, device=x.device, dtype=x.dtype)
				padded_cat_chunks = torch.cat([pad_tensor, cat_chunks], dim=1)
			else:
				padded_cat_chunks = cat_chunks
			extended_chunks.append(padded_cat_chunks)

		extended_chunks = torch.stack(extended_chunks, dim=1)
		h = extended_chunks.view(B * N, target_length, E)

		for block in self.blocks:
			h = block(h)

		h = h.view(B, N, target_length, E)
		final_chunks = h[:, :, -self.config.chunk_size:, :] # discard the appended chunks
		final_out = final_chunks.reshape(B, N * self.config.chunk_size, E)

		logits = F.linear(self.ln_f(final_out), self.wte.weight)

		# remove padding
		if padded:
			if self.generation_mode:
				orig_T = orig_T % self.config.chunk_size
			logits = logits[:, :orig_T, :]

		if labels is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
			return CausalLMOutput(logits=logits, loss=loss)

		return CausalLMOutput(logits=logits)
