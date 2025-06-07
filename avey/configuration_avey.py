from transformers import PretrainedConfig

class AveyConfig(PretrainedConfig):
	model_type = "avey"

	def __init__(
		self,
		vocab_size: int = 50304,
		d_embed: int = 768,
		n_blocks: int = 26,
		expansion_factor: int = 4,
		chunk_size: int = 64,
		k: int = 8,
		context_proportion: float = 0.5,
		**kwargs
	):
		self.vocab_size = vocab_size
		self.d_embed = d_embed
		self.n_blocks = n_blocks
		self.expansion_factor = expansion_factor
		self.chunk_size = chunk_size
		self.k = k
		self.context_proportion = context_proportion

		self.extended_len = self.k * chunk_size
		super().__init__(**kwargs)
