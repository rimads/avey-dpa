from transformers import PretrainedConfig

class TPPConfig(PretrainedConfig):
    """
    Configuration for the Avey language model.
    """

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50257,
        n_layer: int = 12,
        n_head: int = 12,
        n_embed: int = 768,
        rope_theta=10000.0,
        **kwargs
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embed = n_embed
        self.rope_theta = float(rope_theta)
        super().__init__(**kwargs)
