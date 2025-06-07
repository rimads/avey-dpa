# based on https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_TOKENIZER = "p50k_base"

TRAIN_CONFIG = {
	"total_batch_size": 2 ** 19,  # ~0.5M tokens
	"seq_length": 2048,
	"max_steps": 2000,  # 2k steps = ~1B tokens @ 0.5M/step
	"inference_interval": 500,
	"checkpoint_interval": 1000,
	"rng_seed": 11,
	"use_torch_compile": False,

	"adam_beta_1": 0.9,
	"adam_beta_2": 0.95,
	"adam_fused": True,
	"adam_eps": 1e-12,
	"weight_decay": 0.1,
	"grad_norm_clip": 1.0,
}

LR_CONFIG = {
	"max_lr": 6e-4,
	"warmup_steps": 0,
	"min_lr": None,  # defaults to 10% of max_lr
}

PATH_CONFIG = {
	"project_name": "wandb-project",
	"model_name": "rwkv7",
	"backup_to_s3": False,
}
# -----------------------------------------------------------------------------


import random
import numpy as np
import torch
import boto3

import os
import math
import traceback
from typing import Tuple, List

import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from tqdm import tqdm
import wandb
import tiktoken  # TODO: switch out tiktoken for converted hf tokenizer?
import argparse

from rwkv7.src.model import RWKV, RWKVConfig

parser = argparse.ArgumentParser(description="training script")
parser.add_argument(
	"--device_bsz",
	type=int,
	default=1
)
parser.add_argument(
	"--pretrained",
	type=str,
	default=None
)
args, _ = parser.parse_known_args()
d_bsz = int(args.device_bsz)

TRAIN_CONFIG["micro_batch_size"] = d_bsz
LR_CONFIG["min_lr"] = LR_CONFIG["min_lr"] if LR_CONFIG["min_lr"] is not None else LR_CONFIG["max_lr"] / 10
PATH_CONFIG["save_dir"] = os.path.join("checkpoints", PATH_CONFIG["model_name"])

model_name = PATH_CONFIG["model_name"]

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
class DataLoader:
	def __init__(self, B: int, T: int, process_rank: int, num_processes: int):
		self.B = B
		self.T = T
		self.process_rank = process_rank
		self.num_processes = num_processes

		dataset_name = "HuggingFaceFW/fineweb"
		dataset_config = "sample-10BT" # models in the paper were trained using sample-100BT
		self.dataset = load_dataset(dataset_name, name=dataset_config, split="train", num_proc=os.cpu_count())
		self.tokenizer = tiktoken.get_encoding(MODEL_TOKENIZER)
		self.bos_token_id = self.tokenizer.eot_token

		self.dataset_length = len(self.dataset)
		self.reset()

	def reset(self) -> None:
		self.current_idx = self.process_rank
		self.current_buffer: List[int] = []

	def _get_next_tokens(self) -> None:
		required_tokens = self.B * self.T + 1
		while len(self.current_buffer) < required_tokens:
			if self.current_idx >= self.dataset_length:
				self.current_idx = self.process_rank  # wrap-around
			text = self.dataset[self.current_idx]["text"]
			tokens = self.tokenizer.encode(
				text,
				allowed_special={'<|endoftext|>'}, # in case the dataset contains some of these
				disallowed_special=(self.tokenizer.special_tokens_set - {'<|endoftext|>'})
			) + [self.bos_token_id]
			self.current_buffer.extend(tokens)
			self.current_idx += self.num_processes

	def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
		self._get_next_tokens()
		total_tokens = self.B * self.T + 1
		buf = torch.tensor(self.current_buffer[:total_tokens])
		self.current_buffer = self.current_buffer[self.B * self.T:]
		x = buf[:-1].view(self.B, self.T)
		y = buf[1:].view(self.B, self.T)
		return x, y

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def backup_to_s3(local_directory, bucket_name="backup"):
	s3_client = boto3.client('s3')

	for filename in os.listdir(local_directory):
		local_path = os.path.join(local_directory, filename)
		if os.path.isfile(local_path):
			print(f"Uploading {local_path} to S3 bucket {bucket_name}")
			s3_client.upload_file(local_path, bucket_name, f"{local_path}")

def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float) -> torch.optim.Optimizer:
	param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
	decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
	nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

	optim_groups = [
		{'params': decay_params, 'weight_decay': weight_decay},
		{'params': nodecay_params, 'weight_decay': 0.0}
	]

	num_decay_params = sum(p.numel() for p in decay_params)
	num_nodecay_params = sum(p.numel() for p in nodecay_params)
	if master_process:
		print(f"Number of decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
		print(f"Number of non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")

	optimizer = torch.optim.AdamW(
		optim_groups,
		lr=learning_rate,
		betas=(TRAIN_CONFIG["adam_beta_1"], TRAIN_CONFIG["adam_beta_2"]),
		eps=TRAIN_CONFIG["adam_eps"],
		fused=TRAIN_CONFIG["adam_fused"]
	)
	return optimizer

def get_lr(it: int) -> float:
	if it < LR_CONFIG["warmup_steps"]:
		return LR_CONFIG["max_lr"] * (it + 1) / LR_CONFIG["warmup_steps"]

	if it > TRAIN_CONFIG["max_steps"]:
		return LR_CONFIG["min_lr"]

	decay_ratio = (it - LR_CONFIG["warmup_steps"]) / (TRAIN_CONFIG["max_steps"] - LR_CONFIG["warmup_steps"])
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return LR_CONFIG["min_lr"] + coeff * (LR_CONFIG["max_lr"] - LR_CONFIG["min_lr"])

# -----------------------------------------------------------------------------
# Training Main Routine
# -----------------------------------------------------------------------------
def main():
	global master_process  # For logging and distributed setup
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	from torch._inductor import config
	config.fallback_random = True
	seed = TRAIN_CONFIG["rng_seed"]
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.random.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)

	# -----------------------------------------------------------------------------
	# DDP Setup
	# -----------------------------------------------------------------------------
	ddp = int(os.environ.get('RANK', -1)) != -1
	if ddp:
		assert torch.cuda.is_available(), "CUDA is required for DDP training."
		init_process_group(backend='nccl')
		ddp_rank = int(os.environ['RANK'])
		ddp_local_rank = int(os.environ['LOCAL_RANK'])
		ddp_world_size = int(os.environ['WORLD_SIZE'])
		device = f'cuda:{ddp_local_rank}'
		torch.cuda.set_device(device)
		master_process = (ddp_rank == 0)
	else:
		ddp_rank = 0
		ddp_local_rank = 0
		ddp_world_size = 1
		master_process = True
		device = "cpu"
		if torch.cuda.is_available():
			device = "cuda"
		elif torch.backends.mps.is_available():
			device = "mps"
		print(f"Using device: {device}")

	device_type = "cuda" if device.startswith("cuda") else "cpu"

	# -----------------------------------------------------------------------------
	# Initialize WandB and checkpoint directories
	# -----------------------------------------------------------------------------
	os.makedirs(PATH_CONFIG["save_dir"], exist_ok=True)
	log_file = os.path.join(PATH_CONFIG["save_dir"], "log.txt")

	if master_process:
		wandb.init(project=PATH_CONFIG["project_name"], name=PATH_CONFIG["model_name"])

	# -----------------------------------------------------------------------------
	# Data Loader Setup & Gradient Accumulation Computation
	# -----------------------------------------------------------------------------
	B = TRAIN_CONFIG["micro_batch_size"]
	T = TRAIN_CONFIG["seq_length"]
	total_batch_size = TRAIN_CONFIG["total_batch_size"]
	# Ensure total_batch_size is divisible across processes and gradient accumulation steps.
	assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
	grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
	TRAIN_CONFIG["grad_accum_steps"] = grad_accum_steps

	train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

	# tf32
	if device_type == "cuda":
		torch.set_float32_matmul_precision('high')

	# -----------------------------------------------------------------------------
	# Tokenizer & Model Initialization
	# -----------------------------------------------------------------------------
	inference_tokenizer = tiktoken.get_encoding(MODEL_TOKENIZER)

	if args.pretrained is not None:
		model = RWKV.from_pretrained(args.pretrained)
	else:
		config = RWKVConfig(
			vocab_size=inference_tokenizer.n_vocab,
			n_embd=768,
			n_layer=12,
			ctx_len=TRAIN_CONFIG["seq_length"]
		)
		model = RWKV(config)

	if ddp_rank == 0:
		print("PARAMERETS:", sum(p.numel() for p in model.parameters()))
	model.to(device)

	use_torch_compile = TRAIN_CONFIG["use_torch_compile"]
	if use_torch_compile:
		model = torch.compile(model)
	if ddp:
		model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
	raw_model = model.module if ddp else model

	optimizer = configure_optimizers(raw_model, weight_decay=0.1, learning_rate=LR_CONFIG["max_lr"])

	# Inference prompt for generating sample completions.
	prompt_str = "Turing machines are"
	prog_bar = tqdm(
		range(0, TRAIN_CONFIG["max_steps"]),
		desc="Step", initial=0, total=TRAIN_CONFIG["max_steps"],
		disable=(not master_process)
	)

	# -----------------------------------------------------------------------------
	# Training Loop
	# -----------------------------------------------------------------------------
	for step in prog_bar:
		model.train()
		optimizer.zero_grad()
		loss_accum = 0.0

		for micro_step in range(grad_accum_steps):
			x, y = train_loader.next_batch()
			x, y = x.to(device), y.to(device)
			if ddp:
				model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

			with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
				output = model(x, y)
				loss = output.loss
			loss = loss / grad_accum_steps
			loss_accum += loss.detach()
			loss.backward()

		if ddp:
			dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_norm_clip"])
		lr = get_lr(step)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		optimizer.step()

		if master_process:
			wandb.log({
				"train/grad_norm": grad_norm,
				"train/learning_rate": lr,
				"train/loss": loss_accum.item(),
				"train/step": step,
			}, step=step)
		prog_bar.set_postfix({"loss": loss_accum.item()})

		# --- GENERATE SAMPLE COMPLETIONS ---
		if master_process and (step % TRAIN_CONFIG["inference_interval"] == 0) and (step > 0):
			print(f"\n=== Generation Samples at Step {step} ===")
			generation_logs = [
				f"Step: {step}",
				f"Loss: {loss_accum.item():.6f}",
				"Generations:"
			]
			current_input_ids = torch.tensor([inference_tokenizer.encode(prompt_str)], device=device)
			for i in range(5):  # generate 5 completions
				gen_ids = raw_model.generate(current_input_ids, max_new_tokens=30, do_sample=True, temperature=0.7)
				gen_text = inference_tokenizer.decode(gen_ids[0].tolist())
				generation_logs.append(f"  Completion {i+1}: {gen_text}")
				print(f"Completion {i+1}: {gen_text}")
			generation_logs.append("=" * 40 + "\n")
			with open(log_file, "a") as f:
				f.write("\n".join(generation_logs) + "\n")
			print("==========================================\n")

		if master_process and ((step+1) % TRAIN_CONFIG["checkpoint_interval"] == 0) and (step > 0):
			# I added this while loop here cuz i had a training run crash once when the thing ran out of storage. absolutely not fun at all
			while True:
				try:
					ckpt_dir = os.path.join(PATH_CONFIG["save_dir"], f"checkpoint_{step}")
					raw_model.save_pretrained(ckpt_dir)

					if PATH_CONFIG["backup_to_s3"]:
						backup_to_s3(ckpt_dir)

					break
				except Exception as e:
					print(f"\nException during checkpointing: {e}")
					traceback.print_exc()
					input("Checkpointing failed. Please fix the issue and press Enter to retry...")

	if ddp:
		destroy_process_group()


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print("-" * 30)
		print(e)
		print("-" * 30)
		traceback.print_exc()
	finally:
		if 'dist' in globals() and dist.is_initialized():
			destroy_process_group()
