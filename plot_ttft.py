# this script was mostly vibe coded (plus some edits of course)

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from rwkv7.src.model import RWKV
from tpp.modeling_tpp import TPPForCausalLM
from avey.modeling_avey import Avey
import csv

#
# ─── USER CONFIG ───────────────────────────────────────────────────────────────
#
TOKENIZER_DIR = "avey-ai/avey1-tokenizer-base"

# Four models to compare: (display_name, ModelClass, model_directory)
# Replace PlaceholderModelClassX & "path/to/modelX" with your real classes & dirs.
MODEL_SPECS = [
	("Mamba", MambaLMHeadModel, "avey-ai/mamba-dpa-1.5B-100BT"),
	("RWKV-7", RWKV, "avey-ai/rwkv7-dpa-1.5B-100BT"),
	("T++", TPPForCausalLM, "avey-ai/tpp-dpa-1.5B-100BT"),
	("Avey", Avey, "avey-ai/avey1-dpa-1.5B-100BT"),
]

# The context lengths (in tokens) at which you want to measure generation time
DATA_POINTS = [8000, 16000, 24000, 32000, 40000, 64000, 80000, 128000]

# Warmup settings (to discard the first few calls)
WARMUP_CONTEXT_TOKENS = 10    # a small dummy context
NUM_WARMUP_RUNS       = 10    # how many times to run warmup per model

# Sampling settings for generation
DO_SAMPLE   = True
TEMPERATURE = 0.9
SEED        = 42

# Whether to plot the X axis on a log scale
USE_LOG_X = False

# Output files
OUTPUT_CSV  = "long_context_compare.csv"
OUTPUT_PLOT = "long_context_compare.png"
#
# ────────────────────────────────────────────────────────────────────────────────
#

def measure_single_point(model, context_len, pad_token_id):
	"""
	Build a dummy context of length `context_len` and time
	how long it takes to generate one more token.
	"""
	# create [1, context_len] filled with pad_token_id
	input_ids = torch.full(
		(1, context_len),
		pad_token_id,
		dtype=torch.long,
		device="cuda"
	)

	torch.cuda.synchronize()
	t0 = time.time()

	_ = model.generate(
		input_ids,
		do_sample=DO_SAMPLE,
		temperature=TEMPERATURE,
		max_new_tokens=1,
	)

	torch.cuda.synchronize()
	t1 = time.time()
	return t1 - t0

def main():
	# fix randomness so that sampling is consistent
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

	# load shared tokenizer
	tokenizer = AutoTokenizer.from_pretrained(
		TOKENIZER_DIR, trust_remote_code=True
	)

	# pick a pad token id (fallback to eos or 0)
	pad_id = (
		tokenizer.pad_token_id
		if tokenizer.pad_token_id is not None
		else tokenizer.eos_token_id
		if tokenizer.eos_token_id is not None
		else 0
	)

	model_names = [name for name, _, _ in MODEL_SPECS]
	# will hold per‐model lists of timings
	model_times = {name: [] for name in model_names}

	for name, ModelClass, model_dir in MODEL_SPECS:
		print(f"\n=== Loading & measuring {name} ===")
		model = ModelClass.from_pretrained(
			model_dir, trust_remote_code=True
		).to("cuda")
		model.generation_mode = True  # for avey

		# ── Warmup ────────────────────────────────────────────────────────────
		print(
			f"  • Warmup: {NUM_WARMUP_RUNS} runs "
			f"(context={WARMUP_CONTEXT_TOKENS} tokens)… ",
			end="", flush=True
		)
		for _ in range(NUM_WARMUP_RUNS):
			_ = measure_single_point(model, WARMUP_CONTEXT_TOKENS, pad_id)
		print("done.")

		# ── Real measurements ─────────────────────────────────────────────────
		for dp in DATA_POINTS:
			print(f"  • context={dp:>5} tokens … ", end="", flush=True)
			t = measure_single_point(model, dp, pad_id)
			print(f"{t:.4f}s")
			model_times[name].append(t)

		# free GPU before next model
		del model
		torch.cuda.empty_cache()

	# ── Save CSV ────────────────────────────────────────────────────────────────
	with open(OUTPUT_CSV, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["context_length"] + model_names)
		for idx, dp in enumerate(DATA_POINTS):
			row = [dp] + [model_times[n][idx] for n in model_names]
			writer.writerow(row)
	print(f"\nSaved timings to {OUTPUT_CSV}")

	# ── Plot ───────────────────────────────────────────────────────────────────
	plt.figure(figsize=(8, 5))
	for name in model_names:
		plt.plot(
			DATA_POINTS,
			model_times[name],
			marker="o",
			label=name,
			linewidth=1.5
		)
	plt.xlabel("Context length (tokens)")
	plt.ylabel("TTFT (s)")
	plt.title("TTFT vs. Context Length")
	if USE_LOG_X:
		plt.xscale("log")
		plt.gca().xaxis.set_major_formatter(
			plt.FuncFormatter(lambda v, pos: f"{int(v):,}")
		)
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.tight_layout()
	plt.savefig(OUTPUT_PLOT, dpi=150)
	print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
	main()
