# MIT License
#
# Copyright (c) 2025 Haowen Hou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# modified by Devang Acharya, May 2025
# from https://github.com/howard-hou/RWKV-X/blob/main/evaluation/pass_key_rwkv.py
#

import torch
import argparse
from numpy import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import seaborn as sns
import tiktoken
from tpp.modeling_tpp import TPPForCausalLM

def parse_config():
	parser = argparse.ArgumentParser(description='arg parser')
	parser.add_argument('--model_path', type=str, default="avey-ai/tpp-dpa-1.5B-100BT")
	parser.add_argument('--cache_dir', type=str, default="./cache")
	parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
	parser.add_argument('--log_name', type=str, default='plots')
	parser.add_argument('--device', type=str, default='cuda')

	args = parser.parse_args()
	return args


def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix):
	"""Generates a text file and inserts an passkey at a random position."""
	rnd_state = random.get_state()
	random.seed(seed)
	n_garbage_suffix = n_garbage - n_garbage_prefix

	task_description = ""  # would end up causing more issues with non instruct tuned or undertrained models
	garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
	garbage_inf = " ".join([garbage] * 10000)
	assert len(garbage_inf) >= n_garbage
	garbage_prefix = garbage_inf[:n_garbage_prefix]
	garbage_suffix = garbage_inf[:n_garbage_suffix]
	pass_key = random.randint(1, 50000)
	information_line = f"The pass key is {pass_key}"
	final_question = "The pass key is"
	lines = [
		task_description,
		garbage_prefix,
		information_line,
		garbage_suffix,
		final_question,
	]
	random.set_state(rnd_state)
	return "\n".join(lines), str(pass_key)

@torch.inference_mode()
def passkey_retrieval_test(model, encoding, device,
						   n_garbage_prefix, n_garbage=60000, seed=666):
	prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix)
	input_ids = encoding.encode(prompt)
	answer_ids = encoding.encode(answer)
	len_tokens = len(input_ids)

	input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

	output_ids = model.generate(
		input_tensor,
		max_new_tokens=len(answer_ids),
		do_sample=False, # greedy
		pad_token_id=encoding.eot_token or encoding.encode("\n")[-1]
	)[0].tolist()

	gen_ids = output_ids[-len(answer_ids):]

	model_answer = encoding.decode(gen_ids).strip()
	gold_answer  = encoding.decode(answer_ids).strip()
	return (model_answer == gold_answer), len_tokens


def plot_heatmap(df, args):
	cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

	pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
	pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
	# Create the heatmap with better aesthetics
	plt.figure(figsize=(12, 8))  # Can adjust these dimensions as needed
	ax = sns.heatmap(
		pivot_table,
		vmin=10,
		vmax=100,
		fmt="g",
		cmap=cmap,
		cbar_kws={'label': 'Score'},
		linewidths=1.5,
		linecolor='white'
	)

	xtick_labels = pivot_table.columns.tolist()
	xtick_labels_formatted = [f"{int(x)//1000}K" for x in xtick_labels]
	ax.set_xticklabels(xtick_labels_formatted, fontsize=16)

	plt.xlabel('Context Length', fontsize=22)  # X-axis label with larger font
	plt.ylabel('Answer Depth (%)', fontsize=22)  # Y-axis label with larger font
	plt.yticks(rotation=0, fontsize=16)  # Enlarge y-axis labels
	cbar = ax.collections[0].colorbar
	cbar.ax.tick_params(labelsize=16)
	cbar.set_label('Score', fontsize=22)
	plt.tight_layout()  # Fits everything neatly into the figure area
	# save
	log_dir = Path(args.log_name)
	log_dir.mkdir(parents=True, exist_ok=True)
	base_name = Path(args.model_path).stem
	output_stem = log_dir / f"{base_name}_heatmap"
	plt.savefig(f"{output_stem}.png", dpi=300, bbox_inches='tight')
	return output_stem

def main(args):
	device = torch.device(args.device)
	print("Loading model:", args.model_path)
	model = TPPForCausalLM.from_pretrained(
		args.model_path
	).to(device)
	encoding = tiktoken.get_encoding("p50k_base")

	records = []
	ctx_len_list = [1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000]
	for ctx_len in tqdm(ctx_len_list):
		n_garbage = int(3.75 * ctx_len // 1000 * 1000)
		# ten evenly spaced depths
		for depth_idx in range(1, 10):
			n_pref = depth_idx * (n_garbage // 10)
			correct = 0
			tot_tokens = 0
			for seed in range(args.num_tests):
				ok, lt = passkey_retrieval_test(
					model, encoding, device,
					n_garbage_prefix=n_pref,
					n_garbage=n_garbage,
					seed=seed
				)
				correct += ok
				tot_tokens += lt
			acc = correct / args.num_tests * 100
			depth_pct = (n_pref / n_garbage) * 100
			records.append({
				"Context Length": ctx_len,
				"Document Depth": round(depth_pct, -1),
				"Score": acc
			})
	df = pd.DataFrame(records)
	stem = plot_heatmap(df, args)
	df.to_csv(f"{stem}.csv", index=False)

if __name__ == "__main__":
	args = parse_config()
	main(args)
