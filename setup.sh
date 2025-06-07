pip install --upgrade pip
pip install torch torchvision torchaudio torchao torchtune transformers datasets wandb tiktoken boto3 deepspeed ninja seaborn matplotlib gdown "huggingface_hub[cli]" -U
pip install pytorch-lightning==1.9.5

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install .
pip install lm_eval["ruler"]
cd ..
rm -rf ./lm-evaluation-harness
