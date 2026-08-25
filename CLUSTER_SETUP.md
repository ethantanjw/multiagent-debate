# Cluster Setup Guide

## Initial Setup (One Time)

### 1. Clone Repository on Cluster
```bash
cd ~/projects/aip-rgrosse/
git clone https://github.com/Jerick-1380/LLM-Conversation.git
cd LLM-Conversation
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create .env File with API Keys
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
HF_TOKEN=hf_your_huggingface_token_here
EOF
```

**IMPORTANT**: Replace with your real API keys!
- Get OpenAI key from: https://platform.openai.com/api-keys
- Get HuggingFace token from: https://huggingface.co/settings/tokens

### 4. Login to HuggingFace and Accept Llama License
```bash
# Set your token as environment variable
export HF_TOKEN=hf_YOUR_TOKEN_HERE

# Login
huggingface-cli login --token $HF_TOKEN

# Visit this page and accept the license:
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### 5. Download Llama-3.1-8B-Instruct Model (One Time, ~16GB)

**Option A: Download via SLURM job (recommended)**
```bash
sbatch setup_download_model.sh
# Monitor with: tail -f slurm/output/<job_id>_download_llama.out
```

**Option B: Download interactively**
```bash
# Request interactive session
salloc --account=aip-rgrosse --time=2:00:00 --mem=32G --cpus-per-task=4

# Set environment
export MODEL_CACHE_DIR=~/scratch/models
source .venv/bin/activate

# Download
python download_model.py

# Exit interactive session
exit
```

The model will be stored in `~/scratch/models/` for fast access during experiments.

## Running Experiments

### Update Code from GitHub
Before each run, ensure you have the latest code:

```bash
cd ~/projects/aip-rgrosse/LLM-Conversation
git pull origin main
```

### Submit SLURM Job

**For Llama-3.1-8B-Instruct (local model on GPU):**
```bash
sbatch slurm_llama.sh
```

**For GPT models (OpenAI API, no GPU needed):**
```bash
# First, edit config.py to use GPT models:
# GROUP_MODEL = 'gpt-4o-mini'
# ADV_MODEL = 'gpt-4o-mini'

sbatch your_slurm_script.sh
```

### Monitor Job
```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f slurm/output/<job_id>_conversation.out

# Check for errors
grep -i error slurm/output/<job_id>_conversation.out
```

## Troubleshooting

### Error: "FileNotFoundError: .env"
Create the `.env` file with your OpenAI API key (see step 3 above).

### Error: "FileNotFoundError: results/generation_X/..."
This is fixed in the latest code. Make sure to run `git pull origin main`.

### Error: Code mismatch between local and cluster
```bash
# On cluster
cd ~/projects/aip-rgrosse/LLM-Conversation
git fetch origin
git reset --hard origin/main
git clean -fd
```

**WARNING**: This will delete any uncommitted local changes on the cluster!

### Error: Model not loading / Out of memory
```bash
# Check GPU availability
nvidia-smi

# Make sure you requested a GPU in SLURM:
# --gres=gpu:l40s:1

# Llama-3.1-8B requires ~16GB GPU memory
# L40S has 48GB, so it should fit
```

### Model not found in cache
If the model wasn't downloaded properly:
```bash
# Check if model exists
ls -lh ~/scratch/models/models--meta-llama--Llama-3.1-8B-Instruct/

# Re-download if needed
sbatch setup_download_model.sh
```

### Check Results
```bash
# View generation results
ls -lh results/

# Check specific generation
ls -lh results/generation_0/

# View conversation samples
cat results/generation_0/gen0_var0/conversation_samples.json | head -50
```

## Configuration

Edit [config.py](config.py) to change:
- Dataset (`EVOLUTION_DATASET`)
- Number of samples per test (`EVOLUTION_N_SAMPLES`)
- Number of generations (`EVOLUTION_N_GENERATIONS`)
- Population size (`EVOLUTION_POPULATION_SIZE`)
- Model selection (`LLM_MODEL`)

## Cost Estimation

With `gpt-4o-mini`:
- Single generation (8 prompts × 50 questions): ~$2-3
- Full evolution (5 generations): ~$10-15

Reduce `EVOLUTION_N_SAMPLES` in config.py to save costs during testing.
