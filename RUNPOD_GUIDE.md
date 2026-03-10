# Running PersonaDrifting on RunPod

This guide walks you through setting up and running the persona drifting experiment on RunPod GPU pods.

## What the experiment does

The pipeline loads **Llama-3.1-8B-Instruct** with LoRA persona adapters on a local GPU, runs multi-turn conversations using a cloud LLM (Claude or GPT) as the user simulator, then probes value rankings before and after conversation to measure "persona drift." The GPU is needed for the local Llama inference.

## 1. Create a RunPod Pod

1. Log into [RunPod](https://runpod.io) and go to **Pods** > **Deploy**
2. Choose a **GPU Pod** (not Serverless)

### GPU selection

The base model is Llama-3.1-8B in bfloat16 (~16 GB VRAM). Recommended GPUs:

| GPU | VRAM | Notes |
|-----|------|-------|
| RTX 4090 | 24 GB | Cheapest option that works |
| A6000 | 48 GB | Comfortable headroom |
| A100 40GB | 40 GB | Fast, good for full experiment |
| A100 80GB | 80 GB | Overkill but very comfortable |

### Template and storage

- **Template**: Use **RunPod PyTorch** (pre-installed CUDA + PyTorch)
- **Container Disk**: 50 GB minimum (model weights ~16 GB + code + results)
- **Network Volume** (optional but recommended): Attach one so results persist if you delete the pod

## 2. Connect to the Pod

Once the pod is running, click **Connect** and choose:
- **Web Terminal** — quick shell access
- **Jupyter Lab** — if you prefer a notebook/file browser UI
- **SSH** — if you set up your SSH key in RunPod settings

## 3. Get the Code onto the Pod

```bash
# Option A: clone from git
git clone <your-repo-url> PersonaDrifting
cd PersonaDrifting

# Option B: upload via Jupyter Lab file browser
# Option C: scp from your local machine
```

## 4. Run the Setup Script

```bash
cd PersonaDrifting
bash setup_runpod.sh
```

This script will:
1. Install all Python dependencies from `requirements.txt`
2. Prompt you to log into HuggingFace (needed for the gated Llama model)
3. Check for API keys
4. Verify GPU availability

### Before running the script

- **Accept the Llama license**: Go to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on HuggingFace and accept the license agreement
- **Have your HuggingFace token ready**: Get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Have an API key** for the user simulator (Anthropic or OpenAI)

## 5. Run the Experiment

### Quick test (recommended first)

Run a minimal subset to verify everything works:

```bash
cd pipeline

python run_experiment.py \
    --personas sarcasm \
    --value-sets HHH \
    --domains politics \
    --turn-counts 5 \
    --simulator-api-key sk-ant-...
```

### Full experiment

```bash
python run_experiment.py --simulator-api-key sk-ant-...
```

This runs all 10 personas x 3 value sets x all domains x [5, 10, 20] turn counts.

### Using OpenAI instead of Anthropic

```bash
python run_experiment.py \
    --simulator openai \
    --simulator-model gpt-4o-mini \
    --simulator-api-key sk-...
```

### Other useful flags

```bash
# Specific personas only
--personas sarcasm loving goodness

# Specific value sets
--value-sets HHH modelspec

# Only generic domains (politics, therapy, philosophy, coding)
--generic-only

# Only value-aligned domains
--value-aligned-only

# Custom turn counts
--turn-counts 5 10
```

## 6. Results

All output goes to `pipeline/results/`:

```
pipeline/results/
├── all_results.csv          # Aggregated results table
├── experiment.log           # Full log
├── checkpoints/             # Per-condition checkpoints (for resume)
├── conversations/           # Generated conversations
├── runs/                    # Per-condition JSON results
└── plots/                   # Radar charts, heatmaps, trajectory plots
```

## 7. Checkpointing and Resume

The experiment checkpoints after every condition. If your pod dies or you stop the run, just re-run the same command — it will skip completed conditions and pick up where it left off.

## 8. Saving Results Before Stopping the Pod

RunPod container disk is **ephemeral** — if you delete the pod, the data is gone. Before stopping:

```bash
# Option A: push to git
cd PersonaDrifting
git add pipeline/results/
git commit -m "experiment results"
git push

# Option B: copy to a network volume (if attached)
cp -r pipeline/results/ /workspace/results-backup/

# Option C: download via Jupyter Lab file browser

# Option D: scp to your local machine
# (from your local machine)
scp -r root@<pod-ip>:PersonaDrifting/pipeline/results/ ./results/
```

## 9. Cost Tips

- RunPod charges **per hour** while the pod is running
- **Stop** the pod (not delete) to pause billing while keeping data
- The full experiment takes several hours depending on GPU speed and API latency
- Start with the quick test to estimate total runtime before committing to the full run
- Use `tmux` or `screen` so the experiment survives if your browser disconnects:
  ```bash
  tmux new -s experiment
  cd PersonaDrifting/pipeline
  python run_experiment.py --simulator-api-key sk-ant-...
  # Ctrl+B then D to detach; tmux attach -t experiment to reconnect
  ```
