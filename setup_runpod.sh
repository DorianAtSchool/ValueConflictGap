#!/bin/bash
# Setup script for running PersonaDrifting on RunPod.
# Run this once after connecting to your pod:
#   bash setup_runpod.sh
#
# Prerequisites:
#   - You've already cloned/uploaded the repo to the pod
#   - You have a HuggingFace token (for gated Llama model access)
#   - You have an Anthropic or OpenAI API key (for user simulator)

set -e

echo "=========================================="
echo "  PersonaDrifting — RunPod Setup"
echo "=========================================="

# ---- 1. Install Python dependencies ----
echo ""
echo "[1/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ---- 2. HuggingFace login ----
echo ""
echo "[2/4] HuggingFace authentication"
echo "The base model (meta-llama/Llama-3.1-8B-Instruct) is gated."
echo "You must accept the license at:"
echo "  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
echo ""

if [ -z "$HF_TOKEN" ]; then
    read -rp "Enter your HuggingFace token (or press Enter to skip if already logged in): " hf_token
    if [ -n "$hf_token" ]; then
        huggingface-cli login --token "$hf_token"
    fi
else
    echo "Using HF_TOKEN from environment..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# ---- 3. API keys ----
echo ""
echo "[3/4] API key configuration"

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "No API key found in environment."
    echo "You need one for the user simulator (Anthropic or OpenAI)."
    echo ""
    echo "Set one before running the experiment:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    echo "Or pass it inline:"
    echo "  python pipeline/run_experiment.py --simulator-api-key sk-ant-..."
else
    [ -n "$ANTHROPIC_API_KEY" ] && echo "  ANTHROPIC_API_KEY is set."
    [ -n "$OPENAI_API_KEY" ] && echo "  OPENAI_API_KEY is set."
fi

# ---- 4. Verify GPU ----
echo ""
echo "[4/4] GPU check"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected! The experiment requires a CUDA GPU.')
"

# ---- Done ----
echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Quick start (small test run):"
echo "  cd pipeline"
echo "  python run_experiment.py \\"
echo "    --personas sarcasm \\"
echo "    --value-sets HHH \\"
echo "    --domains politics \\"
echo "    --turn-counts 5 \\"
echo "    --simulator-api-key sk-ant-..."
echo ""
echo "Full experiment:"
echo "  cd pipeline"
echo "  python run_experiment.py --simulator-api-key sk-ant-..."
echo ""
echo "Results will be saved to pipeline/results/"
