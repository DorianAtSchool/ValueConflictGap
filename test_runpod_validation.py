#!/usr/bin/env python3
"""RunPod validation test - quick sanity check before full experiment.

This script validates that your RunPod instance is properly configured for
the multi-GPU scenario conversation experiment. Run this first to catch any issues.

Usage on RunPod:
    cd /workspace/PersonaDrifting
    python test_runpod_validation.py

Expected output on success:
    ✓ GPUs detected
    ✓ vLLM installed
    ✓ Models loadable
    ✓ Quick inference test passed
    Ready to run: python pipeline/run_runpod.py --models all ...
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def section(title: str):
    """Print a section header."""
    log.info("=" * 80)
    log.info(title)
    log.info("=" * 80)


def gpu_info():
    """Get detailed GPU information."""
    section("GPU Configuration")

    n_gpus = torch.cuda.device_count()
    log.info(f"Total GPUs: {n_gpus}")

    if n_gpus == 0:
        log.error("✗ No GPUs detected!")
        return False

    total_memory_gb = 0
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        total_memory_gb += memory_gb
        name = torch.cuda.get_device_name(i)
        log.info(f"  GPU {i}: {name} - {memory_gb:.1f}GB")

    log.info(f"Total memory: {total_memory_gb:.1f}GB")

    # CUDA info
    cuda_version = torch.version.cuda
    log.info(f"CUDA: {cuda_version}")
    log.info(f"cuDNN: {torch.backends.cudnn.version()}")
    log.info(f"PyTorch: {torch.__version__}")

    return True


def check_vllm():
    """Check if vLLM is installed."""
    section("vLLM Installation")

    try:
        import vllm

        version = vllm.__version__
        log.info(f"✓ vLLM {version} installed")

        # Try to list available GPUs in vLLM
        try:
            from vllm.utils import get_gpu_memory

            memory = get_gpu_memory()
            log.info(f"vLLM detected {len(memory)} GPUs")
            for gpu_id, mem_gb in memory.items():
                log.info(f"  GPU {gpu_id}: {mem_gb / 1e9:.1f}GB available")
        except Exception as e:
            log.warning(f"Could not get vLLM GPU memory: {e}")

        return True
    except ImportError:
        log.error("✗ vLLM not installed!")
        log.error("Install with: pip install vllm")
        return False


def check_models():
    """Check if major models can be downloaded/accessed."""
    section("Model Availability Check")

    models_to_check = [
        "allenai/Llama-3.1-Tulu-3-8B-SFT",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B",
    ]

    log.info(f"Checking {len(models_to_check)} models...")

    try:
        from huggingface_hub import model_info

        for model_id in models_to_check:
            try:
                info = model_info(model_id)
                gated = info.gated
                library = info.library_name

                status = "⚠ GATED" if gated else "✓"
                log.info(f"  {status} {model_id} ({library})")

                if gated:
                    log.warning(
                        f"    → This model is gated. Ensure you have HF_TOKEN env var set."
                    )
            except Exception as e:
                log.warning(f"  ⚠ Could not check {model_id}: {e}")

        return True
    except Exception as e:
        log.warning(f"Could not check model info: {e}")
        return True


def check_api_keys():
    """Check if API keys are configured."""
    section("API Key Configuration")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        masked = anthropic_key[:10] + "***" + anthropic_key[-4:]
        log.info(f"✓ ANTHROPIC_API_KEY set ({masked})")
    else:
        log.warning("⚠ ANTHROPIC_API_KEY not set (required for Anthropic simulator)")

    if openai_key:
        masked = openai_key[:10] + "***" + openai_key[-4:]
        log.info(f"✓ OPENAI_API_KEY set ({masked})")
    else:
        log.warning("⚠ OPENAI_API_KEY not set (required for OpenAI simulator)")

    if not (anthropic_key or openai_key):
        log.error("✗ At least one API key must be set!")
        return False

    return True


def test_vllm_inference():
    """Test actual vLLM inference."""
    section("vLLM Inference Test")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
        from alignmentmodel_vllm import AlignmentModelVLLM

        log.info("Loading small model for inference test...")
        log.info("(This downloads ~17GB, may take several minutes on first run)")

        start = time.time()

        model = AlignmentModelVLLM(
            hf_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
            is_base_model=False,
        )

        elapsed = time.time() - start
        log.info(f"✓ Model loaded in {elapsed:.1f}s")

        # Test single generation
        messages = [{"role": "user", "content": "Say 'test success' in one word."}]

        log.info("Testing inference...")
        response = model.generate(messages, max_new_tokens=10, temperature=0.0)

        log.info(f"✓ Generation successful: '{response}'")

        # Test batch generation
        log.info("Testing batch generation...")
        batch = [messages] * 3
        responses = model.batch_generate(batch, max_new_tokens=10, batch_size=4)

        log.info(f"✓ Batch generation successful ({len(responses)} samples)")

        model.unload()
        log.info("✓ Model unloaded successfully")

        return True

    except Exception as e:
        log.error(f"✗ Inference test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_experiment_setup():
    """Test that experiment script can be imported."""
    section("Experiment Setup Test")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))

        log.info("Importing experiment script...")
        import run_scenario_conversation_experiment

        log.info("Checking GPU allocation function...")
        allocate_gpus = run_scenario_conversation_experiment.allocate_gpus

        # Test allocation
        gpus = list(range(torch.cuda.device_count()))
        models = ["tulu-3-sft", "tulu-3-dpo", "tulu-3-rlvr"]
        allocation = allocate_gpus(models, gpus)

        log.info(f"✓ GPU allocation works:")
        for model, gpu_ids in allocation.items():
            log.info(f"  {model}: GPUs {gpu_ids}")

        return True

    except Exception as e:
        log.error(f"✗ Experiment setup failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_storage():
    """Check storage space."""
    section("Storage Check")

    try:
        result = subprocess.run(
            ["df", "-h", "/workspace"],
            capture_output=True,
            text=True,
        )

        log.info("Storage at /workspace:")
        log.info(result.stdout)

        # Also check PersonaDrifting directory
        if Path("/workspace/PersonaDrifting").exists():
            log.info("✓ PersonaDrifting directory found")
        else:
            log.warning("⚠ PersonaDrifting directory not found at /workspace")

        return True
    except Exception as e:
        log.warning(f"Could not check storage: {e}")
        return True


def generate_report(results: dict) -> dict:
    """Generate summary report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_count": torch.cuda.device_count(),
        "total_gpu_memory_gb": sum(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(torch.cuda.device_count())
        ),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "tests": results,
        "ready_for_experiment": all(results.values()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate RunPod setup for multi-GPU experiment"
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip the inference test (faster validation)",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save validation report to JSON file",
    )
    args = parser.parse_args()

    log.info("PersonaDrifting RunPod Validation Test")
    log.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Core checks
    results["GPU Detection"] = gpu_info()
    results["vLLM Installation"] = check_vllm()
    results["Model Availability"] = check_models()
    results["API Keys"] = check_api_keys()
    results["Storage"] = test_storage()

    # Optional inference test
    if not args.no_inference:
        results["vLLM Inference"] = test_vllm_inference()
    else:
        log.info("Skipping inference test (--no-inference flag)")

    results["Experiment Setup"] = test_experiment_setup()

    # Summary
    section("VALIDATION SUMMARY")

    report = generate_report(results)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓" if result else "✗"
        log.info(f"{status} {test_name}")

    log.info("-" * 80)
    log.info(f"Passed: {passed}/{total}")

    if report["ready_for_experiment"]:
        log.info("✓ READY FOR EXPERIMENT!")
        log.info("")
        log.info("Next steps:")
        log.info("  1. Set API key: export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY")
        log.info("  2. Run experiment: python pipeline/run_runpod.py --models all ...")
        log.info("")
        exit_code = 0
    else:
        log.error("✗ VALIDATION FAILED - Fix issues above before running experiment")
        exit_code = 1

    # Save report if requested
    if args.save_report:
        report_path = Path(args.save_report)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Report saved to: {report_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
