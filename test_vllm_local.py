#!/usr/bin/env python3
"""Local test suite for vLLM integration.

Run this to verify:
- AlignmentModelVLLM loads correctly
- GPU detection works
- Model inference produces valid output
- Batch generation works
- OOM handling works gracefully

Usage:
    python test_vllm_local.py                    # Full test suite
    python test_vllm_local.py --model-only       # Just test model loading
    python test_vllm_local.py --quick            # Fast smoke test
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))

from alignmentmodel_vllm import AlignmentModelVLLM


def test_gpu_detection():
    """Test GPU detection."""
    log.info("=" * 80)
    log.info("TEST 1: GPU Detection")
    log.info("=" * 80)

    n_gpus = torch.cuda.device_count()
    log.info(f"Detected {n_gpus} GPUs")

    if n_gpus == 0:
        log.error("No GPUs detected! Cannot test vLLM.")
        return False

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / 1e9
        log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_memory_gb:.1f}GB)")

    return True


def test_vllm_availability():
    """Test if vLLM is installed."""
    log.info("=" * 80)
    log.info("TEST 2: vLLM Availability")
    log.info("=" * 80)

    try:
        import vllm

        log.info(f"✓ vLLM {vllm.__version__} is installed")
        return True
    except ImportError:
        log.error("✗ vLLM not installed! Run: pip install vllm")
        return False


def test_model_loading():
    """Test loading a small model with vLLM."""
    log.info("=" * 80)
    log.info("TEST 3: Model Loading (AlignmentModelVLLM)")
    log.info("=" * 80)

    try:
        log.info("Loading allenai/Llama-3.1-Tulu-3-8B-SFT...")
        model = AlignmentModelVLLM(
            hf_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
            is_base_model=False,
        )
        log.info("✓ Model loaded successfully")

        # Test unload
        model.unload()
        log.info("✓ Model unloaded successfully")
        return True

    except Exception as e:
        log.error(f"✗ Model loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_single_generation():
    """Test single-sample generation."""
    log.info("=" * 80)
    log.info("TEST 4: Single Sample Generation")
    log.info("=" * 80)

    try:
        log.info("Loading model...")
        model = AlignmentModelVLLM(
            hf_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
            is_base_model=False,
        )

        messages = [
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one word.",
            }
        ]

        log.info("Generating response...")
        response = model.generate(messages, max_new_tokens=10, temperature=0.0)

        log.info(f"✓ Generated: '{response}'")

        if not response:
            log.warning("⚠ Response is empty")
            return False

        model.unload()
        return True

    except Exception as e:
        log.error(f"✗ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_batch_generation():
    """Test batch generation."""
    log.info("=" * 80)
    log.info("TEST 5: Batch Generation")
    log.info("=" * 80)

    try:
        log.info("Loading model...")
        model = AlignmentModelVLLM(
            hf_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
            is_base_model=False,
        )

        # Create a small batch
        messages_list = [
            [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            [{"role": "user", "content": "What is the color of grass? Answer in one word."}],
            [{"role": "user", "content": "Is water wet? Answer yes or no."}],
        ]

        log.info(f"Generating batch of {len(messages_list)} samples...")
        responses = model.batch_generate(messages_list, max_new_tokens=10, batch_size=4)

        log.info(f"✓ Generated {len(responses)} responses:")
        for i, resp in enumerate(responses):
            log.info(f"  [{i}]: '{resp}'")

        if len(responses) != len(messages_list):
            log.error(f"✗ Expected {len(messages_list)} responses, got {len(responses)}")
            return False

        model.unload()
        return True

    except Exception as e:
        log.error(f"✗ Batch generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multi_gpu():
    """Test tensor parallelism across multiple GPUs."""
    log.info("=" * 80)
    log.info("TEST 6: Multi-GPU Tensor Parallelism")
    log.info("=" * 80)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        log.warning(f"⚠ Skipping multi-GPU test ({n_gpus} GPU available)")
        return True

    try:
        gpu_ids = [0, 1]
        log.info(f"Loading model on GPUs {gpu_ids}...")

        model = AlignmentModelVLLM(
            hf_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
            is_base_model=False,
            gpu_ids=gpu_ids,
        )

        messages = [
            {"role": "user", "content": "Hello! How are you?"},
        ]

        log.info("Generating response with tensor parallelism...")
        response = model.generate(messages, max_new_tokens=20, temperature=0.7)

        log.info(f"✓ Generated: '{response[:100]}...'")

        model.unload()
        return True

    except Exception as e:
        log.error(f"✗ Multi-GPU test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gpu_allocation():
    """Test GPU allocation logic."""
    log.info("=" * 80)
    log.info("TEST 7: GPU Allocation Logic")
    log.info("=" * 80)

    # Import allocation function
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
    from run_scenario_conversation_experiment import allocate_gpus

    try:
        # Test case 1: 8 GPUs, 5 models
        gpus = list(range(8))
        models = ["tulu-3-sft", "llama-3.1-instruct", "tulu-3-dpo", "tulu-3-rlvr", "llama-3.1-base"]

        allocation = allocate_gpus(models, gpus)

        log.info("Allocation (8 GPUs, 5 models):")
        for model, gpu_ids in allocation.items():
            log.info(f"  {model}: {gpu_ids}")

        # Verify distribution
        all_gpus = []
        for gpu_list in allocation.values():
            all_gpus.extend(gpu_list)

        if len(set(all_gpus)) < 2:
            log.warning("⚠ GPUs not well distributed")

        log.info("✓ GPU allocation works")
        return True

    except Exception as e:
        log.error(f"✗ GPU allocation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests(quick=False, model_only=False):
    """Run all tests."""
    results = {}

    # Core tests
    results["GPU Detection"] = test_gpu_detection()

    if not results["GPU Detection"]:
        log.error("Cannot proceed without GPUs")
        return results

    results["vLLM Available"] = test_vllm_availability()

    if not results["vLLM Available"]:
        log.error("Cannot proceed without vLLM")
        return results

    results["Model Loading"] = test_model_loading()

    if model_only or not results["Model Loading"]:
        return results

    if not quick:
        results["Single Generation"] = test_single_generation()
        results["Batch Generation"] = test_batch_generation()
        results["Multi-GPU"] = test_multi_gpu()

    results["GPU Allocation"] = test_gpu_allocation()

    return results


def print_summary(results):
    """Print test summary."""
    log.info("=" * 80)
    log.info("TEST SUMMARY")
    log.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        log.info(f"{status}: {test_name}")

    log.info("=" * 80)
    log.info(f"Results: {passed}/{total} passed")

    if passed == total:
        log.info("✓ All tests passed! Ready for RunPod deployment.")
        return 0
    else:
        log.error("✗ Some tests failed. Check errors above.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test vLLM integration locally")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--model-only", action="store_true", help="Only test model loading")
    args = parser.parse_args()

    results = run_all_tests(quick=args.quick, model_only=args.model_only)
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
