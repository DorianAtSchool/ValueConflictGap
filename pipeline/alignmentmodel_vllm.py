"""vLLM-backed AlignmentModel for multi-GPU tensor parallelism.

Drop-in replacement for AlignmentModel using vLLM for efficient multi-GPU inference.

SECURITY NOTE (2025-2026):
    vLLM has had multiple critical vulnerabilities (CVEs: 2025-59425, 2025-62164,
    2025-66448, CVE-2026-22778, CVE-2026-228). All fixed in v0.14.1+.
    - Current installed: 0.18.0 (SAFE)
    - If concerned, consider alternatives: SGLang, llama.cpp, or HuggingFace TGI.

    Install safe version: pip install "vllm>=0.14.1"
"""

import gc
import logging
import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from config import MAX_NEW_TOKENS_MCQ, MAX_NEW_TOKENS_CONVERSATION, BATCH_SIZE

logger = logging.getLogger(__name__)


def estimate_gpus_needed(hf_id: str, memory_per_gpu_gb: float = 40.0) -> int:
    """Estimate number of GPUs needed for a model.

    Assumes roughly 2 bytes per parameter (bfloat16) with 20% overhead.
    Typical A100 GPUs have ~40GB usable memory.

    Args:
        hf_id: HuggingFace model ID
        memory_per_gpu_gb: Usable memory per GPU (default 40GB for A100)

    Returns:
        Number of GPUs needed
    """
    # Map common model IDs to parameter counts
    # VRAM estimate: ~2 bytes/param (bfloat16) + 20% overhead
    # 8B fits on 1xA100 (40GB), 70B+ needs multiple GPUs
    param_counts = {
        # Llama 3.1 family
        "meta-llama/Llama-3.1-8B": 8,
        "meta-llama/Llama-3.1-8B-Instruct": 8,
        "meta-llama/Llama-3.1-70B": 70,
        "meta-llama/Llama-3.1-70B-Instruct": 70,
        "meta-llama/Llama-3.1-405B": 405,
        # Tulu 3 family (both 8B and 70B variants)
        "allenai/Llama-3.1-Tulu-3-8B-SFT": 8,
        "allenai/Llama-3.1-Tulu-3-8B-DPO": 8,
        "allenai/Llama-3.1-Tulu-3-8B": 8,
        "allenai/Llama-3.1-Tulu-3-70B": 70,
        "allenai/Llama-3.1-Tulu-3-70B-SFT": 70,
        "allenai/Llama-3.1-Tulu-3-70B-DPO": 70,
        # OLMo 2
        "allenai/OLMo-2-0325-32B-Instruct": 32,
        # Qwen 2.5 family
        "Qwen/Qwen2.5-7B-Instruct": 7,
        "Qwen/Qwen2.5-72B-Instruct": 72,
        # Gemma 2
        "google/gemma-2-9b-it": 9,
        "google/gemma-2-27b-it": 27,
        # Mistral Nemo
        "MistralAI/Mistral-Nemo-Instruct-2407": 12,
    }

    param_b = param_counts.get(hf_id)
    if param_b is None:
        # Fallback: try to parse from model ID
        logger.warning(f"Unknown model {hf_id}, assuming 8B parameters")
        param_b = 8

    # Estimate memory needed: 2 bytes per param + 20% overhead
    memory_needed = param_b * 2 * 1.2

    # Calculate GPUs needed (with some headroom)
    n_gpus = max(1, int((memory_needed + memory_per_gpu_gb - 1) / memory_per_gpu_gb))
    return n_gpus


class AlignmentModelVLLM:
    """Wraps a single HF model using vLLM for efficient multi-GPU inference.

    Satisfies the same interface as AlignmentModel for drop-in compatibility.
    """

    def __init__(
        self,
        hf_id: str,
        is_base_model: bool = False,
        gpu_ids: Optional[list[int]] = None,
        gpu_memory_utilization: float = 0.9,
    ):
        """Initialize vLLM-backed model.

        Args:
            hf_id: HuggingFace model ID
            is_base_model: Whether this is a base (non-chat-tuned) model
            gpu_ids: List of GPU IDs to use. If None, auto-detects based on memory.
            gpu_memory_utilization: vLLM GPU memory utilization (0.0-1.0)
        """
        self.hf_id = hf_id
        self.is_base_model = is_base_model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.current_persona = None

        try:
            import vllm
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        # Set up GPU visibility
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            self.gpu_ids = gpu_ids
            tensor_parallel_size = len(gpu_ids)
        else:
            # Auto-detect
            tensor_parallel_size = estimate_gpus_needed(hf_id)
            self.gpu_ids = list(range(tensor_parallel_size))

        logger.info(
            f"Loading {hf_id} with tensor_parallel_size={tensor_parallel_size}, "
            f"gpu_memory_utilization={gpu_memory_utilization}"
        )

        # Initialize vLLM
        self.llm = vllm.LLM(
            model=hf_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,  # Works better on some hardware
        )

        # Load tokenizer separately for formatting
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def load_persona(self, persona: str):
        """No-op — satisfies the probing.py interface."""
        self.current_persona = persona

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert chat messages to a text prompt.

        For chat-tuned models, uses the tokenizer's built-in chat template.
        For base models, uses a simple format.
        """
        if not self.is_base_model:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Base model: simple structured format
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            if role == "System":
                parts.append(f"System: {msg['content'].strip()}\n\n")
            elif role == "User":
                parts.append(f"User: {msg['content'].strip()}\n\n")
            elif role == "Assistant":
                parts.append(f"Assistant: {msg['content'].strip()}\n\n")
        parts.append("Assistant:")
        return "".join(parts)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = MAX_NEW_TOKENS_CONVERSATION,
        temperature: float = 0.7,
    ) -> str:
        """Generate a single response."""
        input_text = self._format_messages(messages)

        # Use vLLM's generate API
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.9 if temperature > 0 else 1.0,
        )

        outputs = self.llm.generate([input_text], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Base models may continue generating additional turns — truncate
        if self.is_base_model:
            for stop in ["\nUser:", "\nHuman:", "\nSystem:"]:
                if stop in response:
                    response = response[: response.index(stop)].strip()

        return response

    def batch_generate(
        self,
        messages_list: list[list[dict]],
        max_new_tokens: int = MAX_NEW_TOKENS_MCQ,
        temperature: float = 0.0,
        batch_size: int = BATCH_SIZE,
    ) -> list[str]:
        """Generate responses for a batch of messages with adaptive batching."""
        from vllm import SamplingParams

        all_responses = []
        i = 0
        current_bs = batch_size

        while i < len(messages_list):
            batch = messages_list[i : i + current_bs]
            texts = [self._format_messages(msgs) for msgs in batch]

            try:
                sampling_params = SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.0,
                    top_p=0.9 if temperature > 0 else 1.0,
                )

                outputs = self.llm.generate(texts, sampling_params)

                for output in outputs:
                    response = output.outputs[0].text.strip()
                    # Base models may continue — truncate
                    if self.is_base_model:
                        for stop in ["\nUser:", "\nHuman:", "\nSystem:"]:
                            if stop in response:
                                response = response[: response.index(stop)].strip()
                    all_responses.append(response)

                i += current_bs

            except RuntimeError as e:
                # Handle OOM by reducing batch size
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    torch.cuda.empty_cache()
                    if current_bs > 1:
                        current_bs = max(1, current_bs // 2)
                        logger.warning(f"OOM: reducing batch size to {current_bs}")
                    else:
                        logger.warning(f"OOM at batch_size=1, skipping batch starting at {i}")
                        all_responses.extend([""] * len(batch))
                        i += len(batch)
                else:
                    raise

        return all_responses

    def unload(self):
        """Free GPU memory so the next model can load."""
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        self.llm = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
