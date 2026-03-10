"""Local model with LoRA persona swapping via peft.

Loads adapters from maius/llama-3.1-8b-it-personas using subfolder per persona,
as documented at https://huggingface.co/maius/llama-3.1-8b-it-personas.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional

from config import BASE_MODEL, PERSONA_HUB, MAX_NEW_TOKENS_MCQ, MAX_NEW_TOKENS_CONVERSATION


class PersonaModel:
    """Wraps a base Llama model with hot-swappable LoRA persona adapters."""

    def __init__(self, base_model: str = BASE_MODEL, device_map: str = "auto", use_personas: bool = True):
        self.base_model_id = base_model
        self.use_personas = use_personas
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.peft_model: Optional[PeftModel] = None
        self.current_persona: Optional[str] = None

    @property
    def device(self):
        return self.base_model.device

    def load_persona(self, persona: str):
        """Load or swap a LoRA adapter for the given persona.

        Uses subfolder= to select the persona from the single HF repo,
        per the maius/llama-3.1-8b-it-personas documentation.
        Skips LoRA loading if use_personas=False.
        """
        if not self.use_personas:
            self.current_persona = persona
            return

        if persona == self.current_persona:
            return

        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                PERSONA_HUB,
                subfolder=persona,
                adapter_name=persona,
            )
        else:
            try:
                self.peft_model.set_adapter(persona)
            except ValueError:
                self.peft_model.load_adapter(
                    PERSONA_HUB,
                    subfolder=persona,
                    adapter_name=persona,
                )
                self.peft_model.set_adapter(persona)

        self.peft_model.eval()
        self.current_persona = persona

    @property
    def model(self):
        return self.peft_model if self.peft_model is not None else self.base_model

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = MAX_NEW_TOKENS_CONVERSATION,
        temperature: float = 0.7,
    ) -> str:
        """Generate a single response from a chat message list."""
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def batch_generate(
        self,
        messages_list: list[list[dict]],
        max_new_tokens: int = MAX_NEW_TOKENS_MCQ,
        temperature: float = 0.0,
        batch_size: int = 32,
    ) -> list[str]:
        """Batch-generate responses for a list of message sequences.

        Automatically halves batch size on OOM and retries, down to batch_size=1.
        """
        all_responses = []
        i = 0
        current_bs = batch_size
        while i < len(messages_list):
            batch = messages_list[i : i + current_bs]
            texts = [
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in batch
            ]
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        top_p=0.9 if temperature > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                for j, ids in enumerate(output_ids):
                    new_tokens = ids[inputs["input_ids"].shape[1]:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    all_responses.append(text)
                i += current_bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    print(f"  OOM: reducing batch size to {current_bs}")
                else:
                    print(f"  OOM at batch_size=1, skipping sample {i}")
                    all_responses.append("")
                    i += 1
        return all_responses
