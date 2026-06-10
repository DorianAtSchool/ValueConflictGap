"""Judge-client helpers for open-ended classification paths."""

from __future__ import annotations

import os
import sys


def build_judge_client(args):
    """Build an Anthropic or OpenAI judge client from parsed CLI args."""
    if args.judge == "anthropic":
        if args.judge_api_key:
            os.environ["ANTHROPIC_API_KEY"] = args.judge_api_key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: No Anthropic API key for judge. Pass --judge-api-key or set ANTHROPIC_API_KEY.")
            sys.exit(1)
        import anthropic

        return anthropic.Anthropic()

    if args.judge_api_key:
        os.environ["OPENAI_API_KEY"] = args.judge_api_key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No OpenAI API key for judge. Pass --judge-api-key or set OPENAI_API_KEY.")
        sys.exit(1)
    import openai

    return openai.OpenAI()
