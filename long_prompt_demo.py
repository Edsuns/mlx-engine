#!/usr/bin/env python3
"""Demo for processing large prompts assembled from mlx_engine/ .py files.

Usage:
    python long_prompt_demo.py --model <path> --max-kv-size 65536 --rounds 5 --prompt-length 100000

Output:
  - Real-time prompt assembly progress (tokens processed / total)
  - Prefill progress (tokens processed / total)
  - Time to first token (TTFT)
  - Token output rate (tok/s) and total output tokens
"""

import argparse
import gc
import glob
import os
import sys
import time

from transformers import AutoTokenizer

from mlx_engine.generate import create_generator, load_model, tokenize
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter
from mlx_engine.utils.token import Token


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Long-prompt inference demo: assemble .py files into a large prompt"
    )
    parser.add_argument("--model", required=True, type=str, help="Path to the model")
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=100000,
        help="Target token count per round. When > 0, reads mlx_engine/ .py files "
        "recursively and assembles a prompt within this token budget.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of times to repeat the .py file input. Each round is appended "
        "to the prompt to reach the target token count.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=65536,
        help="Max context size of the model",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=(2, 3, 4, 6, 8),
        help="Number of bits for KV cache quantization",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        choices=(32, 64, 128),
        help="Group size for KV cache quantization",
    )
    return parser


class PromptReporter(PromptProgressReporter):
    """Reporter that prints real-time prompt processing progress."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.prefill_start = None

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        if is_draft:
            return True
        self.total_prompt_tokens = total_prompt_tokens
        self.prefill_start = time.time()
        self._print(prefill_tokens_processed, total_prompt_tokens)
        return True

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        if is_draft:
            return True
        self._print(prefill_tokens_processed, self.total_prompt_tokens)
        return True

    def finish(
        self, is_draft: bool, prefill_tokens_processed: int | None = None
    ) -> bool:
        if is_draft:
            return True
        tokens = prefill_tokens_processed or 0
        self._print(tokens, self.total_prompt_tokens)
        elapsed = time.time() - self.prefill_start
        print(
            f"\r✓ Prompt processing complete: {self.total_prompt_tokens} ({elapsed:.2f}s)\n",
            flush=True,
        )
        return True

    def _print(self, processed: int, total: int):
        pct = (processed / total * 100) if total > 0 else 0
        print(
            f"\r  Processing prompt: {processed}/{total} tokens ({pct:.1f}%) ",
            end="",
            flush=True,
        )


class StatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0

    def add_tokens(self, tokens: list[Token]):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.total_tokens += len(tokens)

    def print_stats(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        ttft = self.first_token_time - self.start_time if self.first_token_time else 0
        effective_time = total_time - ttft
        tok_per_sec = (
            self.total_tokens / effective_time if effective_time > 0 else float("inf")
        )
        print(f"\n{'=' * 50}")
        print(f"  Time to first token: {ttft:.2f}s")
        print(f"  Output tokens:       {self.total_tokens}")
        print(f"  Output rate:         {tok_per_sec:.2f} tok/s")
        print(f"  Total generation:    {effective_time:.2f}s")
        print(f"{'=' * 50}\n")


def collect_py_files(base_dir: str) -> list[str]:
    pattern = os.path.join(base_dir, "**", "*.py")
    files = sorted(glob.glob(pattern, recursive=True))
    return [
        f
        for f in files
        if "__pycache__" not in f and not os.path.basename(f).startswith(".")
    ]


def build_round_prompt(tokenizer, base_dir: str, max_tokens: int) -> tuple[str, int]:
    """Read .py files within max_tokens budget. Returns (prompt_text, token_count)."""
    py_files = collect_py_files(base_dir)
    if not py_files:
        raise ValueError(f"No .py files found under {base_dir}")

    print(f"Found {len(py_files)} .py files under {base_dir}\n", flush=True)

    parts = []
    for fpath in py_files:
        rel = os.path.relpath(fpath, base_dir)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        header = f"\n{'=' * 60}\n# File: {rel}\n{'=' * 60}\n"
        trial = "\n".join(parts) + header + content
        trial_tokens = len(tokenizer.encode(trial))

        if max_tokens > 0 and trial_tokens > max_tokens:
            # Binary search for max prefix that fits
            lo, hi = 100, len(content)
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                partial = content[:mid]
                t = len(tokenizer.encode("\n".join(parts) + header + partial))
                if t <= max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best > 0:
                partial = content[:best]
                trial = "\n".join(parts) + header + partial
                new_tokens = len(tokenizer.encode(trial))
                print(
                    f"  Added {rel} (partial: {best}/{len(content)} chars, "
                    f"+{new_tokens - len(parts)} tokens) "
                    f"[total: {new_tokens}/{max_tokens}]",
                    flush=True,
                )
                parts.append(header + partial)
            else:
                print(f"  Skipped {rel} (would exceed {max_tokens} tokens)", flush=True)
            break
        else:
            parts.append(header + content)
            print(
                f"  Added {rel} ({len(content)} chars, {trial_tokens} tokens) "
                f"[{trial_tokens}/{max_tokens if max_tokens > 0 else '∞'}]",
                flush=True,
            )

    prompt_str = "\n".join(parts)
    final_count = len(tokenizer.encode(prompt_str))
    return prompt_str, final_count


def resolve_model_path(model_arg):
    if os.path.exists(model_arg):
        return model_arg
    for path in [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
    ]:
        full_path = os.path.join(path, model_arg)
        if os.path.exists(full_path):
            return full_path
    raise ValueError(f"Could not find model '{model_arg}' in local directories")


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load model
    model_path = resolve_model_path(args.model)
    print("Loading model...", end="\n", flush=True)
    model_kit = load_model(
        str(model_path),
        max_kv_size=args.max_kv_size,
        trust_remote_code=False,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        max_seq_nums=1,
    )
    print("\r✓ Model load complete.", end="\n", flush=True)

    # Build prompt from .py files
    if args.prompt_length > 0:
        mlx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_engine")
        if not os.path.isdir(mlx_dir):
            raise ValueError(f"mlx_engine directory not found at {mlx_dir}")

        temp_tokenizer = AutoTokenizer.from_pretrained(model_path)

        text, tokens = build_round_prompt(temp_tokenizer, mlx_dir, args.prompt_length)

        # Append the query prompt
        prompt = text + "\n\nPlease describe these code in 100 words."
        gc.collect()

        total_tokens = len(temp_tokenizer.encode(prompt))
        print(
            f"\nFinal assembled prompt: {tokens} tokens "
            f"+ query suffix, {total_tokens} total tokens\n",
            flush=True,
        )
        del temp_tokenizer
        gc.collect()
    else:
        prompt = args.prompt if args.prompt != "-" else sys.stdin.read()

    # Tokenize with model's tokenizer
    tf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    conversation = [{"role": "user", "content": prompt}]
    prompt_str = tf_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenize(model_kit, prompt_str)
    print(
        f"\r✓ Tokenized prompt with chat template: {len(prompt_tokens)} tokens\n",
        flush=True,
    )

    # Generate
    for r in range(1, args.rounds + 1):
        print(f"==== round {r} start ====\n", flush=True)
        stats = StatsCollector()
        reporter = PromptReporter()
        generator = create_generator(
            model_kit,
            prompt_tokens,
            max_tokens=args.max_tokens,
            temp=args.temp,
            prompt_progress_reporter=reporter,
        )

        for result in generator:
            print(result.text, end="", flush=True)
            stats.add_tokens(result.tokens)
            if result.stop_condition:
                stats.print_stats()
                print(f"Stopped: {result.stop_condition.stop_reason}")
                if result.stop_condition.stop_string:
                    print(f"Stop string: {result.stop_condition.stop_string}")
        print(f"\n==== round {r} end ====\n", flush=True)
