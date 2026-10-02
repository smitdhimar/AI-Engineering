
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
ENV_KEY = "DEEPSEEK_API_KEY"


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #
def call_deepseek(
    api_key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    stream: bool = False,
    timeout: float = 120.0,
) -> dict:
    """Send one chat completion request and return response + metrics."""
    messages = []
    if system:
        messages.append({"role": "system", "content": "You must answer in 1 word"})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        # Ask the server to include token usage in the final stream chunk.
        payload["stream_options"] = {"include_usage": True}

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()
    if stream:
        return _consume_stream(request, start, timeout)
    return _consume_json(request, start, timeout)


def _consume_json(request: urllib.request.Request, start: float, timeout: float) -> dict:
    """Non-streaming path: one JSON body with usage + full latency."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ApiError(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except urllib.error.URLError as exc:
        raise ApiError(None, f"Network error: {exc.reason}") from exc

    latency = time.perf_counter() - start
    data = json.loads(body)

    choice = data.get("choices", [{}])[0]
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return {
        "text": choice.get("message", {}).get("content", ""),
        "reasoning": choice.get("message", {}).get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        "latency": latency,
        "ttft": None,
        "tokens_per_sec": (
            completion_tokens / latency if completion_tokens and latency else None
        ),
        "raw": data,
    }


def _consume_stream(request: urllib.request.Request, start: float, timeout: float) -> dict:
    """Streaming path: print deltas live, measure time-to-first-token."""
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict = {}
    finish_reason = None
    ttft = None
    first_content = False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    content = delta.get("content")
                    if content:
                        if not first_content:
                            ttft = time.perf_counter() - start
                            first_content = True
                        chunks.append(content)
                        sys.stdout.write(content)
                        sys.stdout.flush()

                    reason = delta.get("reasoning_content")
                    if reason:
                        reasoning_chunks.append(reason)
    except urllib.error.HTTPError as exc:
        raise ApiError(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except urllib.error.URLError as exc:
        raise ApiError(None, f"Network error: {exc.reason}") from exc

    if chunks:
        sys.stdout.write("\n")
    latency = time.perf_counter() - start
    completion_tokens = usage.get("completion_tokens")
    return {
        "text": "".join(chunks),
        "reasoning": "".join(reasoning_chunks) or None,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        "latency": latency,
        "ttft": ttft,
        "tokens_per_sec": (
            completion_tokens / latency if completion_tokens and latency else None
        ),
        "raw": usage,
    }


class ApiError(Exception):
    def __init__(self, status: int | None, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"DeepSeek API error (HTTP {status}): {body}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value, fmt: str = "{}", fallback: str = "n/a") -> str:
    return fallback if value is None else fmt.format(value)


def print_report(result: dict, model: str, stream: bool) -> None:
    line = "=" * 62
    print()
    print(line)
    print("  RESPONSE")
    print(line)
    if not stream:
        if result.get("reasoning"):
            print("[reasoning]")
            print(result["reasoning"].strip())
            print()
        print(result["text"].strip() or "(empty response)")

    print()
    print(line)
    print("  METRICS")
    print(line)
    print(f"  model              : {model}")
    print(f"  finish reason      : {_fmt(result['finish_reason'])}")
    print(f"  input tokens       : {_fmt(result['prompt_tokens'], '{:,}')}")
    print(f"  output tokens      : {_fmt(result['completion_tokens'], '{:,}')}")
    print(f"  total tokens       : {_fmt(result['total_tokens'], '{:,}')}")
    if result.get("cache_hit_tokens") is not None:
        print(
            f"  prompt cache       : hit {_fmt(result['cache_hit_tokens'], '{:,}')}"
            f" / miss {_fmt(result['cache_miss_tokens'], '{:,}')}"
        )
    print(f"  latency            : {_fmt(result['latency'], '{:.3f} s')}")
    if stream:
        print(f"  time to 1st token   : {_fmt(result['ttft'], '{:.3f} s')}")
    print(f"  throughput         : {_fmt(result['tokens_per_sec'], '{:.1f} tok/s')}")
    print(line)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call the DeepSeek chat API and print the response plus token/latency metrics."
    )
    parser.add_argument("prompt", nargs="?", help="User prompt. Omit for interactive mode.")
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"DeepSeek API key (defaults to ${ENV_KEY} env var).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap on output tokens.")
    parser.add_argument("--stream", action="store_true", help="Stream the response token by token.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print raw JSON result.")
    return parser


def resolve_api_key(cli_value: str | None) -> str:
    key = cli_value or os.environ.get(ENV_KEY)
    if key:
        return key.strip()
    if sys.stdin.isatty():
        key = getpass.getpass(f"{ENV_KEY} not set. Paste your DeepSeek API key: ").strip()
        if key:
            return key
    raise SystemExit(
        f"No API key found. Set the {ENV_KEY} environment variable or pass --api-key."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = resolve_api_key(args.api_key)

    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            try:
                prompt = input("Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 130
    if not prompt:
        print("Error: empty prompt.", file=sys.stderr)
        return 2

    try:
        result = call_deepseek(
            api_key=api_key,
            prompt=prompt,
            model=args.model,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True,
            timeout=args.timeout,
        )
    except ApiError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if args.as_json:
        printable = {k: v for k, v in result.items() if k != "raw"}
        print(json.dumps(printable, indent=2))
    else:
        print_report(result, model=args.model, stream=args.stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
