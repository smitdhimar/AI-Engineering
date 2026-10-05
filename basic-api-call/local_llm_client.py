
#!/usr/bin/env python3
"""
Local LLM chat client for an Ollama server.

Posts to http://127.0.0.1:11434/api/chat (default model qwen3.5:9b) and prints
both the response and the metrics worth monitoring:

  - input (prompt) token count   <- prompt_eval_count
  - output (eval) token count    <- eval_count
  - response text
  - latency (wall clock, server-side timings, time-to-first-token)

Everything that shapes the request is configurable, via a JSON config file,
CLI flags, or interactive /commands in --chat mode:
  - host / model
  - system message and the role used for it
  - conversation history (load from a file, save to a file, or keep in --chat)
  - the role of the message you send
  - temperature, top_p, num_predict (max tokens), seed, thinking mode

Usage:
    python3 local_llm_client.py "Why is the sky blue?"
    python3 local_llm_client.py --system "Answer in one word." --temperature 0.2 "Capital of France?"
    python3 local_llm_client.py --config config.json --show-payload "hello"
    python3 local_llm_client.py --chat --history chat.json --save-history chat.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_HOST = "http://127.0.0.1:11434"
CHAT_ENDPOINT = "/api/chat"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_CONFIG_FILE = "config.json"
VALID_ROLES = ("system", "user", "assistant", "tool")


# --------------------------------------------------------------------------- #
# Request configuration
# --------------------------------------------------------------------------- #
@dataclass
class ChatConfig:
    """Every knob that shapes a request. Override via config file or CLI flags."""

    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    system: str | None = None
    system_role: str = "system"
    role: str = "user"
    temperature: float = 0.7
    top_p: float | None = None
    num_predict: int | None = None
    seed: int | None = None
    think: bool | None = None
    stream: bool = True
    timeout: float = 300.0
    history: list[dict] = field(default_factory=list)

    def endpoint(self) -> str:
        return self.host.rstrip("/") + CHAT_ENDPOINT

    def build_messages(self, prompt: str, history: list[dict] | None = None) -> list[dict]:
        """System message (if any) + prior turns + your new message."""
        messages: list[dict] = []
        if self.system:
            messages.append({"role": self.system_role, "content": self.system})
        messages.extend(self.history if history is None else history)
        messages.append({"role": self.role, "content": prompt})
        return messages

    def build_payload(self, messages: list[dict]) -> dict:
        options: dict = {"temperature": self.temperature}
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.seed is not None:
            options["seed"] = self.seed

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "options": options,
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload


# --------------------------------------------------------------------------- #
# Config / history files
# --------------------------------------------------------------------------- #
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from None


def load_config_file(path: Path) -> dict:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object of settings.")
    known = {f.name for f in fields(ChatConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        print(f"Warning: ignoring unknown keys in {path}: {', '.join(unknown)}", file=sys.stderr)
    return {key: value for key, value in raw.items() if key in known}


def normalize_messages(items, source: str) -> list[dict]:
    if not isinstance(items, list):
        raise SystemExit(f"{source} must be a JSON array of messages.")
    messages: list[dict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SystemExit(f"{source}: message #{index} must be a JSON object.")
        role, content = item.get("role"), item.get("content")
        if role is None or content is None:
            raise SystemExit(f"{source}: message #{index} needs both 'role' and 'content'.")
        if role not in VALID_ROLES:
            raise SystemExit(
                f"{source}: message #{index} has unknown role {role!r} "
                f"(expected one of: {', '.join(VALID_ROLES)})."
            )
        messages.append({"role": role, "content": content})
    return messages


def load_history_file(path: Path) -> list[dict]:
    raw = _read_json(path)
    if isinstance(raw, dict):  # allow {"messages": [...]}
        raw = raw.get("messages")
    return normalize_messages(raw, str(path))


def append_history_file(path: Path, messages: list[dict]) -> None:
    existing: list[dict] = []
    if path.exists():
        raw = _read_json(path)
        if isinstance(raw, dict):
            raw = raw.get("messages")
        existing = normalize_messages(raw, str(path))
    path.write_text(
        json.dumps(existing + messages, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #
def call_llm(config: ChatConfig, payload: dict) -> dict:
    """POST one chat request to Ollama and return the response + metrics."""
    request = urllib.request.Request(
        config.endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    if config.stream:
        return _consume_stream(request, start, config.timeout, config.host)
    return _consume_json(request, start, config.timeout, config.host)


def _consume_json(
    request: urllib.request.Request, start: float, timeout: float, host: str
) -> dict:
    """Non-streaming path: one JSON body, full latency, exact token counts."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            print(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ApiError(exc.code, body, _model_hint() if exc.code == 404 else None) from exc
    except urllib.error.URLError as exc:
        raise ApiError(
            None, f"Could not reach Ollama at {host}: {exc.reason}", _connection_hint()
        ) from exc

    message = data.get("message") or {}
    return _result(
        data=data,
        text=message.get("content") or "",
        thinking=message.get("thinking"),
        role=message.get("role"),
        finish_reason=data.get("done_reason"),
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
        latency=time.perf_counter() - start,
        ttft=None,
    )


def _nanos_to_seconds(value):
    """Ollama reports durations in nanoseconds."""
    return None if value is None else value / 1e9


def _result(
    *,
    data: dict,
    text: str,
    thinking: str | None,
    role: str | None,
    finish_reason: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency: float,
    ttft: float | None,
) -> dict:
    """Normalize an Ollama reply (streaming or not) into one flat result dict."""
    eval_duration = _nanos_to_seconds(data.get("eval_duration"))

    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "text": text,
        "thinking": thinking or None,
        "role": role,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency": latency,
        "ttft": ttft,
        "server_total": _nanos_to_seconds(data.get("total_duration")),
        "model_load": _nanos_to_seconds(data.get("load_duration")),
        "prompt_eval": _nanos_to_seconds(data.get("prompt_eval_duration")),
        "eval": eval_duration,
        # Model speed: output tokens over generation time reported by the server.
        "tokens_per_sec": (
            completion_tokens / eval_duration if completion_tokens and eval_duration else None
        ),
        # End-to-end speed: output tokens over wall-clock time.
        "wall_tokens_per_sec": (
            completion_tokens / latency if completion_tokens and latency else None
        ),
        "raw": data,
    }


def _consume_stream(
    request: urllib.request.Request, start: float, timeout: float, host: str
) -> dict:
    """Streaming path: prints deltas live and measures time-to-first-token.

    Ollama streams newline-delimited JSON; the final object carries the token
    counts and the server-side timings. Time-to-first-token is measured from the
    first token of any kind, so a reasoning model's thinking phase counts.
    """
    chunks: list[str] = []
    thinking_chunks: list[str] = []
    final: dict = {}
    ttft = None
    first_token = False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    raise ApiError(None, str(chunk["error"]), _model_hint())

                message = chunk.get("message") or {}

                thinking = message.get("thinking")
                if thinking and not thinking_chunks:
                    # A long thinking phase looks like a hang, so say something.
                    print("\n[thinking...]", file=sys.stderr, flush=True)

                content = message.get("content")
                if content or thinking:
                    if not first_token:
                        ttft = time.perf_counter() - start
                        first_token = True

                if content:
                    chunks.append(content)
                    sys.stdout.write(content)
                    sys.stdout.flush()

                if thinking:
                    thinking_chunks.append(thinking)

                if chunk.get("done"):
                    final = chunk
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ApiError(exc.code, body, _model_hint() if exc.code == 404 else None) from exc
    except urllib.error.URLError as exc:
        raise ApiError(
            None, f"Could not reach Ollama at {host}: {exc.reason}", _connection_hint()
        ) from exc

    if chunks:
        sys.stdout.write("\n")

    return _result(
        data=final,
        text="".join(chunks),
        thinking="".join(thinking_chunks),
        role="assistant",
        finish_reason=final.get("done_reason"),
        prompt_tokens=final.get("prompt_eval_count"),
        completion_tokens=final.get("eval_count"),
        latency=time.perf_counter() - start,
        ttft=ttft,
    )


class ApiError(Exception):
    def __init__(self, status: int | None, body: str, hint: str | None = None) -> None:
        self.status = status
        self.body = body
        self.hint = hint
        label = f"HTTP {status}" if status else "request failed"
        message = f"Ollama request failed ({label}): {body.strip()}"
        if hint:
            message = f"{message}\n\n{hint}"
        super().__init__(message)


def _connection_hint() -> str:
    return (
        "Is the server running?\n"
        "    ollama serve                      # start it and leave it running\n"
        "    curl http://127.0.0.1:11434       # should answer 'Ollama is running'"
    )


def _model_hint() -> str:
    return (
        "Check that the model tag exists:\n"
        "    ollama list                       # exact tags, e.g. qwen3:8b\n"
        "    ollama pull <model>"
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value, fmt: str = "{}", fallback: str = "n/a") -> str:
    return fallback if value is None else fmt.format(value)


def print_metrics(result: dict, config: ChatConfig) -> None:
    """One-line metrics summary, used by --chat so replies stay readable."""
    parts = [
        f"in {_fmt(result['prompt_tokens'])} tok",
        f"out {_fmt(result['completion_tokens'])} tok",
        f"latency {_fmt(result['latency'], '{:.2f} s')}",
    ]
    if result["ttft"] is not None:
        parts.append(f"ttft {_fmt(result['ttft'], '{:.2f} s')}")
    parts.append(f"eval {_fmt(result['tokens_per_sec'], '{:.1f} tok/s')}")
    print(f"  [{config.model} | " + " | ".join(parts) + "]")


def print_report(result: dict, config: ChatConfig, show_response: bool = True) -> None:
    line = "=" * 62
    print()
    if show_response:
        print(line)
        print("  RESPONSE")
        print(line)
        if result.get("thinking"):
            print("[thinking]")
            print(result["thinking"].strip())
            print()
        print(result["text"].strip() or "(empty response)")
        print()

    print(line)
    print("  METRICS")
    print(line)
    print(f"  host                : {config.endpoint()}")
    print(f"  model               : {config.model}")
    print(f"  roles               : system={config.system_role} prompt={config.role}")
    print(f"  temperature         : {config.temperature}")
    print(f"  finish reason       : {_fmt(result['finish_reason'])}")
    print(f"  input tokens        : {_fmt(result['prompt_tokens'], '{:,}')}")
    print(f"  output tokens       : {_fmt(result['completion_tokens'], '{:,}')}")
    print(f"  total tokens        : {_fmt(result['total_tokens'], '{:,}')}")
    print(f"  latency (wall clock): {_fmt(result['latency'], '{:.3f} s')}")
    if result["ttft"] is not None:
        print(f"  time to 1st token   : {_fmt(result['ttft'], '{:.3f} s')}")
    print(f"  server total        : {_fmt(result['server_total'], '{:.3f} s')}")
    print(f"  model load          : {_fmt(result['model_load'], '{:.3f} s')}")
    print(f"  prompt eval         : {_fmt(result['prompt_eval'], '{:.3f} s')}")
    print(f"  eval                : {_fmt(result['eval'], '{:.3f} s')}")
    print(f"  throughput (eval)   : {_fmt(result['tokens_per_sec'], '{:.1f} tok/s')}")
    print(f"  throughput (wall)   : {_fmt(result['wall_tokens_per_sec'], '{:.1f} tok/s')}")
    print(line)


def _jsonable(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "raw"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a local Ollama model and print token counts, response and latency.",
        epilog=(
            "Precedence: built-in defaults < config file < CLI flags. "
            f"A ./{DEFAULT_CONFIG_FILE} file is loaded automatically when present."
        ),
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send. Omit to read stdin or use --chat.")
    parser.add_argument(
        "--config", type=Path, help=f"JSON settings file (default: ./{DEFAULT_CONFIG_FILE} if it exists)."
    )
    parser.add_argument("--host", default=argparse.SUPPRESS, help=f"Ollama base URL (default: {DEFAULT_HOST}).")
    parser.add_argument("--model", default=argparse.SUPPRESS, help=f"Model tag (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "-s", "--system", default=argparse.SUPPRESS, help="System message sent at the start of the conversation."
    )
    parser.add_argument("--no-system", action="store_true", help="Send no system message, ignoring the config file.")
    parser.add_argument(
        "--system-role", default=argparse.SUPPRESS, help="Role used for the system message (default: system)."
    )
    parser.add_argument(
        "--role",
        default=argparse.SUPPRESS,
        help=f"Role for the prompt you send: {', '.join(VALID_ROLES)} (default: user).",
    )
    parser.add_argument("--history", type=Path, help="JSON file with prior messages to replay before your prompt.")
    parser.add_argument("--save-history", type=Path, help="Append this exchange to a JSON file.")
    parser.add_argument("--temperature", type=float, default=argparse.SUPPRESS, help="Sampling temperature (default: 0.7).")
    parser.add_argument("--top-p", type=float, default=argparse.SUPPRESS, help="Nucleus sampling cutoff.")
    parser.add_argument("--max-tokens", type=int, default=argparse.SUPPRESS, help="Cap generated tokens (Ollama's num_predict).")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="RNG seed for reproducible output.")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable/disable thinking mode on reasoning models (default: server default).",
    )
    parser.add_argument("--no-stream", action="store_true", help="Wait for the whole reply instead of streaming it.")
    parser.add_argument("--chat", action="store_true", help="Interactive loop that remembers history.")
    parser.add_argument("--timeout", type=float, default=argparse.SUPPRESS, help="Request timeout in seconds (default: 300).")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the result as JSON (implies --no-stream).")
    parser.add_argument("--show-payload", action="store_true", help="Print the JSON body that is POSTed.")
    parser.add_argument("--print-config", action="store_true", help="Print the effective settings and exit.")
    return parser


def build_config(args: argparse.Namespace) -> ChatConfig:
    """Defaults < config file < CLI flags."""
    data: dict = {}

    config_path = args.config
    if config_path is None:
        candidate = Path(__file__).with_name(DEFAULT_CONFIG_FILE)
        config_path = candidate if candidate.exists() else None
    if config_path is not None:
        data.update(load_config_file(config_path))

    for name in ("host", "model", "system", "system_role", "role", "temperature",
                 "top_p", "seed", "timeout", "think"):
        if hasattr(args, name):
            data[name] = getattr(args, name)
    if hasattr(args, "max_tokens"):
        data["num_predict"] = args.max_tokens
    if args.no_system:
        data["system"] = None
    if args.no_stream:
        data["stream"] = False

    config = ChatConfig(**data)

    for label, role in (("system_role", config.system_role), ("role", config.role)):
        if role not in VALID_ROLES:
            raise SystemExit(f"Invalid {label} {role!r}. Choose one of: {', '.join(VALID_ROLES)}.")

    if args.history:
        config.history = config.history + load_history_file(args.history)
    if args.as_json and config.stream:
        # Live tokens and a single JSON document can't share stdout.
        config.stream = False
    return config


def read_prompt(value: str | None) -> str | None:
    """Prompt from argv, piped stdin, or an interactive line. None means cancelled."""
    if value and value.strip():
        return value.strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    try:
        return input("Prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return None


def handle_command(
    line: str, config: ChatConfig, history: list[dict], save_path: Path | None
) -> str | None:
    """Apply an interactive /command. Returns 'exit' when quitting."""
    command, _, argument = line[1:].partition(" ")
    command, argument = command.lower(), argument.strip()

    if command in ("exit", "quit"):
        return "exit"
    if command == "reset":
        history.clear()
        print("history cleared")
    elif command == "system":
        config.system = argument or None
        print(f"system message = {config.system!r}")
    elif command == "system-role":
        config.system_role = argument
        print(f"system role = {config.system_role}")
    elif command == "role":
        if argument in VALID_ROLES:
            config.role = argument
            print(f"prompt role = {config.role}")
        else:
            print(f"role must be one of: {', '.join(VALID_ROLES)}")
    elif command in ("temp", "temperature"):
        try:
            config.temperature = float(argument)
        except ValueError:
            print("usage: /temp 0.7")
        else:
            print(f"temperature = {config.temperature}")
    elif command == "model":
        if argument:
            config.model = argument
        print(f"model = {config.model}")
    elif command == "history":
        print(json.dumps(history, indent=2, ensure_ascii=False) if history else "(empty)")
    elif command == "save":
        if save_path is None:
            print("start with --save-history <file> to enable /save")
        else:
            append_history_file(save_path, history)
            print(f"saved {len(history)} messages to {save_path}")
    else:
        print(f"unknown command: /{command}")
    return None


def chat_loop(config: ChatConfig, save_path: Path | None, as_json: bool) -> int:
    print(f"Chatting with {config.model} at {config.endpoint()}")
    print("Commands: /exit  /reset  /system <text>  /role <role>  /temp <value>  /model <tag>  /history  /save\n")

    history: list[dict] = list(config.history)

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if handle_command(line, config, history, save_path) == "exit":
                return 0
            continue

        payload = config.build_payload(config.build_messages(line, history))
        if not as_json:
            print("assistant> ", end="", flush=True)

        try:
            result = call_llm(config, payload)
        except ApiError as exc:
            print(f"\n{exc}", file=sys.stderr)
            continue

        if as_json:
            print(json.dumps(_jsonable(result), indent=2))
        else:
            if not config.stream:
                print(result["text"].strip())
            print_metrics(result, config)

        history.append({"role": config.role, "content": line})
        history.append({"role": "assistant", "content": result["text"]})
        if save_path:
            append_history_file(save_path, history[-2:])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(args)

    if args.print_config:
        print(json.dumps(asdict(config), indent=2, ensure_ascii=False))
        return 0

    if args.chat:
        return chat_loop(config, args.save_history, args.as_json)

    prompt = read_prompt(args.prompt)
    if prompt is None:
        return 130
    if not prompt:
        print("Error: empty prompt.", file=sys.stderr)
        return 2

    payload = config.build_payload(config.build_messages(prompt))
    payload["format"] = "json";
    if args.show_payload:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print()

    try:
        result = call_llm(config, payload)
    except ApiError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(_jsonable(result), indent=2))
    else:
        print_report(result, config)

    if args.save_history:
        append_history_file(
            args.save_history,
            [
                {"role": config.role, "content": prompt},
                {"role": "assistant", "content": result["text"]},
            ],
        )
        print(f"Saved 2 messages to {args.save_history}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
