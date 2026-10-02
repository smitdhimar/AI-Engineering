# basic-api-call

Two zero-dependency clients that report the metrics worth monitoring on every call:

- `deepseek_client.py` — hosted **DeepSeek** API
- `local_llm_client.py` — local **Ollama** server (`POST /api/chat`)

| Metric | DeepSeek source | Ollama source |
| --- | --- | --- |
| **Input token count** | `usage.prompt_tokens` | `prompt_eval_count` |
| **Output token count** | `usage.completion_tokens` | `eval_count` |
| **Response** | `choices[0].message.content` | `message.content` |
| **Latency** | wall-clock timer | wall-clock timer + server timings |

Zero dependencies — standard library only (Python 3.9+).

## DeepSeek client (`deepseek_client.py`)

### Setup

```bash
export DEEPSEEK_API_KEY="sk-..."
```

### Run

```bash
# one-shot prompt
python3 deepseek_client.py "Explain what an API is in one sentence."

# streaming (prints tokens as they arrive, plus time-to-first-token)
python3 deepseek_client.py --stream "Write a haiku about tokens."

# no args -> interactive: asks for key (if unset) then prompt
python3 deepseek_client.py
```

### Options

| Flag | Purpose |
| --- | --- |
| `--api-key` | Override the `DEEPSEEK_API_KEY` env var |
| `--model` | Default `deepseek-chat` (try `deepseek-reasoner` for reasoning) |
| `--system` | System prompt |
| `--temperature` | Sampling temperature (default 1.0) |
| `--max-tokens` | Cap output tokens |
| `--stream` | Stream response + measure TTFT |
| `--json` | Emit metrics as JSON (handy for piping/logging) |
| `--timeout` | Request timeout in seconds (default 120) |

### Example output

```
==============================================================
  RESPONSE
==============================================================
An API (Application Programming Interface) is a defined way for
one program to request data or actions from another.

==============================================================
  METRICS
==============================================================
  model              : deepseek-chat
  finish reason      : stop
  input tokens       : 18
  output tokens      : 37
  total tokens       : 55
  prompt cache       : hit 0 / miss 18
  latency            : 2.914 s
  throughput         : 12.7 tok/s
==============================================================
```

## Local Ollama client (`local_llm_client.py`)

Chats with `http://127.0.0.1:11434/api/chat` (default model `qwen3.5:9b`) and prints the response plus input/output tokens and latency.

```bash
ollama serve                      # once, in its own terminal
ollama list                       # confirm the exact model tag

python3 local_llm_client.py "What is an API?"
python3 local_llm_client.py --system "Answer in one word." --temperature 0.2 "Capital of France?"
python3 local_llm_client.py --chat --save-history chat.json   # interactive, remembers context
python3 local_llm_client.py --show-payload "hello"            # inspect what gets POSTed
python3 local_llm_client.py --print-config                    # see the effective settings
```

### Settings

Precedence is **built-in defaults < config file < CLI flags**. A `config.json` next to the script is loaded automatically, so the easiest way to change things permanently is to edit it:

```json
{
  "model": "qwen3.5:9b",
  "system": "You are a terse assistant. Answer in one sentence.",
  "system_role": "system",
  "role": "user",
  "temperature": 0.7,
  "think": false,
  "stream": true,
  "history": []
}
```

| Flag | Purpose |
| --- | --- |
| `--config` | Settings file (default: `./config.json` if present) |
| `--host` / `--model` | Server URL / model tag |
| `-s, --system` / `--no-system` | System message (or send none) |
| `--system-role` | Role used for the system message (default `system`) |
| `--role` | Role for the message you send: `system`, `user`, `assistant`, `tool` |
| `--history FILE` | JSON messages to replay before your prompt |
| `--save-history FILE` | Append the exchange to a JSON file |
| `--temperature`, `--top-p`, `--max-tokens`, `--seed` | Sampling options (`--max-tokens` maps to Ollama's `num_predict`) |
| `--think` / `--no-think` | Thinking mode on reasoning models |
| `--chat` | Interactive loop with in-memory history |
| `--no-stream` | Wait for the whole reply instead of streaming |
| `--json` | Metrics as JSON (implies `--no-stream`) |

In `--chat` mode: `/exit`, `/reset`, `/system <text>`, `/role <role>`, `/temp <value>`, `/model <tag>`, `/history`, `/save`.

`--history` / `--save-history` files are a JSON array of messages (a `{"messages": [...]}` wrapper also works):

```json
[
  { "role": "user", "content": "My name is Smit." },
  { "role": "assistant", "content": "Nice to meet you, Smit." }
]
```

### Example output

```
Red is a primary color.

==============================================================
  METRICS
==============================================================
  host                : http://127.0.0.1:11434/api/chat
  model               : qwen3.5:9b
  roles               : system=system prompt=user
  temperature         : 0.7
  finish reason       : stop
  input tokens        : 33
  output tokens       : 7
  total tokens        : 40
  latency (wall clock): 0.638 s
  time to 1st token   : 0.301 s
  server total        : 0.625 s
  model load          : 0.005 s
  prompt eval         : 0.279 s
  eval                : 0.272 s
  throughput (eval)   : 25.7 tok/s
  throughput (wall)   : 11.0 tok/s
==============================================================
```

### Notes

- **`output tokens` counts thinking tokens too.** With thinking enabled a one-sentence answer can report ~885 output tokens (and take ~44 s); set `"think": false` or pass `--no-think` for quick, cheap calls.
- While a reasoning model thinks, the client prints `[thinking...]` on stderr so a long pause doesn't look like a hang; the thinking text itself is shown on stdout at the end.
- `latency (wall clock)` is measured client-side; `server total`, `model load`, `prompt eval` and `eval` come from Ollama's nanosecond timings. The first call after a cold start includes model load time.
- `time to 1st token` is only measured in streaming mode (the default).
- If the connection fails you get `ollama serve` / `ollama list` hints instead of a traceback.

## Notes

- DeepSeek endpoint: `https://api.deepseek.com/chat/completions` (OpenAI-compatible schema).
- Both clients use `--json` output for logging input/output tokens over time; `raw` payloads are stripped.
