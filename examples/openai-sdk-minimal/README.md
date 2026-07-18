# OpenAI SDK + fak: set one base URL

The smallest possible working repo for the **universal integration recipe**: point the
official [OpenAI SDK](https://github.com/openai/openai-python) at a `fak serve` gateway
and every tool call the model proposes crosses a governed **syscall boundary** — with
**one line of change** (`base_url`) and no other edits to your app.

`app.py` does two things:

1. Makes one ordinary `chat.completions` call through fak (your app, unchanged).
2. Asks fak to **adjudicate** two proposed tool calls *without running them*
   (`POST /v1/fak/adjudicate` — a pre-execution verdict only) and prints the verdict,
   so you can see the boundary decide.

## Run it

```bash
# 1. Start a local gateway in another terminal. With no --base-url it serves without
#    a real upstream, which is enough to demonstrate the syscall boundary:
fak serve --addr 127.0.0.1:8080

# 2. Run the client:
pip install -r requirements.txt
python app.py
```

Expected output (the model reply line appears only if you wired an upstream with
`fak serve --base-url …`; the verdicts always print):

```text
model reply: 'OK.'                # or a skip note if no upstream is configured

proposed tool call            ->  fak verdict
----------------------------------------------------
  read_file                 ->  verdict=ALLOW
  Bash                      ->  verdict=DENY reason=DEFAULT_DENY disposition=TERMINAL
```

`read_file` matches the built-in floor's `read_` allow-prefix; `Bash "git push …"` is
not on the allow-list, so the default-deny floor refuses it — **a verdict, not a crash**
(deny-as-value). The exact reason/disposition come straight from fak, so this output is
whatever your running gateway actually returns.

## The one line that matters

```python
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key=KEY)
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the whole integration
```

Everything else is standard OpenAI-SDK code. Swap `base_url` back to the provider and
your app is un-governed again — the boundary is exactly this one setting.

## Notes and honest fences

- **The verdict endpoint needs no model.** `/v1/fak/adjudicate` returns the
  pre-execution decision only (no dispatch, no engine), so it works even against a
  gateway with no upstream wired — that is why the demo is self-contained.
- **`chat.completions` needs an upstream.** Point fak at one with
  `fak serve --base-url http://host:11434/v1 --model <name>` (a local Ollama, vLLM,
  llama.cpp, or a hosted provider). The chat call is wrapped in a `try` so the verdict
  demo still runs without it.
- **Auth.** fak only requires a Bearer key if started with `--require-key-env FAK_GATEWAY_KEY`;
  the keyless local demo ignores the header. Set `FAK_GATEWAY_KEY` (and start the server
  with `--require-key-env`) to exercise the authenticated path.
- **Custom floor.** Pass `fak serve --policy policy.json` to govern which tools are
  allowed; validate it offline with `fak policy --check policy.json`. See the
  [compose-ollama recipe](../compose-ollama/README.md) for a policy + full stack.

See [`docs/cli-reference.md`](../../docs/cli-reference.md) for the gateway surface and
[`docs/fak/deployment-guide.md`](../../docs/fak/deployment-guide.md) for production wiring.
