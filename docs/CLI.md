# fak CLI reference

This page summarizes the commands a new user needs. It was written from the
`--help` output of a local build (`go build -o fak ./cmd/fak`, Windows amd64,
commit `a244ed05879`). When the help text and this page disagree, the help
text wins. Run `fak help` for the overview, `fak help --verbs` for the full
verb index, and `fak <verb> --help` for any verb.

## Verbs at a glance

| Verb | What it does |
|---|---|
| `fak up` | Turnkey local model provisioner (macOS, Linux, Windows): picks a model tier for your memory, auto-selects CUDA/Vulkan/Metal/CPU, serves it on `127.0.0.1:8080`, opens a chat prompt. |
| `fak run <model> [prompt]` | Runs an in-kernel model: a chat REPL, or one-shot with a prompt. Also replays recorded traces with `--trace`. |
| `fak chat` | Minimal chat client against a running `fak serve` gateway. |
| `fak serve` | The OpenAI-compatible model gateway, with a live dashboard at `http://127.0.0.1:8080/`. |
| `fak model` | Resolve, download, list, and inspect model references. |
| `fak manage` | Wraps an agent harness (for example `claude`) so every tool call passes the kernel. |
| `fak doctor` | Diagnoses serve readiness, TLS trust, and answer quality. |
| `fak version` | Prints the build version, commit, and Go toolchain. |

There is no `fak gateway`, `fak models`, or `fak install` verb. The gateway is
`fak serve`, model listing is `fak model ls`, and installation is
`install.sh` (see [INSTALL.md](../INSTALL.md)).

## fak up

```sh
fak up [--model <tier>] [--context <tokens>] [--addr 127.0.0.1:8080] [--headless]
```

- `--model <tier>` overrides the auto-selected tier (for example `27B` or `70B`).
- `--headless` runs the server in the foreground without the chat REPL.
- `--dry-run` prints the plan without starting anything. `--json` emits the plan as JSON.
- `--memory-gib <gib>` overrides the detected system memory.
- `--kv-precision f32|q8_0` picks the KV cache tier. `q8_0` fits about twice the context.

`fak up status`, `fak up off`, and `fak up on` manage the macOS launch agent.
They are macOS-only.

## fak run

```sh
fak run <model> [prompt]
fak run --trace FILE [--engine ID] [--policy FILE]
```

- `<model>` is an alias (see `fak model ls`, for example `smollm2`), an
  `hf://owner/repo/model.gguf` URI, or a local `.gguf` path.
- With no prompt, you get an interactive REPL. Press Ctrl-D to exit.
- `--backend auto|cpu|<name>` picks the compute backend. An unavailable named backend fails loudly.
- `--trace` replays a recorded JSON tool-call trace offline. `fak replay --trace FILE` is the same.

## fak chat

```sh
fak chat --base-url http://127.0.0.1:8080/v1 --model <model>
```

- `--base-url` points at any OpenAI-compatible endpoint. Empty uses the provider environment or the selected gateway.
- `--json` emits a machine-readable receipt in headless mode.
- `--max-turns N` caps the model turns for one human turn (default 10).
- `--code-workspace <dir>` sets the root for the built-in code tools (default: current directory).

## fak serve

```sh
fak serve [--addr 127.0.0.1:8080] [--base-url URL] [--model NAME] [--guard]
fak serve help <category>
```

`fak serve` help is grouped by task: `start`, `model`, `cache`, `context`,
`policy`, `session`, `native`, `observe`, `fleet`, and `all`. Use
`fak serve help all` for every flag.

Without `--guard`, requests pass through without tool-call checks. `--guard`
(or `FAK_GUARD=on`) turns on the tool guard. Pass `--policy FILE` to load a
capability floor and `--require-key-env VAR` to require a bearer token.

## fak model

```sh
fak model ls                 # aliases and which are cached
fak model pull <ref>         # download into the local cache
fak model load <ref>         # resolve to a cached file path
fak model info <ref>         # family, format, context, weights, memory floor
```

Other subcommands (`fit`, `plan`, `quantize`, `preflight`, and the acceptance
and canary gates) are for maintainers and are listed by `fak model --help`.

## fak manage

```sh
fak manage [flags] [--] <agent command...>
fak manage --policy my-floor.json -- claude
```

Runs an agent harness through the kernel, so each tool call is checked against
the capability floor before it runs. Operator subcommands: `fak manage allow
<tool>`, `fak manage disable`, and `fak manage policy explain|diff`.

## fak doctor

```sh
fak doctor serve [--model-bytes N] [--headroom 0.15] [--json]
fak doctor trust [--probe=false] [--host HOST:PORT] [--json]
fak doctor [--text - | --file PATH] [--json]
```

- `doctor serve` checks the CPU SIMD tier, whether the target model fits free
  memory with headroom, and NUMA layout. It exits 1 on an unready row.
- `doctor trust` checks which trust store fak validates upstream TLS with, and
  whether a corporate CA bundle is needed. It is read-only.
- `doctor` with `--text` or `--file` checks a text for repetition and runaway length.

## fak version

```sh
fak version
```

Prints the version, the build commit (with `+uncommitted` on a dirty tree),
the commit date, and the Go toolchain and platform.

## Environment

- `FAK_GUARD=on` turns on the tool guard for `fak serve`.
- `FAK_VERSION=<x.y.z>` pins the installer to a release tag.
- `FAK_BACKEND` sets the default compute backend; `--backend` overrides it.
- `FAK_UP_KV_PRECISION` sets the default KV precision for `fak up`.
