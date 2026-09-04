# Getting started with fak

*[`README.md`](README.md) is the single canonical front door for the product overview and evaluation. This page owns one concrete job: guiding you through the setup sequence from an initial binary to a verified installation.*

> **TL;DR:** Follow the three sequential steps below: verify offline with Tier 0, configure the gateway proxy with Tier 1, and explore in-kernel local serving with Tier 2.

```bash
# Quick install (macOS / Linux):
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

---

## The setup sequence

The setup workflow advances across three tiers:

1. Step 1: Offline verification (Tier 0). Prove the tool-call boundary and capability floor with zero downloads (~2 min).
2. Step 2: Gateway setup (Tier 1). Place `fak serve` in front of an existing model API or local engine like Ollama.
3. Step 3: Local serve (Tier 2). Run in-kernel execution and local GGUF models on your machine.

---

## 0. Prerequisites

- Go 1.26+: `fak/go.mod` specifies Go 1.26. With Go's default `GOTOOLCHAIN=auto`, the toolchain fetches automatically on first build. Check with `go version`.
- Tier 0 and Tier 2 (synthetic): require only the `fak` binary. No GPU, API key, or network access needed.
- Tier 1 (gateway): requires any OpenAI-compatible model server (e.g. Ollama or vLLM) or a cloud provider key.
- Tier 2 (real weights): requires Python 3.10+ for model conversion; see [in-kernel model documentation](docs/fak/in-kernel-model.md).

---

## 1. Get the binary

Choose the method that fits your environment:

### Prebuilt binary (recommended for adopters)

Download prebuilt binaries from the [latest release](https://github.com/anthony-chaudhary/fak/releases/latest):

| Method | Command |
|---|---|
| One-liner (Linux/macOS) | `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh \| sh` |
| Manual download | Download `fak_<version>_<os>_<arch>.tar.gz` (`.zip` on Windows), extract, and move `fak` to your `PATH`. |
| Docker | `docker build -t fak https://github.com/anthony-chaudhary/fak.git` |

Supported release targets: linux (amd64/arm64), darwin (amd64/arm64), and windows (amd64).

To verify SLSA build provenance:

```bash
gh attestation verify fak_<version>_<os>_<arch>.tar.gz --repo anthony-chaudhary/fak
gh attestation verify SHA256SUMS --repo anthony-chaudhary/fak
```

### Install with Go

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

### Build from source (contributors)

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak
./fak help
```

> **Windows note:** Build natively with `go build -o fak.exe ./cmd/fak`. Windows host Application-Control policies may block compiled test binaries; run test suites inside WSL with `./test.ps1`. The binary itself runs natively without issues.

---

## 2. Step 1: Offline verification (Tier 0 — zero downloads, ~2 min)

Run the deterministic offline proof from any directory:

```bash
./fak agent --offline
```

### Expected output

The run exercises a deterministic mock planner against an injection fixture. You should see output matching this capture:

```text
metric                        now(base)          fak
--------------------------   ----------   ----------
injection in context                YES           no
destructive op executed             YES           no
task completed (booked)             YES          YES

HEADLINE
  poisoned result blocked   : YES
  destructive op prevented  : YES

report written: agent-report.json
```

The proof succeeds when `task completed (booked)` is `YES` in both columns, while `poisoned result blocked` and `destructive op prevented` are `YES` under `fak`.

### Capability floor preflight checks

Verify structural refusal independent of any model:

```bash
# Blocked: create_user is not on the default allow-list
./fak preflight --tool create_user --args '{"_positional":["alice"]}'
# -> verdict=DENY reason=DEFAULT_DENY by=monitor

# Allowed: get_user_details is on the allow-list
./fak preflight --tool get_user_details --args '{}'
# -> verdict=ALLOW
```

Inspect and validate the policy manifest:

```bash
./fak policy --dump > floor.json
./fak policy --check floor.json
```

See [`POLICY.md`](POLICY.md) for manifest schema details.

### Guard posture: default-open convenience with fail-closed safety

When you front an agent with `fak guard`, the runtime defaults to **`default_open`** posture:
- **Developer convenience**: Unlisted benign tools (such as new CLI tools, custom test runners, or MCP server integrations) are admitted automatically without requiring manual edits to an allowlist manifest.
- **Fail-closed safety**: Catastrophic commands are blocked fail-closed before execution via the **Dangerous Gotchas catalog** (blocking recursive deletions like `rm -rf`, raw disk operations, pipe-to-shell RCE, fork bombs, local privilege escalation, and infrastructure destruction), alongside compiled-in FROZEN invariants (SSRF egress protection and out-of-tree write escapes).

For compliance, enterprise, or zero-trust environments requiring a strict default-deny capability floor where every permitted tool must be explicitly enumerated:

```bash
# Run with strict default-deny (unlisted tools hit DEFAULT_DENY)
fak guard --posture fail_closed -- claude

# Or set via environment variable
export FAK_GUARD_POSTURE=fail_closed
```

You can inspect or dump the strict fail-closed capability floor anytime with `fak guard --dump-strict-policy`. See [`POLICY.md`](POLICY.md) for complete posture rules, gotchas catalog details, and carveouts.

---

## 3. Step 2: Gateway setup (Tier 1 — practical serving path)

`fak serve` acts as an OpenAI-compatible gateway that intercepts and adjudicates tool calls. Upstream models generate proposals; `fak` checks them against your capability floor before your client executes them.

### Example with Ollama

Start Ollama and pull a test model:

```bash
ollama serve &
ollama pull qwen2.5:1.5b
```

Launch `fak serve` pointing to Ollama:

```bash
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b
```

Verify service health from another terminal:

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"qwen2.5:1.5b","ok":true}
```

Point any OpenAI-compatible client or agent at `http://127.0.0.1:8080/v1`.

### Key gateway routes

| Route | Function |
|---|---|
| `POST /v1/chat/completions` | OpenAI wire proxy with tool call adjudication |
| `POST /v1/messages` | Anthropic wire proxy with tool call adjudication |
| `GET /healthz` | Unauthenticated liveness check |
| `GET /v1/models` | Advertises served model identifier |
| `POST /v1/fak/syscall` | Direct execution of one adjudicated tool call |
| `POST /v1/fak/adjudicate` | Evaluate tool proposal without dispatching |
| `GET /metrics` | Prometheus metrics exposition |

### Production hardening

Harden the gateway by enforcing a policy manifest and API authentication:

```bash
./fak serve --addr 0.0.0.0:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b \
  --policy floor.json \
  --require-key-env FAK_TOKEN
```

Clients can pass the key via `Authorization: Bearer <token>` or Anthropic's `x-api-key: <token>` header.

---

## 4. Step 3: Local serve (Tier 2 — in-kernel model execution)

For fully local, private inference without external servers, `fak` can run pure-Go model forward passes and local GGUF models directly in-kernel.

Run an agent wrapped with a local GGUF model:

```bash
fak guard --gguf qwen2.5:7b -- claude
```

The GGUF model loads directly into the kernel address space, adjudicates every tool call locally, and requires no external API keys or network requests.

Kernel developers working on native forward passes and KV-cache mechanics should consult the dedicated [in-kernel model guide](docs/fak/in-kernel-model.md).

---

## Troubleshooting

| Symptom | Cause and resolution |
|---|---|
| `go: go.mod requires go >= 1.26` | Upgrade Go to 1.26+ or enable `GOTOOLCHAIN=auto`. |
| `Application Control policy has blocked this file` (Windows) | OS restriction on fresh test executables. Run test suites in WSL with `./test.ps1`. |
| `no such file testdata/...` | Run commands from the repository root, or supply an absolute path to `--trace`. |
| `address already in use` | Another service is using port 8080. Choose a different port using `--addr`. |

For runtime connection issues, tool refusals, or upstream HTTP errors, consult the [first-run troubleshooting guide](docs/adoption/troubleshooting-first-run.md).

---

## Next steps

- Connect your existing agent: see the [integration guides](docs/integrations/README.md) for recipes covering Claude Code, Codex, Cursor, and custom harnesses.
- Find your specific task: browse the [task router](START-HERE.md) to navigate directly to authoritative documentation for your workflow.
- Walk through the full tutorial: follow the [guided first session](docs/fak/tutorial.md) for detailed explanations and captured terminal outputs.
