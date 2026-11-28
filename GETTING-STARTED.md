# Getting started with fak

Start with the [local model setup](#local-model-setup) for your hardware. If you
already have a model server such as Ollama, go straight to [gateway setup](#gateway-setup).

## 0. Prerequisites

- For source builds, Go 1.26+: `fak/go.mod` specifies Go 1.26. With Go's default `GOTOOLCHAIN=auto`, the toolchain fetches automatically on first build. Check with `go version`.
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
| Manual download | Select the Metal or Vulkan GPU archive in [INSTALL.md](INSTALL.md); preserve Vulkan `spirv/` beside `fak`. |
| NVIDIA CUDA | `docker run --rm --gpus all ghcr.io/anthony-chaudhary/fak:cuda-latest version` |

The installer defaults to Apple Silicon Metal or Linux amd64 Vulkan, including
AMD Strix Halo. The Vulkan archive requires GNU glibc 2.39+ (Ubuntu 24.04+),
the system Vulkan loader, and a compatible GPU driver. CPU builds are secondary
references (`--variant cpu`). The [v0.59.0 release](https://github.com/anthony-chaudhary/fak/releases/tag/v0.59.0)
includes both GPU archives and their SHA-256 checksums. Missing assets produce an error with
source-build or explicit CPU-reference instructions. Packaging does not establish
model/GPU qualification.

To verify SLSA build provenance:

```bash
gh attestation verify fak_<version>_<os>_<arch>.tar.gz --repo anthony-chaudhary/fak
gh attestation verify SHA256SUMS --repo anthony-chaudhary/fak
```

### Install with Go

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

This builds the latest release from source and includes Inkling-2-Mini. On Apple Silicon
the Metal build requires CGo and the Xcode command line tools. Leave
`CGO_ENABLED` at its default of 1. For a prebuilt binary, the installer
downloads GitHub's Latest release. Set `FAK_VERSION=0.59.0` to pin it.
For NVIDIA CUDA, use the published `ghcr.io/anthony-chaudhary/fak:0.59.0-cuda`
image or build from source below.

### Build from source (contributors)

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak
./fak help
```

### Inkling-2-Mini on NVIDIA CUDA

The plain `go build` above runs Inkling-2 on the host CPU. For the CUDA device path,
build the binary with the CUDA backend, then run the pinned GGUF:

```bash
go run ./cmd/fak-dev build-cuda binary --arch sm_120f --cuda-home /opt/cuda ./cmd/fak ./fak
./fak run /path/to/Inkling-2-Mini-UD-Q3_K_XL.gguf
```

- `--arch` is the card's compute capability (`sm_120f` for RTX 50-series). `--cuda-home` is the CUDA Toolkit root.
- `make cuda-build-sm120f` builds only the kernel library (`libfakcuda.a`); it does not produce a runnable `./fak` with the CUDA backend. Use the `binary` command above.
- On WSL, if the link fails with `cannot find -lcuda`, set `LIBRARY_PATH=<cuda>/lib64/stubs` (or `lib/stubs`) for the build.
- The binary needs the CUDA 12 runtime sonames (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) on `LD_LIBRARY_PATH`.
- Backend choice: with `--backend` omitted, a CUDA build selects CUDA for Inkling-2 when the binary's cubins cover the card. The run prints `fak run: <arch> backend=cuda layers=<N> passes=<N>` to stderr (`cmd/fak/run_inkling2.go`), where `<arch>` is the GGUF's `general.architecture` (`inkling2` or `inkling2_loop`), not a fixed `inkling2` string. A CPU fallback prints `backend=cpu`. Check that line before trusting any throughput number. Pass `--backend cuda` to fail loud instead of falling back to CPU.

> **Windows note:** Build natively with `go build -o fak.exe ./cmd/fak`. Microsoft Defender Antivirus real-time ML detections repeatedly quarantine freshly compiled test binaries; run test suites inside WSL with `./test.ps1`. The binary itself runs natively without issues.

---

## 2. Check a tool policy

### Capability floor preflight checks

Verify structural refusal independent of any model. The built-in floor is
default-open, so unlisted tools are allowed until you deny them. Write a
small policy that denies one tool, then check it:

```bash
cat > floor.json <<'EOF'
{
  "version": "fak-policy/v1",
  "posture": "default_open",
  "allow": ["get_user_details"],
  "deny": {"refund_payment": "POLICY_BLOCK"}
}
EOF
./fak policy --check floor.json

# Blocked by the manifest's explicit deny
./fak preflight --tool refund_payment --args '{"amount":10}' --policy floor.json
# -> verdict=DENY reason=POLICY_BLOCK by=monitor

# Allowed: no rule matches, so the default-open posture admits it
./fak preflight --tool get_user_details --args '{}' --policy floor.json
# -> verdict=ALLOW reason=NONE by=monitor
```

Pass `--policy` to test a manifest; without it, preflight uses the built-in
floor, which admits most tools by default.

> **Windows shell note:** The single-quoted `--args '{...}'` works unchanged in git-bash and PowerShell 7. Windows PowerShell 5.1 strips the inner JSON quotes at the native-process boundary; use `--args '{\"_positional\":[\"alice\"]}'` there. On cmd.exe, use `--args "{""_positional"":[""alice""]}"`.

Inspect and validate the policy manifest:

```bash
./fak policy --dump > floor.json
./fak policy --check floor.json
```

> **Windows PowerShell 5.1 note:** Output redirection (`>`) in Windows PowerShell 5.1 writes UTF-16LE with a BOM, and `Set-Content -Encoding utf8` writes UTF-8 with a BOM. Write `floor.json` as UTF-8 without a BOM (or use PowerShell 7 / `git-bash`) before running `fak policy --check floor.json`.

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

Under `fail_closed`, an unlisted tool is refused with
`verdict=DENY reason=DEFAULT_DENY` instead of being admitted.

You can inspect or dump the strict fail-closed capability floor anytime with `fak guard --dump-strict-policy`. See [`POLICY.md`](POLICY.md) for complete posture rules, gotchas catalog details, and carveouts.

---

<a id="gateway-setup"></a>

## 3. Gateway setup (Tier 1 — practical serving path)

`fak serve` acts as an OpenAI-compatible gateway. **By default it is a raw inference
pipe** — requests pass through with zero interception processing. Pass `--guard` (or
set `FAK_GUARD=on`) to arm the tool guard: upstream models generate proposals; `fak`
checks them against your capability floor before your client executes them, and inbound
tool results are screened (secret/PII masking, injection quarantine). The startup
messages state the resolved posture.

### Example with Ollama

Start Ollama and pull a model. Replace `<model>` below with any tag from `ollama list`:

```bash
ollama serve &
ollama pull <model>
```

Launch `fak serve` pointing to Ollama:

```bash
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model <model>
```

Verify service health from another terminal:

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"<model>","ok":true}
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
./fak serve --addr 0.0.0.0:8080 --base-url http://localhost:11434/v1 --model <model> \
  --policy floor.json \
  --require-key-env FAK_TOKEN
```

Clients can pass the key via `Authorization: Bearer <token>` or Anthropic's `x-api-key: <token>` header.

---

<a id="local-model-setup"></a>

## 4. Local serve (Tier 2 — in-kernel model execution)

[Install the build for your platform](INSTALL.md#local-model-setup). In a terminal, run:

```sh
fak up
```

On macOS, Linux, and Windows, `fak up` picks a model tier for your system
memory, auto-selects the available accelerator (Metal on Apple Silicon, CUDA
or Vulkan on Linux/Windows, or CPU), and downloads the model on the first run.
At `you>`, ask a coding question. Leave the terminal open to use its chat
completions endpoint at `http://127.0.0.1:8080/v1`. For a lightweight one-shot
alternative, use `fak run smollm2` from the [quick start](README.md#try-fak).

To let fak choose a larger model based on available memory, run `fak up`.
In v0.56.0 the automatic 70B download can return HTTP 401. Select the public
27B model with `fak up --model 27B` (about 17 GB).

Connect OpenCode or Pi to the chat completions endpoint from a second terminal
with `fak opencode` or `fak pi`. Claude Code and Codex use the Messages and
Responses endpoints. Start `fak up` (or `fak serve --gguf /path/to/model.gguf`)
for those clients, then run `fak claude` or `fak codex`. See the
[agent guides](docs/integrations/README.md).

For fully local, private inference without external servers, `fak` can run pure-Go model forward passes and local GGUF models directly in-kernel on your silicon:

### A. Instant interactive REPL (`fak run`)

Run an interactive chat session directly in your terminal (no background server needed):

```bash
# Pull and chat with Inkling-2-Mini (resumable download, auto-cached)
fak run inkling2-mini

# Or run a one-shot query with SmolLM2
fak run smollm2 "Explain KV cache in one line"
```

### B. Apple Silicon Metal GPU acceleration (`fak serve` + `fak chat`)

On Apple Silicon, the installer selects the native Metal GPU archive by default.
The backend must still be qualified on the actual machine and supported model.
CPU reference archives cannot provide Metal acceleration. Ways to obtain a GPU build:

1. **`sh install.sh`** downloads `fak_<VERSION>_darwin_arm64_metal.tar.gz` on Apple Silicon. The v0.59.0 archive includes the Metal backend. For another version without that archive, use option 2.
2. **Build from source with `CGO_ENABLED=1`** — see the [Mac local models guide](docs/fak/mac-local-models.md) for the build and qualification steps.
3. **NVIDIA GPUs** ship as the separate ghcr `-cuda` container image instead — see the [deployment guide](docs/fak/deployment-guide.md).

```bash
# Terminal 1: start the server (Metal auto-selected only when the build supports it and the device qualifies)
fak up

# Terminal 2: chat with it (auto-connects to local server and detects model)
fak chat
```

Or connect explicitly if using custom ports or model routing:
```bash
fak chat --base-url http://127.0.0.1:8080/v1 --model inkling2-mini
```

For complete Apple Silicon unified memory sizing and multi-agent setup, see the [Mac local models guide](docs/fak/mac-local-models.md).

### C. Guard existing agents with local models (`fak guard`)

Run an agent wrapped with a local GGUF model:

```bash
fak guard --gguf /path/to/Inkling-2-Mini-UD-Q3_K_XL.gguf -- claude
```

The GGUF model loads directly into the kernel address space, adjudicates every tool call locally, and requires no external API keys or network requests.

Kernel developers working on native forward passes and KV-cache mechanics should consult the dedicated [in-kernel model guide](docs/fak/in-kernel-model.md).

---

## Troubleshooting

| Symptom | Cause and resolution |
|---|---|
| `go: go.mod requires go >= 1.26` | Upgrade Go to 1.26+ or enable `GOTOOLCHAIN=auto`. |
| `Application Control policy has blocked this file` (Windows) | Microsoft Defender Antivirus real-time ML quarantines fresh test executables. Run test suites in WSL with `./test.ps1`. |
| `no such file testdata/...` | Run commands from the repository root, or supply an absolute path to `--trace`. |
| `address already in use` | Another service is using port 8080. Choose a different port using `--addr`. |

For runtime connection issues, tool refusals, or upstream HTTP errors, consult the [first-run troubleshooting guide](docs/adoption/troubleshooting-first-run.md).

---

## Next steps

- Connect your existing agent: see the [integration guides](docs/integrations/README.md) for recipes covering Claude Code, Codex, Cursor, and custom harnesses.
- Run parallel subagents: see the [subagents guide](docs/subagents-guide.md) for orchestrating concurrent multi-agent fanout with `fak up` and zero-cold-start cache reuse.
- Find your specific task: browse the [task router](START-HERE.md) to navigate directly to authoritative documentation for your workflow.
- Walk through the full tutorial: follow the [guided first session](docs/fak/tutorial.md) for detailed explanations and captured terminal outputs.
