# Launch kit: fak with Inkling-2-Mini (2026-11-24)

This is the current launch copy. It replaces the older drafts in this directory
([show-hn.md](show-hn.md), [x-thread.md](x-thread.md),
[reddit-localllama.md](reddit-localllama.md)), which lead with an earlier story.

Draft only. Posting is done by a person. Every number below appears in the
receipts table at the end; do not add a number that is not there.

## Positioning

fak is the runtime around your coding agent. It serves a local model and
refuses the tool calls you did not allow. With Inkling-2-Mini on one 16 GB
RTX 5080, fak decodes at 118.1 tok/s against stock llama.cpp at 81.0 in the
same window (1.46x, ABBA x2), and level with the realugbun llama.cpp fork.

Speed is the hook, not the pitch. The decode result answers the usual objection
to the guard ("your engine is slow"). Against the fork it is parity, not a lead,
and the copy must show the prefill loss next to the headline number.

Three facts go in every post:

- Inkling-2-Mini support is on `main`, not in a published release. v0.57.0 was
  tagged but never published as Latest and cannot be installed through the Go
  module proxy; the latest installable release is v0.56.0, which predates
  Inkling. Build from a clone of `main`.
- The comparison rates are in-process decode benches. The one number measured
  through the user command is `fak run` decode (about 116 tok/s). There is no
  served-endpoint rate.
- The plain `go build` binary has no CUDA backend. The RTX 5080 numbers need
  the separate `-tags cuda` build.

### Install

CPU, and Metal on Apple Silicon (cgo on, the Go default there). Needs Go 1.26
or later and git:

```sh
git clone --depth 1 https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak      # fak.exe on Windows
./fak run <path>/Inkling-2-Mini-UD-Q3_K_XL.gguf
```

Do not point people at `go install` for this launch. The v0.57.0 tag cannot be
fetched through the Go module proxy (its source tree is over the proxy's
500 MiB module zip cap), so `go install ...@latest` still resolves to v0.56.0,
and `...@main` downloads a module zip of about 116 MB before it compiles. The
release installer (`install.sh`, Homebrew) installs the latest published
release, v0.56.0, which does not contain Inkling either.

CUDA (the RTX 5080 numbers). The CUDA backend compiles only under
`-tags cuda` with cgo, linked against kernels built by nvcc
(`internal/compute/cuda.go`). Building it needs a CUDA toolkit; see
[docs/cuda-dev.md](../cuda-dev.md). From the same clone on a CUDA host:

```sh
make cuda-build    # nvcc -> libfakcuda.a, then go build -tags cuda
make cuda-build-sm120   # RTX 50-series such as the 5080; the default arch is sm_89
```

The 118.1 and 119.9 tok/s numbers are the in-process decode bench, which runs
as a `-tags cuda` test binary. The exact commands, with `--arch sm_120` for the
RTX 5080, are under "Reproduce" in
[the resident-decode receipt](../benchmarks/receipts/inkling2-cuda-resident-rtx5080-20261124/RESULTS.md).

Fresh-user path, measured on one RTX 5080 from a clean home directory: the
documented from-source CUDA build (clone, `make cuda-build-sm120`, link) took
about 8.5 minutes. `fak run` then produced its first token in about 13 s, for
both a fresh and a repeated process, and decoded at about 116 tok/s (116.9 and
116.1). That 13 s includes process start, loading the GGUF into VRAM and
prefill; the file was already in the OS page cache, so a disk-cold first run is
not measured. This is the first number measured through the user command rather
than the in-process bench. It is still not a served-endpoint rate.

## Hacker News

**Title:**

Show HN: fak: Inkling-2-Mini at 118.1 tok/s vs stock llama.cpp 81.0 on one 5080

**First comment:**

> fak is a single Go binary that serves local models to coding agents and checks
> each tool call before it runs. This week we brought up Inkling-2-Mini
> (Apache-2.0, looped MoE) natively on CUDA and Metal. On a 16 GB RTX 5080 with
> the unsloth UD-Q3_K_XL GGUF, decode went from 6.32 tok/s to fork level through
> resident weights, one replayed CUDA graph per token, and fused ops. In one
> window on one card (ABBA x2, same file), fak decoded at 118.1 tok/s and stock
> llama.cpp at 81.0, so 1.46x. In a separate window the realugbun llama.cpp fork
> got 119.5 against our 119.9: parity with the fork, not a win. Where we lose:
> prefill. Opt-in batched prefill reaches pp512 862.7, about a third of
> llama.cpp's 2362 (measured on different boxes), and we trail the fork at
> around 2K tokens of context. These are in-process benches, not served rates.
> Through the user command, a fresh from-source CUDA build took about 8.5
> minutes, and `fak run` gave a first token in about 13 s (model load included)
> and about 116 tok/s decode. Inkling is not in a published release yet
> (`go install ...@latest` resolves to v0.56.0, which predates it), so build
> from a clone: `git clone --depth 1 https://github.com/anthony-chaudhary/fak.git`,
> then `cd fak && go build -o fak ./cmd/fak` and `./fak run <path>.gguf` gives
> the CPU and Apple Silicon Metal path. The CUDA numbers need the `-tags cuda`
> build (`make cuda-build`, or `make cuda-build-sm120` for RTX 50-series, on a
> CUDA host; see docs/cuda-dev.md). Each number links to a receipt.

## r/LocalLLaMA

**Title:**

Inkling-2-Mini on a 16 GB 5080: 118.1 tok/s decode in fak vs 81.0 in stock llama.cpp, same window (1.46x, ABBA x2); level with the realugbun fork (receipts, plus where we lose)

**Body:**

> Same card and GGUF. In a separate window the realugbun fork got 119.5
> against our 119.9, so that one is parity. From a fresh clone, the CUDA build
> took about 8.5 minutes, and `fak run` gave a first token in about 13 s (model
> load included) and about 116 tok/s decode, measured through the command
> rather than the bench. M3 Pro Metal runs level with Splash (40.3 vs 37.5).
> Opt-in speculative decode reached 151.7 tok/s on one node; it is off by
> default because it is not bit-exact. We lose on prefill (about a third of
> llama.cpp, measured on different boxes) and at around 2K tokens of depth.
> The comparison numbers are in-process
> benches; none is a served rate. Inkling is not in a published release. Build from a clone of main (`git clone --depth 1`, then
> `go build -o fak ./cmd/fak`) for CPU and Metal; the 5080 numbers need the
> `-tags cuda` build (docs/cuda-dev.md).
> fak's actual job is guarding the tool calls of the
> coding agent you point at it; the engine work is so that the local model is
> not the slow part.

Attach both charts: [decode comparison](assets/inkling2-decode-comparison.svg)
and [decode climb](assets/inkling2-decode-climb.svg).

## X thread

1. fak checks your coding agent's tool calls and serves it a local model.
   Inkling-2-Mini now runs natively: 118.1 tok/s vs stock llama.cpp's 81.0 in
   one window on one RTX 5080 (1.46x, ABBA x2). In-process decode bench, not a
   served rate. Receipts linked.
2. Against the realugbun llama.cpp fork it is parity, not a lead: 119.9 vs
   119.5, measured in a separate window.
3. The climb: 6.32 → 82.5 → 101.6 → 119.9. The 101.6 step was
   measured on a second box with the same card model.
   [chart: [inkling2-decode-climb.svg](assets/inkling2-decode-climb.svg)]
4. Where we lose: prefill is about a third of llama.cpp (measured on different
   boxes), and we trail the fork at around 2K tokens of context.
   [chart: [inkling2-decode-comparison.svg](assets/inkling2-decode-comparison.svg)]
5. Fresh user, through `fak run` rather than the bench: from-source CUDA build
   about 8.5 min, first token about 13 s (model load included), about 116 tok/s
   decode.
6. M3 Pro Metal: 40.3 tok/s, level with Splash.
7. `fak claude` / `fak opencode` / `fak pi`: same agent, every tool call
   checked. Inkling is not in a published release: clone main,
   `go build -o fak ./cmd/fak`. CUDA is the separate `-tags cuda` build
   (docs/cuda-dev.md).

## fak versus alternatives

| Alternative | Where fak is better | Where fak loses today |
|---|---|---|
| llama.cpp (stock and fork) | 1.46x stock decode on Inkling-2-Mini (118.1 vs 81.0, same window, one RTX 5080). Guard and router built in. | Ties the fork, does not beat it. Prefill is about a third of llama.cpp (862.7 vs 2362 pp512, different boxes). Trails the fork at 1946 tokens of depth (89.0 vs 113.8). |
| Ollama, LM Studio | Fork-level decode on this model. Tool-call guard. No opaque model store. | They install in one click; fak needs a build from `main` for Inkling. No Vulkan or AMD backend. |
| vLLM, SGLang | Runs Inkling-2-Mini on one 16 GB card. | Not a matched comparison. They are built for multi-GPU serving; fak has no measured served rate on this model, so there is no matched number to cite either way. |
| LiteLLM, Portkey | Wraps the agent harness, not only the endpoint; failover headers are readable. | They lead on per-key limits and PII handling. Call it parity. |
| Cloud-only harnesses | Local by default, binds 127.0.0.1; the router can still reach a cloud model. | Local model quality: Qwen3.8-27B leads the dev-work target match (44 vs 28–34 of 240 scored items in a 300-item set). |

Other losses to state plainly: the CUDA backend is a separate `-tags cuda`
build that needs a CUDA toolkit (Metal is in the default cgo build on Apple
Silicon), `go install` cannot fetch the v0.57.0 tag through the Go module
proxy (source tree over the 500 MiB zip cap), and the from-source CUDA build
takes about 8.5 minutes on a fresh clone.

## Safe claims

| Say | Say with caveat |
|---|---|
| 118.1 vs 81.0 tok/s, 1.46x stock llama.cpp in one window (ABBA x2) | 119.9 vs 119.5, parity with the realugbun fork (separate window) |
| `fak run` first token about 13 s, about 116 tok/s decode | First token includes model load; file was page-cached, not disk-cold |
| Every number has a receipt | Kernel changes kept output matching on the bench prompt (one long-context oracle case is order-sensitive) |
| Tokenizer matches Hugging Face on 46/46 corpus lines | Speculative decode 151.7 tok/s (opt-in, not bit-exact, one node) |
| Prefill trails llama.cpp | M3 Pro level with Splash (both use draft verification) |
| Build from a clone of `main` | 35-task suite 89.5% vs 87.6% (model quality; both served by llama.cpp) |
| The guard refuses with fixed reason codes | "Default-deny" only when naming strict mode; the shipped default allows unlisted tools |

Do not say: "faster than llama.cpp", "fastest Inkling runtime", "bit-identical"
without qualification, any served-endpoint rate or served time to first token,
151.7 as the default speed, or a `fak run inkling2-mini` alias.

## Receipts

Ancestry check, run 2026-11-25 before this revision:

```sh
git fetch origin
git merge-base --is-ancestor b64405de760e2f13a62594ff8d029f4599fcdeb4 origin/main   # exit 0
git merge-base --is-ancestor cc6dca0b47a9c6a6c9166fb94efd819b1f380ee1 origin/main   # exit 0
```

Result: `b64405de76` (the build that measured 119.9) and `cc6dca0b47` (the
build that measured 118.1 and the fresh-user run) are ancestors of public
`origin/main` at `f8f6c6d993`. "On main" below means the measured build is an
ancestor of `origin/main`.

Benchmark campaign records are listed in
[docs/benchmarks/published/](../benchmarks/published/). Where a row says "held
privately", the raw logs are in the maintainer archive because they record
local paths; the published campaign record summarises the run.

| Claim | Number | Hardware class | Runs | On main | Public evidence |
|---|---|---|---|---|---|
| fak vs stock llama.cpp decode, same window, same card and file | 118.1 vs 81.0 tok/s (1.46x; 1.48x without the first fak run, 113.6) | RTX 5080 16 GB | ABBA x2: 4 fak runs of 127 steps, 4 `llama-bench -r 3` | yes (`cc6dca0b47`) | receipt held privately (same-window launch evidence, 2026-11-25) |
| Fresh-user path: from-source CUDA build, then `fak run` | build 507 s (about 8.5 min); first token about 13.1 s; decode 116.9 / 116.1 tok/s | RTX 5080 16 GB | 1 build, 2 `fak run` processes (fresh and repeated) | yes (`cc6dca0b47`) | receipt held privately (same window); first token includes model load, file page-cached |
| fak decode, Inkling-2-Mini UD-Q3_K_XL (separate window, fork parity) | 119.9 tok/s | RTX 5080 16 GB | 2 (ABBA B arm, 127 steps each) | yes (`b64405de76`) | campaign record in [published/](../benchmarks/published/) (fork parity, 2026-11-24); raw logs held privately |
| realugbun llama.cpp fork decode, same card and file | 119.5 tok/s | RTX 5080 16 GB | 2 windows of `llama-bench -r 3` | n/a (external) | same campaign record |
| Stock llama.cpp decode, same card and file (superseded by the same-window row) | 81.0 tok/s | RTX 5080 16 GB | `llama-bench -r 3`, two days earlier | n/a (external) | campaign record in [published/](../benchmarks/published/) (OSS baseline, 2026-11-22) |
| Decode at 1946 tokens of depth, fak vs fork | 89.0 vs 113.8 tok/s | RTX 5080 16 GB | 1 each | yes | fork parity campaign record |
| Decode climb start | 6.32 tok/s | RTX 5080 16 GB | 4 | yes | campaign record in [published/](../benchmarks/published/) (GPU decode climb, 2026-11-23) |
| Resident decode | 82.5 tok/s | RTX 5080 16 GB | ABBA | yes | GPU decode climb campaign record |
| Fused-op decode, second box | 101.6 tok/s | RTX 5080 16 GB (second box) | ABBA | yes | fused climb campaign record; compare within that box only |
| Opt-in batched prefill, pp512 | 862.7 tok/s | RTX 5080 16 GB | 2 (ABBA) | yes | batched prefill campaign record |
| llama.cpp prefill, pp512 | 2362 tok/s | RTX 5080 16 GB (different box) | `llama-bench -r 3` | n/a (external) | OSS baseline campaign record |
| Opt-in speculative decode | 151.7 tok/s | RTX 5080 16 GB | 2 | yes | spec decode campaign record; one node, not bit-exact |
| M3 Pro Metal decode vs Splash 1.7.2 | 40.3 vs 37.5 tok/s (97.5-107.4% across prompts) | M3 Pro 36 GB | 4 processes, 6 measured requests per shape | yes (`1f31533f2`) | [m3pro-inkling2-mini-metal/RESULTS.md](../benchmarks/published/m3pro-inkling2-mini-metal/RESULTS.md) |
| Tokenizer parity with Hugging Face | 46/46 corpus lines | CPU | 1 corpus pass | yes | receipt held privately |
| Dev-work target match, Qwen3.8-27B vs the other models compared (model quality, not a fak engine rate) | 44 vs 28–34 of 240 scored items (300-item set) | all models served by llama.cpp at 4-bit | see receipt | n/a (model quality) | receipt held privately |
| 35-task suite (model quality, not a fak engine rate) | 89.5% vs 87.6% | both models served by llama.cpp | 35 tasks x 3 reps | n/a (model quality) | receipt held privately |

Charts: [decode comparison](assets/inkling2-decode-comparison.svg),
[decode climb](assets/inkling2-decode-climb.svg).
