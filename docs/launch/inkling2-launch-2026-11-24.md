# Launch kit: fak with Inkling-2-Mini (2026-11-24)

This is the current launch copy. It replaces the older drafts in this directory
([show-hn.md](show-hn.md), [x-thread.md](x-thread.md),
[reddit-localllama.md](reddit-localllama.md)), which lead with an earlier story.

Draft only. Posting is done by a person. Every number below appears in the
receipts table at the end; do not add a number that is not there.

## Positioning

fak is the runtime around your coding agent. It serves a local model and
refuses the tool calls you did not allow. With Inkling-2-Mini it now does that
at the decode speed of the realugbun llama.cpp fork on a 16 GB card.

Speed is the hook, not the pitch. The decode result answers the usual objection
to the guard ("your engine is slow"). It is parity with a fork, not a lead, and
the copy must show the prefill loss next to the headline number.

Three facts go in every post:

- Inkling-2-Mini support is on `main`, not in a tagged release. v0.57.0 does not
  contain it. Install from `main`.
- All rates are in-process decode benches, not served-endpoint rates.
- The plain `go install` binary has no CUDA backend. The RTX 5080 numbers need
  the separate `-tags cuda` build.

### Install

CPU, and Metal on Apple Silicon (cgo on, the Go default there). This is the
install the M3 Pro receipt used:

```sh
go install github.com/anthony-chaudhary/fak/cmd/fak@main
fak run <path>/Inkling-2-Mini-UD-Q3_K_XL.gguf
```

CUDA (the RTX 5080 numbers). The CUDA backend compiles only under
`-tags cuda` with cgo, linked against kernels built by nvcc
(`internal/compute/cuda.go`). Building it needs a CUDA toolkit; see
[docs/cuda-dev.md](../cuda-dev.md). From a clone on a CUDA host:

```sh
make cuda-build    # nvcc -> libfakcuda.a, then go build -tags cuda
```

The 119.9 tok/s number is the in-process decode bench, which runs as a
`-tags cuda` test binary. The exact commands, with `--arch sm_120` for the
RTX 5080, are under "Reproduce" in
[the resident-decode receipt](../benchmarks/receipts/inkling2-cuda-resident-rtx5080-20261124/RESULTS.md).

## Hacker News

**Title:**

Show HN: fak – a guard and local runtime for coding agents, now running Inkling-2-Mini at 119.9 tok/s on a 5080

**First comment:**

> fak is a single Go binary that serves local models to coding agents and checks
> each tool call before it runs. This week we brought up Inkling-2-Mini
> (Apache-2.0, looped MoE) natively on CUDA and Metal. On a 16 GB RTX 5080 with
> the unsloth UD-Q3_K_XL GGUF, decode went from 6.32 to 119.9 tok/s through
> resident weights, one replayed CUDA graph per token, and fused ops. Stock
> llama.cpp gets 81.0 on that card and file and the realugbun fork gets 119.5,
> so this is parity with the realugbun llama.cpp fork, not a win. Where we lose:
> prefill. Opt-in batched prefill reaches pp512 862.7, about a third of
> llama.cpp, and we trail the fork at around 2K tokens of context. These are
> in-process benches, not served rates. Inkling is not in v0.57.0 yet:
> `go install github.com/anthony-chaudhary/fak/cmd/fak@main`, then
> `fak run <path>.gguf`, gives the CPU and Apple Silicon Metal path. The CUDA
> numbers need the `-tags cuda` build (`make cuda-build` on a CUDA host; see
> docs/cuda-dev.md). Each number links to a receipt.

## r/LocalLLaMA

**Title:**

Inkling-2-Mini on a 16 GB 5080: 119.9 tok/s decode in fak, 81.0 in stock llama.cpp, 119.5 in the realugbun fork (receipts, plus where we lose)

**Body:**

> Same card and GGUF. M3 Pro Metal runs level with Splash (40.3 vs 37.5).
> Opt-in speculative decode reached 151.7 tok/s on one node; it is off by
> default because it is not bit-exact. We lose on prefill (about a third of
> llama.cpp) and at around 2K tokens of depth. All numbers are in-process
> benches. Build from main; the 5080 numbers need the `-tags cuda` build
> (docs/cuda-dev.md), while plain `go install ...@main` covers CPU and Metal.
> fak's actual job is guarding the tool calls of the
> coding agent you point at it; the engine work is so that the local model is
> not the slow part.

Attach both charts: [decode comparison](assets/inkling2-decode-comparison.svg)
and [decode climb](assets/inkling2-decode-climb.svg).

## X thread

1. fak guards your coding agent's tool calls and serves it a local model.
   Inkling-2-Mini now runs natively: 119.9 tok/s decode on one RTX 5080, level
   with the realugbun llama.cpp fork (119.5), above stock (81.0). Receipts linked.
2. The climb in three days: 6.32 → 82.5 → 101.6 → 119.9. The 101.6 step was
   measured on a second box with the same card model.
   [chart: [inkling2-decode-climb.svg](assets/inkling2-decode-climb.svg)]
3. Where we lose: prefill is about a third of llama.cpp, and we trail the fork
   at around 2K tokens of context.
   [chart: [inkling2-decode-comparison.svg](assets/inkling2-decode-comparison.svg)]
4. M3 Pro Metal: 40.3 tok/s, level with Splash.
5. `fak claude` / `fak opencode` / `fak pi`: same agent, every tool call
   checked. Install from main; CUDA is the separate `-tags cuda` build
   (docs/cuda-dev.md).

## fak versus alternatives

| Alternative | Where fak is better | Where fak loses today |
|---|---|---|
| llama.cpp (stock and fork) | About 1.48x stock decode on Inkling-2-Mini (cross-day, different harness). Guard and router built in. | Ties the fork, does not beat it. Prefill is about a third of llama.cpp (862.7 vs 2362 pp512, different boxes). Trails the fork at 1946 tokens of depth (89.0 vs 113.8). |
| Ollama, LM Studio | Fork-level decode on this model. Tool-call guard. No opaque model store. | They install in one click; fak needs a build from `main` for Inkling. No Vulkan or AMD backend. |
| vLLM, SGLang | Runs Inkling-2-Mini on one 16 GB card. | Not a matched comparison. They are built for multi-GPU serving; fak has no measured served rate, and trails vLLM on 27B models and on H200. |
| LiteLLM, Portkey | Wraps the agent harness, not only the endpoint; failover headers are readable. | They lead on per-key limits and PII handling. Call it parity. |
| Cloud-only harnesses | Local by default, binds 127.0.0.1; the router can still reach a cloud model. | Local model quality: Qwen3.8-27B leads the dev-work target match (44 vs 28–34 of 240 scored items in a 300-item set). |

Other losses to state plainly: the CUDA backend is a separate `-tags cuda`
build that needs a CUDA toolkit (Metal is in the default cgo build on Apple
Silicon), and no
fresh-user receipt yet covers `go install @main` through to the first token.

## Safe claims

| Say | Say with caveat |
|---|---|
| 119.9 tok/s decode, parity with the realugbun fork | About 1.48x stock llama.cpp (cross-day, different harness) |
| Every number has a receipt | Kernel changes kept output matching on the bench prompt (one long-context oracle case is order-sensitive) |
| Tokenizer matches Hugging Face on 46/46 corpus lines | Speculative decode 151.7 tok/s (opt-in, not bit-exact, one node) |
| Prefill trails llama.cpp | M3 Pro level with Splash (both use draft verification) |
| Install from `main` | 35-task suite 89.5% vs 87.6% (model quality; both served by llama.cpp) |
| The guard refuses with fixed reason codes | "Default-deny" only when naming strict mode; the shipped default allows unlisted tools |

Do not say: "faster than llama.cpp", "fastest Inkling runtime", "bit-identical"
without qualification, any served or time-to-first-token rate, 151.7 as the
default speed, or a `fak run inkling2-mini` alias.

## Receipts

Ancestry check, run 2026-11-24 before writing this file:

```sh
git fetch origin
git merge-base --is-ancestor b64405de760e2f13a62594ff8d029f4599fcdeb4 origin/main   # exit 0
```

Result: `b64405de76` (the build that measured 119.9) is an ancestor of public
`origin/main` at `00544707db`. "On main" below means the measured build is an
ancestor of `origin/main`.

Benchmark campaign records are listed in
[docs/benchmarks/published/](../benchmarks/published/). Where a row says "held
privately", the raw logs are in the maintainer archive because they record
local paths; the published campaign record summarises the run.

| Claim | Number | Hardware class | Runs | On main | Public evidence |
|---|---|---|---|---|---|
| fak decode, Inkling-2-Mini UD-Q3_K_XL | 119.9 tok/s | RTX 5080 16 GB | 2 (ABBA B arm, 127 steps each) | yes (`b64405de76`) | campaign record in [published/](../benchmarks/published/) (fork parity, 2026-11-24); raw logs held privately |
| realugbun llama.cpp fork decode, same card and file | 119.5 tok/s | RTX 5080 16 GB | 2 windows of `llama-bench -r 3` | n/a (external) | same campaign record |
| Stock llama.cpp decode, same card and file | 81.0 tok/s | RTX 5080 16 GB | `llama-bench -r 3`, two days earlier | n/a (external) | campaign record in [published/](../benchmarks/published/) (OSS baseline, 2026-11-22) |
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
