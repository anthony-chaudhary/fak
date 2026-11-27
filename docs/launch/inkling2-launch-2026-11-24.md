# Launch kit: fak with Inkling-2-Mini (2026-11-24, refreshed for v0.59.0)

This is the current launch copy. It replaces the older drafts in this directory
([show-hn.md](show-hn.md), [x-thread.md](x-thread.md),
[reddit-localllama.md](reddit-localllama.md)), which lead with an earlier story.

Draft only. Posting is done by a person. Every number below appears in the
receipts table at the end; do not add a number that is not there.

## Positioning

fak is the runtime around your coding agent. It serves a local model and
refuses the tool calls you did not allow. With Inkling-2-Mini on one 16 GB
RTX 5080, fak decodes at 118.1 tok/s in its in-process decode bench, against
81.0 for stock llama.cpp in the same window (1.46x, fak in-process bench vs
llama-bench, ABBA x2), and level with the realugbun llama.cpp fork at 119.9
against 119.5 in a separate window.

The second beat is opt-in CUDA speculative decode: 171.5 tok/s at depth 2 and
180.9 at depth 4, against 119.7 plain on the same receipt (1.43x and 1.51x). On
CUDA it is off by default and not bit-exact on every prompt, so it is never the
headline. Per prompt, plain-to-speculative speedups run from 1.09x to 1.35x.

Speed is the hook, not the pitch. The decode result answers the usual objection
to the guard ("your engine is slow"). Against the fork it is parity, not a lead,
and the losses sit next to the headline: decode at 1946 tokens of depth (89.0
against the fork's 113.8). Prefill is now a matched win: batched prefill is on
by default, and over HTTP on one card in one window fak's pp512 prompt rate is
3880.4 tok/s against the fork's 3092.2 (1.25x). In-process it is 4598.2. Both
prefill rows were measured at `97ccb4a7d9`, before W4A4 grouped prefill became
the default (`27355b23dd6`).

Three facts go in every post:

- Inkling-2-Mini ships in `v0.59.0` (the current GitHub release, with Apple
  Silicon Metal and Linux amd64 Vulkan archives). Both `install.sh` and
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` install it. The
  release has no CUDA binary, so the RTX 5080 numbers still need a clone and
  `make cuda-build-sm120` (or the CUDA Docker image).
- The comparison rates are in-process decode benches. The one number measured
  through the user command is `fak run` decode (116.5 tok/s, mean of two runs).
  There is no served-endpoint rate.
- The plain `go build` binary has no CUDA backend. The RTX 5080 numbers need
  the separate `-tags cuda` build.

### Install

CPU, and Metal on Apple Silicon. Needs Go 1.26 or later (or use the `v0.59.0`
release archive via `install.sh`; see [INSTALL.md](../../INSTALL.md)):

```sh
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # v0.59.0
fak run inkling-2-mini         # short name for the pinned inkling2-mini GGUF
fak run <path>/Inkling-2-Mini-UD-Q3_K_XL.gguf   # or a local file
```

Or build from a clone of `main`: `git clone --depth 1
https://github.com/anthony-chaudhary/fak.git`, then `go build -o fak ./cmd/fak`
(`fak.exe` on Windows).

The short names `inkling` and `inkling-2-mini` chain to the `inkling2-mini`
catalog entry (commit `4e47f88205`, an ancestor of `v0.59.0`), so all three name
the same pinned GGUF. On Apple Silicon, `v0.59.0` runs Inkling-2-Mini on fak's
pure `internal/metalgemm` resident Metal path (with speculative decode on by
default via the auto draft); the earlier vendored Splash port has been retired.

CUDA (the RTX 5080 numbers). The CUDA backend compiles only under
`-tags cuda` with cgo, linked against kernels built by nvcc
(`internal/compute/cuda.go`). Building it needs a CUDA toolkit; see
[docs/cuda-dev.md](../cuda-dev.md). From the same clone on a CUDA host:

```sh
make cuda-build    # nvcc -> libfakcuda.a, then go build -tags cuda
make cuda-build-sm120   # RTX 50-series such as the 5080; the default arch is sm_89
```

The 118.1 and 119.9 tok/s numbers are the in-process decode bench, which runs
as a `-tags cuda` test binary. The resident-decode campaign record is in
[the resident-decode receipt](../benchmarks/published/aegis-inkling2-resident-cuda/RESULTS.md).

Fresh-user path, measured on one RTX 5080 from a clean home directory: the
documented from-source CUDA build (clone, `make cuda-build-sm120`, link) took
507 s, about eight and a half minutes. `fak run` then produced its first token
in about 13.1 s, for both a fresh and a repeated process, and decoded at 116.5
tok/s, the mean of the two processes (each run is in the receipts table). That
13.1 s includes process start, loading the GGUF into VRAM and prefill; the file
was already in the OS page cache, so a disk-cold first run is not measured. This
is the first number measured through the user command rather than the in-process
bench. It is still not a served-endpoint rate.

## Hacker News

**Title:**

Show HN: fak: Inkling-2-Mini at 118.1 tok/s vs stock llama.cpp 81.0 on one 5080

**First comment:**

> fak is a single Go binary that serves local models to coding agents and checks
> each tool call before it runs. This week we brought up Inkling-2-Mini
> (Apache-2.0, looped MoE) natively on CUDA and Metal. On a 16 GB RTX 5080 with
> the unsloth UD-Q3_K_XL GGUF, decode went from a first native build at 6.32
> tok/s, through resident weights at 82.5 and a fused step at 101.6 on a second
> box, to fork level at 119.9. In one window on one card (ABBA x2, same file),
> fak's in-process bench decoded at 118.1 tok/s and llama-bench on stock
> llama.cpp at 81.0, so 1.46x. In a separate window the realugbun llama.cpp fork
> got 119.5 against our 119.9: parity with the fork, not a win. Opt-in CUDA
> speculative decode reaches 171.5 tok/s at depth 2 and 180.9 at depth 4, against
> 119.7 plain on the same receipt (1.43x and 1.51x). On CUDA it is off by default
> and not bit-exact on every prompt. Prefill is a matched win: with batched
> prefill on by default, the pp512 prompt rate over HTTP is 3880.4 tok/s against
> the fork's 3092.2 on the same card in the same window (1.25x; prompt rate, not
> end-to-end latency; measured at `97ccb4a7d9`, before W4A4 grouped prefill
> became the default). Where we lose: decode at 1946 tokens of depth, 89.0
> against the fork's 113.8. Batch-of-one is a loss too (65.3 against 119.9
> single-stream). The batched path wins at width: 783.7 tok/s aggregate across
> 24 streams (opt-in f16 KV, single run). These are in-process benches, not
> served rates. Through the user command, a fresh from-source CUDA build took
> 507 s, and `fak run` gave a first token in about 13.1 s (model load included)
> and 116.5 tok/s decode. Inkling ships in `v0.59.0`: `install.sh` or
> `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` installs it, and
> `fak run inkling-2-mini` gives the CPU and Apple Silicon Metal path. The
> release has no CUDA binary, so the CUDA numbers need a clone and the `-tags
> cuda` build (`make cuda-build`, or `make cuda-build-sm120` for RTX 50-series,
> on a CUDA host; see docs/cuda-dev.md). Each number links to a receipt.

## r/LocalLLaMA

**Title:**

Inkling-2-Mini on a 16 GB 5080: 118.1 tok/s decode in fak vs 81.0 in stock llama.cpp, same window (1.46x, ABBA x2); level with the realugbun fork (receipts, plus where we lose)

**Body:**

> Same card and GGUF. In a separate window the realugbun fork got 119.5
> against our 119.9, so that one is parity. Opt-in CUDA speculative decode
> reaches 171.5 tok/s at depth 2 and 180.9 at depth 4, against 119.7 plain on
> the same receipt (1.43x and 1.51x). On CUDA it is off by default because it is
> not bit-exact on every prompt; per-prompt speedups run from 1.09x to 1.35x.
> From a fresh clone, the CUDA build took 507 s, and `fak run` gave a first
> token in about 13.1 s (model load included) and 116.5 tok/s decode, measured
> through the command rather than the bench. Prefill over HTTP beats the fork on
> the same card in the same window: pp512 prompt rate 3880.4 against 3092.2
> tok/s (1.25x, prompt rate only; measured at `97ccb4a7d9`, before W4A4 grouped
> prefill became the default). Where we lose: decode at 1946 tokens of depth
> (89.0 against 113.8); batch-of-one (65.3 against 119.9 single-stream, 0.54x);
> and the opt-in PDL flag, which is slower on this card (110.7 against 117.2,
> 0.94x), so it stays off. The batched path wins at width: 783.7 tok/s aggregate
> across 24 streams (opt-in f16 KV, single run), which is not a per-stream rate.
> Context depth: at 8192 tokens the measured ms/token went from 21.47 to 13.30
> (1.61x); at 19 tokens it is within noise (8.63 against 8.55). The comparison
> numbers are in-process benches; none is a served rate. Inkling ships in
> `v0.59.0`: `install.sh` or
> `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` for CPU and Metal;
> the 5080 numbers need a clone and the `-tags cuda` build (docs/cuda-dev.md).
> fak's actual job is guarding the tool calls of the coding agent you point at
> it; the engine work is so that the local model is not the slow part.

Attach the charts: [decode comparison](assets/inkling2-decode-comparison.svg),
[decode climb](assets/inkling2-decode-climb.svg),
[batched serving](assets/inkling2-batched-serving.svg), and
[context depth](assets/inkling2-context-depth.svg).

## X thread

1. fak checks your coding agent's tool calls and serves it a local model.
   Inkling-2-Mini now runs natively: fak's in-process bench at 118.1 vs
   llama-bench on stock llama.cpp at 81.0 in one window on one RTX 5080 (1.46x,
   ABBA x2). Not a served rate. Receipts linked.
2. Against the realugbun llama.cpp fork it is parity, not a lead: 119.9 vs
   119.5, measured in a separate window.
3. Second beat: opt-in CUDA speculative decode reaches 171.5 tok/s at depth 2
   and 180.9 at depth 4, against 119.7 plain (1.43x and 1.51x). Not bit-exact on
   every prompt, off by default on CUDA.
   [chart: [inkling2-decode-climb.svg](assets/inkling2-decode-climb.svg)]
4. Prefill is a matched win: pp512 prompt rate over HTTP 3880.4 vs the fork's
   3092.2, same card and window (1.25x; measured at 97ccb4a7d9). Where we lose:
   decode at 1946 tokens of depth is 89.0 against the fork's 113.8; batch of one
   is 65.3 against 119.9.
   [chart: [inkling2-decode-comparison.svg](assets/inkling2-decode-comparison.svg)]
5. Batched: 783.7 tok/s aggregate across 24 streams (opt-in f16 KV, single run;
   not per stream); cohort prefill end to end goes from 137.7 to 372.1 (2.70x) on
   16 HTTP streams.
   [chart: [inkling2-batched-serving.svg](assets/inkling2-batched-serving.svg)]
6. Fresh user, through `fak run` rather than the bench: from-source CUDA build
   507 s, first token about 13.1 s, 116.5 tok/s decode.
7. Apple Silicon Metal: `v0.59.0` runs Inkling-2-Mini on pure
   `internal/metalgemm` resident Metal (pure resident receipts pending; the
   earlier 2026-11-24 M3 Pro record used the retired vendored Splash port and is
   not claimed as pure fak-native Metal).
8. `fak claude` / `fak opencode` / `fak pi`: same agent, every tool call
   checked. Inkling ships in `v0.59.0`: `install.sh` or `go install ...@latest`.
   CUDA is a clone and the `-tags cuda` build (docs/cuda-dev.md).

## fak versus alternatives

| Alternative | Where fak is better | Where fak loses today |
|---|---|---|
| llama.cpp (stock and fork) | Stock decode on Inkling-2-Mini, same window on one RTX 5080: fak in-process bench 118.1 vs llama-bench 81.0 tok/s (1.46x). Prefill over HTTP beats the fork in a matched window: pp512 prompt rate 3880.4 vs 3092.2 tok/s (1.25x; measured at 97ccb4a7d9). Guard and router built in. | Ties the fork on decode, does not beat it (119.9 vs 119.5). Trails the fork at 1946 tokens of depth (89.0 vs 113.8). |
| Ollama, LM Studio | Fork-level decode on this model. Tool-call guard. No opaque model store. | They install in one click; fak ships Metal and Vulkan archives in `v0.59.0`, while CUDA still needs a `-tags cuda` source build or the CUDA container. |
| vLLM, SGLang | Runs Inkling-2-Mini on one 16 GB card. | Not a matched comparison. They are built for multi-GPU serving; fak has no measured served rate on this model, so there is no matched number to cite either way. |
| LiteLLM, Portkey | Wraps the agent harness, not only the endpoint; failover headers are readable. | They lead on per-key limits and PII handling. Call it parity. |
| Cloud-only harnesses | Local by default, binds 127.0.0.1; the router can still reach a cloud model. | Local model quality: Qwen3.8-27B leads the dev-work target match (44 vs 28–34 of 240 scored items in a 300-item set). |

Other losses to state plainly: the CUDA backend is a separate `-tags cuda`
build that needs a CUDA toolkit (Metal is in the default cgo build on Apple
Silicon and in the `v0.59.0` macOS release archive), `v0.59.0` has no prebuilt
CUDA binary asset (only the CUDA container or a from-source build), the
from-source CUDA build took 507 s on a fresh clone, opt-in PDL is slower on
sm_120 (110.7 against 117.2 tok/s, 0.94x), and speculative decode with PDL is
slower too (137.1 against 146.9, 0.93x), so PDL stays off by default.

## Safe claims

| Say | Say with caveat |
|---|---|
| fak in-process bench 118.1 vs llama-bench 81.0 tok/s, 1.46x stock llama.cpp in one window (ABBA x2) | 119.9 vs 119.5, parity with the realugbun fork (separate window) |
| Opt-in CUDA speculative decode: 171.5 tok/s at depth 2, 180.9 at depth 4, against 119.7 plain (1.43x, 1.51x) | Opt-in, off by default on CUDA; not bit-exact on every prompt (capitals prompt); per-prompt speedups 1.09x to 1.35x |
| `fak run` first token about 13.1 s, 116.5 tok/s decode | First token includes model load; file was page-cached, not disk-cold |
| Batched aggregate 783.7 tok/s across 24 streams | Aggregate across streams, not a per-stream rate; opt-in f16 KV; single run |
| Cohort prefill 137.7 to 372.1 tok/s (2.70x) end to end, 16 HTTP streams | Chat prompts, not the 19-token OSS prompts; the OSS reference is 352.36 aggregate |
| Context depth: 8192-token ms/token 21.47 to 13.30 (1.61x) | Short context (19 tokens) is within noise (8.63 against 8.55) |
| Prefill: pp512 prompt rate over HTTP 3880.4 vs the realugbun fork's 3092.2 tok/s (1.25x) | Same card and window; prompt rate, not end-to-end latency; one window; measured at 97ccb4a7d9, before W4A4 grouped prefill became the default; the answer check abstained at the 32-token cap for every engine |
| In-process batched prefill pp512 4598.2 tok/s, on by default | In-process bench, not an HTTP rate; one window; measured at 97ccb4a7d9, before W4A4 grouped prefill became the default |
| Historical M3 Pro Metal (retired vendored Splash port): 40.3 tok/s vs Splash 37.5 | **Historical only, retracted as a pure fak-native Metal claim**; the vendored Splash port is retired and pure `internal/metalgemm` resident receipts are pending |
| `install.sh` or `go install ...@latest` (`v0.59.0`), or a clone of `main` | Inkling-2-Mini ships in `v0.59.0` for CPU, Metal, and Linux Vulkan; CUDA needs a clone build (`-tags cuda`) or the CUDA container |
| The guard refuses with fixed reason codes | "Default-deny" only when naming strict mode; the shipped default allows unlisted tools |

Do not say: "faster than llama.cpp", "fastest Inkling runtime", "bit-identical"
without qualification, any served-endpoint rate or served time to first token,
any CUDA speculative figure as the default speed, the retired vendored Splash
M3 Pro figure as a pure `internal/metalgemm` fak-native Metal claim, a
per-stream rate for an aggregate row, or that `inkling` or `inkling-2-mini`
names a different build from `inkling2-mini`: both short names chain to the same
pinned GGUF on `main` (commit `4e47f88205`). The earlier 151.7 tok/s speculative figure is superseded by the depth 2 row above and is not quoted.

## Receipts

Ancestry check, run 2026-11-25 before this revision:

```sh
git fetch origin
git merge-base --is-ancestor b64405de760e2f13a62594ff8d029f4599fcdeb4 origin/main   # exit 0
git merge-base --is-ancestor cc6dca0b47a9c6a6c9166fb94efd819b1f380ee1 origin/main   # exit 0
git merge-base --is-ancestor b64405de760e2f13a62594ff8d029f4599fcdeb4 v0.59.0       # exit 0
git merge-base --is-ancestor cc6dca0b47a9c6a6c9166fb94efd819b1f380ee1 v0.59.0       # exit 0
```

Both builds are ancestors of `origin/main` and of tag `v0.59.0` (`2fb2f300a3e`),
the current release installed by `install.sh` and `go install ...@latest`.

Result: `b64405de76` (the build that measured 119.9) and `cc6dca0b47` (the
build that measured 118.1 and the fresh-user run) are ancestors of public
`origin/main`. The index's other measured commits were checked the same way
(`2e7682ba2a`, `507693d73b`, `66bcd27c6e`, `1d7f1dcf92`, `1f31533f2`,
`67bb976926a`, `93b941b07f`, `bb247124cc`): all are ancestors, so "on main: yes"
below means the measured build is an ancestor of `origin/main`. A row with no
commit in the index says so.

Benchmark campaign records are listed in
[docs/benchmarks/published/](../benchmarks/published/). Where a row says
"receipt held privately", the raw logs are in the maintainer archive because they
record local paths; the published campaign record summarises the run. An
"Index id" cell names the headline row (`hl:<id>`) that the charts and this table
share. A row that says "receipt held privately, not indexed" is not gated.

| Claim | Number | Hardware class | Runs | On main | Public evidence | Index id |
|---|---|---|---|---|---|---|
| fak vs stock llama.cpp decode, same window, same card and file (fak in-process bench vs llama-bench) | 118.1 vs 81.0 tok/s (1.46x); 119.6 with the first fak run excluded (first run 113.6) | RTX 5080 16 GB | ABBA x2: 4 fak runs of 127 steps, 4 `llama-bench -r 3` | yes (`cc6dca0b47`) | receipt held privately (same-window launch evidence) | `hl:samewindow.fak.decode_tok_s`, `hl:samewindow.stock.decode_tok_s`, `hl:samewindow.fak.excl_first.decode_tok_s` |
| Fresh-user path: from-source CUDA build, then `fak run` | build 507 s; first token 13.1 s; decode 116.5 tok/s (runs 116.9 and 116.1) | RTX 5080 16 GB | 1 build, 2 `fak run` processes (fresh and repeated) | yes (`cc6dca0b47`) | receipt held privately (same window); first token includes model load, file page-cached | `hl:freshuser.build_s`, `hl:freshuser.first_token_s`, `hl:freshuser.run_decode_tok_s` |
| fak decode, Inkling-2-Mini UD-Q3_K_XL (separate window, fork parity) | 119.9 tok/s | RTX 5080 16 GB | ABBA, 2 samples per arm | yes (`b64405de76`) | campaign record in [published/](../benchmarks/published/) (fork parity, 2026-11-24); raw logs held privately | `hl:fork.fak.decode_tok_s` |
| realugbun llama.cpp fork decode, same card and file | 119.5 tok/s | RTX 5080 16 GB | ABBA, 2 samples per arm | no commit (external) | same campaign record | `hl:fork.realugbun.decode_tok_s` |
| Decode at 1946 tokens of depth, fak vs fork (a loss, cross-engine, different harness) | 89.0 vs 113.8 tok/s | RTX 5080 16 GB | fak: one 31-step oracle run; fork: `llama-bench` tg32 at depth, -r 2 | yes (`b64405de76`, fak) | fork parity campaign record | `hl:fork.depth1946.fak_tok_s`, `hl:fork.depth1946.fork_tok_s` |
| Decode climb, first native build | 6.32 tok/s | RTX 5080 16 GB | ABBA n=2 per arm | no commit in index (not verified) | campaign record in [published/](../benchmarks/published/) (GPU decode climb) | `hl:climb.first_native.tok_s` |
| Decode climb, resident CUDA weights | 82.5 tok/s | RTX 5080 16 GB | ABBA n=2 per arm | no commit in index (not verified) | GPU decode climb campaign record (resident CUDA) | `hl:climb.resident.tok_s` |
| Decode climb, fused ops on a second box | 101.6 tok/s | RTX 5080 16 GB (second box) | ABBA n=2 per arm | yes (`1d7f1dcf92`) | fused climb campaign record; compare within that box only | `hl:climb.fused_box2.tok_s` |
| Opt-in speculative decode, depth 2 and depth 4, and plain decode on the same receipt | 171.5 (1.43x) and 180.9 (1.51x) vs 119.7 tok/s | RTX 5080 16 GB | ABBA n=2 per arm, profile off | yes (`67bb976926a`) | spec decode campaign record; one node; opt-in; not bit-exact on every prompt (capitals prompt); per-prompt speedups 1.09x to 1.35x | `hl:spec.plain.tok_s`, `hl:spec.d2.tok_s`, `hl:spec.d4.tok_s`, `hl:spec.d2.prompt_min_ratio`, `hl:spec.d2.prompt_max_ratio` |
| Opt-in speculative decode, earlier prior tree (superseded by the row above) | 151.7 tok/s, history only; superseded | RTX 5080 16 GB | ABBA n=2 per arm | yes (`93b941b07f`, landed equivalent) | climb history | `hl:spec.d2_prior.tok_s` |
| Opt-in PDL decode, plain (a loss, off by default) | 110.7 vs 117.2 tok/s (0.94x) | RTX 5080 16 GB | ABBA n=2 per arm, 127 steps | yes (`cc6dca0b47`) | PDL campaign record (2026-11-25); PDL measured slower on sm_120 | `hl:pdl.plain.on_tok_s`, `hl:pdl.plain.off_tok_s` |
| Opt-in PDL decode, with speculative depth 2 (a loss, off by default) | 137.1 vs 146.9 tok/s (0.93x) | RTX 5080 16 GB | ABBA n=2 per arm, 127 steps | yes (`cc6dca0b47`) | same PDL campaign record | `hl:pdl.spec.on_tok_s`, `hl:pdl.spec.off_tok_s` |
| Batched prefill, pp512, in-process (on by default) | 4598.2 tok/s | RTX 5080 16 GB sm_120 | ABBA, median of the warm samples, profile off | base yes (`e4dc39ee1f`); measured build `97ccb4a7d9` is that base plus the opt-in prefill-chunk lane, run with the chunk unset | matched prefill campaign record (2026-11-25); in-process, not an HTTP rate | `hl:prefill.batched.pp512_tok_s` |
| Prefill over HTTP, pp512 prompt rate, fak vs the realugbun fork (a matched win) | 3880.4 vs 3092.2 tok/s (1.25x) | RTX 5080 16 GB sm_120, same card and window | fork, native, native, fork order, server prompt time median | base yes (`e4dc39ee1f`, measured build as above); fork no commit (external) | same campaign record; prompt rate only, not end-to-end latency; the answer check abstained at the 32-token cap for every engine | `hl:prefill.http.native.pp512_tok_s`, `hl:prefill.http.fork.pp512_tok_s` |
| Batch of one (C1) against the single-stream reader (a loss by design) | 65.3 vs 119.9 tok/s (0.54x) | RTX 5080 16 GB sm_120 | single run | yes (`2e7682ba2a`) | batched decode campaign record (2026-11-25) | `hl:batched.c1.decode_tok_s`, `hl:batched.single_stream.tok_s` |
| Batched decode, C8, few-row kernels on and off (aggregate across 8 streams) | 333.7 vs 312.7 tok/s (1.07x) | RTX 5080 16 GB sm_120 | ABBA n=2 per arm | yes (`2e7682ba2a`) | batched decode campaign record | `hl:batched.c8.rows_on_tok_s`, `hl:batched.c8.rows_off_tok_s` |
| Batched decode, C24, f16 KV (aggregate across 24 streams, not per stream) | 783.7 tok/s | RTX 5080 16 GB sm_120 | single run | yes (`2e7682ba2a`) | batched decode campaign record; opt-in f16 KV | `hl:batched.c24_f16.agg_tok_s` |
| Cohort prefill, C16 HTTP, lockstep against ragged (aggregate across 16 streams, median of four rounds) | 137.7 to 372.1 tok/s (2.70x) | RTX 5080 16 GB sm_120 | 4 rounds per arm, median | ragged yes (`507693d73b`, a test commit matched to main by commit subject); lockstep no commit in index | cohort-prefill serve receipt (summary in the campaign record) | `hl:c16.cohort_prefill.lockstep_e2e_tok_s`, `hl:c16.cohort_prefill.ragged_e2e_tok_s` |
| OSS llama-server, C16 aggregate end to end (reference, 19-token raw prompts) | 352.36 tok/s | RTX 5080 16 GB sm_120 | upstream llama-server 33aeb17 | no commit (external) | OSS reference in the campaign record | `hl:oss.c16.e2e_tok_s` |
| Context depth, 8192-token prompt, ms/token before and after (tiled large-cap attend) | 21.47 to 13.30 ms/token (1.61x) | RTX 5080 16 GB sm_120 | ABBAAB n=3 per arm, 128 output tokens | after: yes (`bb247124cc`) | context cliff campaign record | `hl:ctx.p8192_d16416.before_ms_tok`, `hl:ctx.p8192_d16416.after_ms_tok` |
| Context depth, 19-token prompt, ms/token before and after (within noise) | 8.63 to 8.55 ms/token | RTX 5080 16 GB sm_120 | ABBAAB n=3 per arm, 128 output tokens | after: yes (`bb247124cc`) | context cliff campaign record | `hl:ctx.p19_d16416.before_ms_tok`, `hl:ctx.p19_d16416.after_ms_tok` |
| Historical M3 Pro Metal (retired vendored Splash port) decode vs Splash 1.7.2, short cells | 40.3 vs 37.5 tok/s (1.07x); 55.5 vs 54.9 (1.01x); 47.8 vs 46.5 (1.03x) | M3 Pro 36 GB | paired decode, two runs per arm | yes (`1f31533f2`, retired) | [m3pro-inkling2-mini-metal/RESULTS.md](../benchmarks/published/m3pro-inkling2-mini-metal/RESULTS.md) (historical; not pure `internal/metalgemm`) | `hl:metal.m3pro.capitals32.fak_tok_s`, `hl:metal.m3pro.capitals32.splash_tok_s`, `hl:metal.m3pro.fibonacci32.fak_tok_s`, `hl:metal.m3pro.capitals128.fak_tok_s` |
| Historical M3 Pro Metal (retired vendored Splash port) decode, 128-token fibonacci cell (a loss) | 63.6 vs 65.2 tok/s (0.97x) | M3 Pro 36 GB | paired decode, two runs per arm | yes (`1f31533f2`, retired) | same historical record | `hl:metal.m3pro.fibonacci128.fak_tok_s`, `hl:metal.m3pro.fibonacci128.splash_tok_s` |
| Historical M3 Pro Metal (retired vendored Splash port) decode, 128-token p128/32 cell (not claimed) | 177.1 vs 119.6 tok/s (1.48x) | M3 Pro 36 GB | paired decode, two runs per arm | yes (`1f31533f2`, retired) | same historical record; not claimed | `hl:metal.m3pro.p128_32.fak_tok_s`, `hl:metal.m3pro.p128_32.splash_tok_s` |
| Tokenizer parity with Hugging Face | 46/46 corpus lines | CPU | 1 corpus pass | not indexed | receipt held privately, not indexed | receipt held privately, not indexed |
| Dev-work target match, Qwen3.8-27B vs the other models compared (model quality, not a fak engine rate) | 44 vs 28–34 of 240 scored items (300-item set) | all models served by llama.cpp at 4-bit | see receipt | n/a (model quality) | receipt held privately, not indexed | receipt held privately, not indexed |

Charts: [decode comparison](assets/inkling2-decode-comparison.svg),
[decode climb](assets/inkling2-decode-climb.svg),
[batched serving](assets/inkling2-batched-serving.svg), and
[context depth](assets/inkling2-context-depth.svg).

## How these numbers stay current

Every number an indexed row draws or cites is the display value of one row in
[docs/benchmarks/published/HEADLINES.json](../benchmarks/published/HEADLINES.json),
an index generated from the private receipts. The four charts are drawn from
that index by `go run ./cmd/fak launch-visual`.

What is gated:

- Indexed decimal numbers, and integers that carry a unit word or sit in a table
  row citing a headline, in this document and in the generated SVGs must trace to
  an index row (its display value or ratio label, or an `allowed_numbers` entry
  with a reason). At commit time, LAUNCH_NUMBERS_FRESHNESS checks this when a
  staged path is under `docs/launch/`, the chart data file, or the index. In CI,
  `go run ./cmd/fak launch-visual --check` fails on SVG drift or an untraced number.
- A table row that cites `hl:<id>` must show that row's display value.
- `fak-benchdb publish -check` fails closed when a receipt changed, or when a
  campaign is neither indexed nor listed as not indexed.

What is advisory (private pre-commit, not a refusal): HEADLINE_LAG, when the index
is behind a changed receipt, and HEADLINE_UNINDEXED, when a campaign has no index
row and no not-indexed reason.

What is not gated: rows marked "receipt held privately, not indexed"; the
qualitative claims in the alternatives table; and the caveat wording, which is
copied from each index row.

A re-measured receipt changes a charted number only after it is republished into
the index, so the index is the thing to refresh.
