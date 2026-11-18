---
title: "Pipeline parallelism: cutting the layer stack across workers"
description: "Your model does not fit on one machine. The layer stack is the natural place to cut it: ForwardBand runs a contiguous band of layers, PipelineStage gives each worker one band, and StageTransport carries the hidden state across the boundary. The split is not an approximation — it is the same computation, and the repo proves it bit-exact. Level 3 of the fak inference ladder."
slug: pipeline-parallelism
keywords:
  - pipeline parallelism explained
  - layer parallelism
  - ForwardBand
  - StageTransport
  - pipeline bubble
  - model does not fit on one GPU
  - bit-exact pipeline split
  - incremental pipeline decode
  - fak inference ladder
date: 2026-11-18
---

# Pipeline parallelism

*You are on **Level 3 of 5** of the [fak inference ladder](README.md).*

> **Audience.** You have read [Level 1](level-1-what-is-inference.md) (a forward pass
> turns token ids into next-token scores) and [Level 2](level-2-the-layer-loop.md) (the
> middle of that pass is a stack of N same-shaped layers, connected by one small vector).
> Now the practical question: **the model does not fit on one machine — how do I split
> it?** You do not need to know any attention math for this rung. You need to be
> comfortable reading a Go function signature and a byte layout.

> **Short answer.** Cut the **stack**. Give worker 0 layers `[0, 8)`, worker 1 layers
> `[8, 16)`, and so on. Each worker loads only its own layers' weights, runs only its own
> layers, and hands the hidden state — one small `[seq][hidden]` block — to the next
> worker. The thing that crosses a boundary is the *activation*, not the weights. This is
> called **pipeline parallelism** (PP), and in fak the split is not an approximation: a
> cut stack produces **bit-for-bit identical** logits to the uncut model, and the repo
> has tests that assert exactly that.

## What are the two ways to split a model?

There are two, and they are genuinely different operations. Get the distinction in one
sentence each and the rest of this ladder falls into place:

- **Pipeline parallelism** cuts the **stack**. The boundary between two stages is a
  **hidden-state tensor** — the `[seq][hidden]` bus from Level 2. Nothing inside a layer
  is touched. *This page is about this one.*
- **Tensor parallelism** cuts **inside a layer**. The boundary is a **collective** — an
  `AllGather` or an `AllReduceSum` — that has to run *in the middle of* the layer's
  arithmetic, because the layer's own matmul has been split across workers.
  [Level 4](level-4-tensor-and-expert-parallelism.md) is about that one.

Here is the same model drawn both ways:

```text
  PIPELINE PARALLELISM (this page)          TENSOR PARALLELISM (Level 4)
  cut BETWEEN layers                        cut INSIDE a layer

  worker 0 ┌──────────────┐                 worker 0 ┌──────────────┐
           │  layer 0     │                          │  layer 0     │
           │  layer 1     │                          │  ┌────────┐  │
           │  layer 2     │                          │  │ W left │  │
           │  layer 3     │                          │  └────────┘  │
           └──────┬───────┘                          └──────┬───────┘
                  │ hidden state                            │ collective
                  │ [seq][hidden]                           │ AllGather /
                  │  ← ONE tensor, small                    │ AllReduceSum
           ┌──────▼───────┐                          ┌──────▼───────┐
  worker 1 │  layer 4     │                 worker 1 │  ┌────────┐  │
           │  layer 5     │                          │  │ W right│  │
           │  layer 6     │                          │  └────────┘  │
           │  layer 7     │                          │  layer 0     │
           └──────────────┘                          └──────────────┘

  the seam is a TENSOR                        the seam is a COLLECTIVE
  between two whole layers                    inside one layer's matmul
```

The practical difference: a pipeline boundary is a **send**, and a tensor boundary is a
**synchronisation**. A send can be pipelined and overlapped; a synchronisation cannot,
because the layer literally cannot finish its arithmetic until every rank has
contributed. That is why PP is the easier axis to reason about, and why it is the one
this rung teaches first.

## Why is the layer stack the natural cut?

Level 2 already gave you the three properties. Restated as the reason the cut is *cheap*:

1. **The layers are sequential.** Layer 5 reads layer 4's output, so the work is already
   ordered. A cut does not create a dependency; it just assigns an existing one to a
   different machine.
2. **The layers are identically shaped.** "Run layers `[0, 8)`" and "run layers `[8, 16)`"
   are the *same function* on different data. One code path, N calls.
3. **The tensor crossing the boundary is small.** Only the `[seq][hidden]` bus crosses.
   The weights — the billions of numbers, the thing that did not fit — stay put on
   whichever worker owns them.

Point 3 is the one that pays. Compare the two seams by *what has to move*:

```text
  PIPELINE BOUNDARY                     TENSOR BOUNDARY
  ─────────────────                     ───────────────
  what crosses:  the activation         what crosses:  partial results of a matmul
  size:          seq × hidden × 4 B     size:          depends on the shard
  when:          after a whole layer    when:          MID-LAYER, twice per block
  can overlap?   yes — send and keep    can overlap?   no — the layer cannot
                 computing the next                    finish until the
                 microbatch                            collective returns
```

A pipeline boundary is a clean seam you can put a network in front of. A tensor boundary
is a hole cut in the middle of a computation. Both are real, both are in this repo, and
they compose — but they are not the same kind of cut, and conflating them is the single
most common confusion in this area.

## What is the unit of execution?

Here is the load-bearing claim of this entire rung, and it is a claim about *identity*,
not about approximation.

The unit is `ForwardBand` (`internal/model/partition.go:122`):

```go
func (m *Model) ForwardBand(x [][]float32, lo, hi int, isLast bool) ([][]float32, [][]float32, error)
```

Read the signature as a sentence: *"take this already-embedded hidden state `x`, run
layers `lo` up to but not including `hi` over it, and if `isLast`, also run the final
norm and the LM head and give me the logits."*

Now the claim. The doc comment on `ForwardBand` states it directly:

> `Forward` is exactly `embedBand(ids)` followed by `ForwardBand(x, 0, NumLayers, true)`.

And that is literally what the code does. `Forward` (`internal/model/forward.go:483`)
ends with:

```go
return m.forwardHiddenRows(m.embedBand(ids))
```

`embedBand` (`internal/model/partition.go:91`) is the token-embedding lookup — the head of
the forward pass. `forwardHiddenRows` (`internal/model/forward.go:508`) is the layer loop,
`for l := 0; l < cfg.NumLayers; l++`. So the monolithic forward pass **is** a single band
covering every layer, with the embedding in front and the head at the end.

Why does that matter so much? Because it means the split is not a re-derivation. It is the
*same instruction stream* over the *same bytes*:

- Every weight is read by per-layer **name** (`model.layers.N.*`), so a worker holding only
  `[lo, hi)` reads exactly the weights the monolith would have read for those layers, and
  no others.
- RoPE is keyed on the **absolute** layer index, so layer 12 behaves like layer 12 whether
  it is the fifth layer of a band or the thirteenth layer of the whole stack.
- The payload is `float32`, which round-trips bit-exact as IEEE-754 little-endian.

Put those together and a cut stack cannot produce a different answer. The repo does not
ask you to take that on faith — `TestForwardBandTwoWayMatchesMonolithic`
(`internal/model/forward_band_test.go:17`) runs the full `Forward`, then runs the same
model as two bands, and asserts `max|delta| = 0` on the logits.

`ForwardBand` also **fails closed** rather than producing garbage. It rejects a malformed
range (`lo < 0 || hi <= lo || hi > cfg.NumLayers`, `internal/model/partition.go:124`), and
for a GLM-MoE-DSA model it rejects a band that would *begin* in the middle of an
IndexShare group (`internal/model/partition.go:127`) — a "shared" indexer layer reuses the
previous "full" layer's index, and a band starting on the shared layer never computed it.
That is a real architectural constraint, not a formality: it is why the plan validator
below has an opinion about *where* you may cut.

## How does a worker load only its own layers?

Splitting the *computation* is only half the job. The other half is the reason you are
here: the worker must not materialise weights it will never use, or you have not saved any
memory.

That is the loader-side band filter, three small pieces in `internal/model/partition.go`:

- `layerWindow` (`:28`) — a half-open band `[lo, hi)` of layer indices. Its zero value
  keeps everything, so a load with no option is byte-for-byte the old behaviour.
- `keepsLayer` (`:40`) — "does this worker keep layer N?" Note the first branch: a
  **layer-agnostic** tensor (index `< 0` by convention) is *always* kept. The embedding,
  the final norm, and an untied LM head have no layer index, and a worker that owns the
  head or the embedding needs them regardless of which band it holds.
- `WithLayerWindow(lo, hi)` (`:61`) — the `LoadOption` (`:54`) a worker passes to keep only
  its own band. `hi <= lo` would select nothing and is rejected at apply time as a
  programming error rather than silently loading an empty model.

The name-parsing rule lives in one place, `tensorLayerForWindow` (`:80`), so the loader
does not duplicate it: it returns the layer index a tensor belongs to, or `-1` for a
layer-agnostic tensor.

```text
  ONE CHECKPOINT ON DISK
  ┌──────────────────────────────────────────────────────────────┐
  │  model.embed_tokens.weight          (layer-agnostic)         │
  │  model.layers.0.*                   (layer 0)                │
  │  model.layers.1.*                   (layer 1)                │
  │  ...                                                         │
  │  model.layers.31.*                  (layer 31)               │
  │  model.norm.weight                  (layer-agnostic)         │
  │  lm_head.weight                     (layer-agnostic)         │
  └──────────────────────────────────────────────────────────────┘
        │                    │                    │
        │ WithLayerWindow    │ WithLayerWindow    │ WithLayerWindow
        │ (0, 8)             │ (8, 16)            │ (24, 32)
        ▼                    ▼                    ▼
  ┌───────────┐        ┌───────────┐        ┌───────────┐
  │ embed     │        │ layers    │        │ layers    │
  │ layers 0-7│        │ 8-15      │  ...   │ 24-31     │
  │           │        │           │        │ norm      │
  │           │        │           │        │ lm_head   │
  └───────────┘        └───────────┘        └───────────┘
   stage 0              stage 1              stage 3
   (First)              (interior)           (Last)
```

`TestPipelineSeparatelyLoadedStagesMatchMonolithic` (`internal/model/pipeline_test.go:27`)
is the test that makes this real rather than aspirational. It loads the *same* checkpoint
three times with different windows, asserts that stage A holds **no** layer-2 weights and
stage B holds **no** layer-0 or layer-1 weights, and then runs the pipeline and checks the
logits against the monolith at `max|delta| = 0`. The stages are genuinely partitioned
checkpoints, not one shared model wearing two hats.

## How do stages talk to each other?

The seam is one interface, `StageTransport` (`internal/model/pipeline.go:225`):

```go
type StageTransport interface {
	Send(hidden [][]float32, nextLo, dstStage int) ([][]float32, int, error)
}
```

That is the whole contract. `Send` takes a stage's output hidden state, the absolute layer
index the receiver should resume at, and the receiving stage's index (for diagnostics
only), and returns what the next stage receives.

There are two implementations, and the pair is the point:

- `LocalTransport` (`:234`) — the single-box default. It marshals the hidden state and
  immediately unmarshals it. It is the *stand-in* for a wire send, and it is bit-exact by
  construction because it is an IEEE-754 round-trip.
- `TCPTransport` (`internal/model/pipeline_transport.go:31`) — a **real socket**. It
  marshals, writes a length-prefixed frame to a peer, reads the reply frame, and
  unmarshals it (`Send`, `:44`). On loopback this is a genuine cross-process send.

The reason both exist is honesty. `LocalTransport` alone would leave "the wire is a swap
underneath the contract" as an aspiration — a claim with no implementation behind it.
`TCPTransport` is that implementation, and `TestTCPTransportMatchesLocal`
(`internal/model/pipeline_test.go:534`) asserts the socket path produces the same bytes as
the in-process path. So the pipeline is demonstrably transport-agnostic on hardware that
exists, and a real fleet swaps in NCCL or gRPC the same way: a new `StageTransport`, not a
change to the generation loops.

## What is actually on the wire?

The codec is `MarshalHidden` (`internal/model/pipeline.go:160`) and `UnmarshalHidden`
(`:188`). The layout is a fixed **12-byte header** (`hiddenHeaderBytes = 12`, `:153`)
followed by the float words:

```text
  byte offset   0        4        8        12
                │        │        │        │
                ▼        ▼        ▼        ▼
              ┌────────┬────────┬────────┬──────────────────────────────────┐
              │  seq   │ hidden │ nextLo │  seq × hidden × float32 words    │
              │ int32  │ int32  │ int32  │  row-major, little-endian        │
              └────────┴────────┴────────┴──────────────────────────────────┘
              └──────── 12 bytes ───────┘
                 hiddenHeaderBytes = 12

  seq      = how many positions are in this frame
  hidden   = the width of each position's vector
  nextLo   = the ABSOLUTE layer index the receiver must resume at
  payload  = seq*hidden float32 words, each via math.Float32bits
```

Two details are worth pausing on.

**Why `math.Float32bits` and not a cast?** Because `Float32bits` is the *identity bit
pattern*. It preserves signed zero and NaN payloads, so the round-trip cannot perturb a
single bit. A codec that "helpfully" normalised `-0.0` to `0.0` would break the bit-exact
gates, and the failure would be a one-in-a-million logits delta that nobody could find.

**Why does the header carry `nextLo` at all?** This is the interesting one. The receiving
stage does not trust that it was handed the right tensor. It **verifies** it:

```go
if gotLo != wantLo {
    return nil, fmt.Errorf("model: handoff resume layer %d != stage %d band start %d", gotLo, dstStage, wantLo)
}
```

That is `handoff` (`internal/model/pipeline.go:253`), and it is a **fail-closed check, not
a convention**. If a frame is routed to the wrong worker — a misconfigured plan, a stale
connection, a reordered send — the worker refuses rather than running its band against a
hidden state that was never meant for it. The same check is repeated on the worker side in
`ServeBand` (`internal/model/pipeline_transport.go:169`). A pipeline that silently computed
the wrong answer would be far worse than one that stops.

`UnmarshalHidden` fails closed too: a buffer shorter than the header, a negative `seq` or
`hidden`, or a length that does not equal `12 + seq*hidden*4` is an error, not a
best-effort decode (`:189`–`:201`).

## How is the plan described?

A plan is a validated tiling of the layer range into stages. Four types, all in
`internal/model/pipeline.go`:

- `StageSpec{Lo, Hi, First, Last}` (`:41`) — one worker's responsibility: the half-open
  band `[Lo, Hi)` it owns, plus its role. `First` means it embeds the token ids (it is the
  head of the pipeline); `Last` means it runs the final norm and LM head and returns logits
  (it is the tail).
- `PipelineStage` (`:51`) — a runnable worker: a standalone `Model` plus the `StageSpec` it
  owns.
- `PartitionPlan` (`:57`) — an ordered, validated list of `StageSpec`s.
- `NewPartitionPlan(cfg, cuts)` (`:68`) — builds a plan from **interior cut points**.
  `cuts = {8, 16, 24}` over 32 layers gives `[0,8)`, `[8,16)`, `[16,24)`, `[24,32)`.
- `Validate` (`:101`) — re-checks a plan and fails closed on any malformed tiling.

### A worked example: 32 layers, four stages

```text
  cfg.NumLayers = 32
  cuts          = {8, 16, 24}

  stage 0   [0, 8)    First=true   Last=false   embeds the ids, runs layers 0-7
  stage 1   [8, 16)   First=false  Last=false   runs layers 8-15
  stage 2   [16, 24)  First=false  Last=false   runs layers 16-23
  stage 3   [24, 32)  First=false  Last=true    runs layers 24-31, then norm + head

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  the layer stack, cut into four bands                                    │
  ├──────────────────┬──────────────────┬──────────────────┬─────────────────┤
  │  layers 0 – 7    │  layers 8 – 15   │  layers 16 – 23  │  layers 24 – 31 │
  │  + embedding     │                  │                  │  + norm + head  │
  │  stage 0 (First) │  stage 1         │  stage 2         │  stage 3 (Last) │
  └────────┬─────────┴────────┬─────────┴────────┬─────────┴────────┬────────┘
           │                  │                  │                  │
           │  hidden state    │  hidden state    │  hidden state    │
           │  [seq][hidden]   │  [seq][hidden]   │  [seq][hidden]   │
           │  nextLo = 8      │  nextLo = 16     │  nextLo = 24     │
           ▼                  ▼                  ▼                  ▼
        ── send ──►        ── send ──►        ── send ──►      logits [seq][vocab]
        (StageTransport)   (StageTransport)   (StageTransport)  (terminal)

  What crosses each boundary: ONE [seq][hidden] tensor. Not the weights.
  What does NOT cross: the weights, the KV cache, the DSA top-k state.
```

`Validate` enforces the properties that make that picture well-formed (`:101`–`:148`):

- the bands **tile** `[0, NumLayers)` contiguously — the first starts at 0, the last ends
  at `NumLayers`, and each `Lo` equals the previous `Hi` (no gap, no overlap);
- every band is non-empty and in range;
- exactly one stage is marked `First` (and it is the first), exactly one is marked `Last`
  (and it is the last);
- for a GLM-MoE-DSA config, every interior boundary falls on a full- or dense-indexer
  layer, so no IndexShare group is split across the wire.

That last rule is the plan-time lift of the per-band check inside `ForwardBand`. It is
enforced *before any worker loads weights*, which is the right time to find out that your
cut is illegal.

## How does a whole generation run?

Two loops, and the difference between them is the most interesting engineering in this
rung.

### The simple loop: re-forward the whole sequence

`RunPipeline(ids, stages)` (`internal/model/pipeline.go:281`) is the single-box end-to-end
harness. It is `RunPipelineWith(ids, stages, LocalTransport{})` (`:289`), and the loop is
short: stage 0 embeds and runs its band; at each boundary the hidden state goes through
`handoff`; each subsequent stage runs its band; the last stage runs the head and returns
logits.

`PipelineGenerate(prompt, n, stages)` (`:346`) wraps that in a greedy decode loop. Its own
doc comment is refreshingly blunt about the cost: it **re-forwards the growing sequence
each step — O(n²)**. Step 1 forwards 1 token, step 2 forwards 2, step 3 forwards 3. It is
the runnable proof that fak generates across genuinely partitioned stages. It is not a
throughput path, and it does not pretend to be.

### The real loop: incremental decode

The interesting version keeps a **KV cache** — the per-layer memory of every position the
model has already seen — resident inside the stage that owns the layers. That is
`PipelineStageDecoder` (`:383`):

```go
type PipelineStageDecoder struct {
	Spec    StageSpec
	Session *Session
}
```

Why a `Session` and not a bare cache? The doc comment answers it: the `Session` is the
unit so that the **last stage's head is literally `Session.glmDsaHead`** — the exact oracle
the monolithic `Session.Generate` uses. There is no second head path that could drift from
the first. And out-of-band cache layer slices stay `nil`, so resident KV is bounded to the
band. That is the whole point of pipeline parallelism: the memory a worker holds is
proportional to the layers it owns.

The pieces:

- `NewPipelineStageDecoder(spec, m)` (`:391`) — builds a stateful decoder over a windowed
  model and allocates the band's KV cache.
- `decode(id, x, pos)` (`:400`) — advances this stage's band by **one position**.
- `StepPipelineDecode(id, pos, stages)` (`:474`) — advances *every* stage by one position
  for token `id` at absolute position `pos`, threading the single-position hidden across
  the transport, and returns the last stage's logits.
- `StepPipelineDecodeWith(...)` (`:480`) — the same, parameterised by the `StageTransport`.
- `PipelineGenerateIncremental(prompt, n, stages)` (`:514`) — the full loop: reset, prefill
  the prompt in lockstep at positions `0..len-1`, then decode from position `len`.
- `ResetStageDecoders(stages)` (`:438`) — restores every stage to a fresh, position-zero
  cache. This one is a footgun guard: a decoder carries state across calls, so reusing a
  decoded-on set for a new sequence without resetting would silently **concatenate** the
  two sequences.
- `assertBandAtPos(st, pos)` (`:452`) — fails closed if a stage's band has not run exactly
  `pos` times. It reads the stage's *own* band cache rather than a global length, because
  interior stages never advance a global position counter. A stage that has drifted out of
  lockstep is rejected before its next append would mis-assert a row.
- `validateDecoderPlan(stages)` (`:409`) — re-checks a hand-built decoder set the same way
  `RunPipeline` checks `PipelineStage`s: nil sessions rejected, every stage's `NumLayers`
  must agree (a stage loaded against the wrong checkpoint is rejected, not silently mixed),
  and the specs must form a valid `PartitionPlan`.

### Why does decode need a stateful per-stage decoder?

Because the KV cache lives *inside* the stage that owns the layers. This is the part that
makes incremental PP harder than incremental single-worker decode, and it is worth a
diagram:

```text
  RE-FORWARD (RunPipeline)                INCREMENTAL (StepPipelineDecode)
  ────────────────────────                ────────────────────────────────
  every step re-runs the whole            every step appends ONE position
  sequence through every stage            to each stage's own band cache

  step 3: forward [t0 t1 t2 t3]           step 3: append t3 to stage 0's cache
          through all 32 layers                    append t3 to stage 1's cache
          (t0, t1, t2 recomputed)                  append t3 to stage 2's cache
                                                   append t3 to stage 3's cache
  cost: O(n²) over the generation          cost: O(n) over the generation

  ┌──────────────────────────────────────────────────────────────────────┐
  │  stage 0 KV cache:  layers 0-7   only                                │
  │  stage 1 KV cache:  layers 8-15  only                                │
  │  stage 2 KV cache:  layers 16-23 only                                │
  │  stage 3 KV cache:  layers 24-31 only                                │
  └──────────────────────────────────────────────────────────────────────┘
  Each stage's resident KV is bounded to its band. That is the memory win.
```

And `validateDecoderPlan` refusing a plan whose cut lands on a shared layer is the same
rule as `Validate`'s, applied to the decode path: a stage that begins on a GLM shared
indexer layer never computed the index it is supposed to reuse, so the plan is refused at
plan time rather than producing a subtly wrong token at position 400.

## What does the worker side look like?

So far the stages have all been in one process. The worker side is what makes it a real
pipeline.

`ServeBand(conn, stage, downstream)` (`internal/model/pipeline_transport.go:150`) is the
band-running worker serve loop. It reads a frame, decodes it to a hidden state, checks the
resume layer against its own band start, runs `ForwardBand` over its resident band, and
replies with its **own** output frame:

- the **last** stage marshals the per-position logits it produced under a sentinel resume
  layer (`replyResumeLayer = -1`, `:126`) — a value distinct from every valid band start,
  so a logits reply can never be mistaken for a hidden state handed to a downstream band;
- an **interior** stage forwards its output hidden state to the next worker over
  `downstream` and relays that worker's reply back upstream unchanged.

It serves frames until the connection closes, so one worker handles every token of a
generation over a persistent connection. A nil `downstream` on an interior stage is a
configuration error, surfaced rather than deadlocked.

`EchoFrames(conn)` (`:106`) is the reference peer: it reads each frame and writes it
straight back. It is the identity peer that proves the transport itself preserves the
payload bit-for-bit. `ServeBand` is the real peer that replaces it.

`RunPipelineAcrossWorkers(ids, first, downstream)` (`:221`) is the multi-node counterpart
of `RunPipelineWith`. It runs **only the first stage** in this process and delegates every
later stage to a remote `ServeBand` worker. Where `RunPipelineWith` uses the transport as a
marshalling boundary with an echo peer, here the remote bands run *only* inside the worker
serve loops — so a two-stage run is a genuine cross-process pipeline.

### The engine bridge

`internal/modelengine/pipeline.go` puts that substrate behind the same engine surface the
in-process engine uses, so a host can register and select it through the normal driver path
instead of a test-only harness. (A note on the word "kernel": this repo uses it in three
unrelated senses — the OS-metaphor agent kernel, an HPC compute kernel, and a literal CUDA
kernel. The `"inkernel-pipeline"` id below is the *in-process* sense: the engine that runs
the model in the same process, as opposed to a remote worker. It is not a CUDA kernel.)

- `PipelineEngineID = "inkernel-pipeline"` (`:26`) — the default id for a PP head engine.
- `PipelineEngine` (`:32`) — drives the first stage in-process and reaches the rest through
  `downstream`, typically a `TCPTransport` whose peer is `ServeBand`.
- `NewPipelineEngine(first, downstream)` (`:46`) — builds it. `first` must be the `First`
  stage; if it is not also `Last`, `downstream` must reach the next worker.
- `SetWorkerCoupler` (`:51`) / `WorkerCoupler` (`:59`) — the coupler seam.
- `Admit(ctx, call)` (`:101`) — starts a streamed generation. It is deliberately
  conservative: one admitted request re-forwards the growing sequence through
  `RunPipelineAcrossWorkers` once per generated token, then streams that token. That is
  slower than an incremental PP scheduler, and the file says so — but it puts the real
  `ServeBand` loop behind the same admit/stream/reclaim surface without forking the proven
  worker code.

Note the mutex in `PipelineEngine` (`:41`): `TCPTransport` is a request/reply stream over
one connection, so concurrent admitted requests are serialised to stop their
length-prefixed frames from interleaving.

## What is the honest cost?

Pipeline parallelism buys **fit**. It does not, by itself, buy **speed**. Say that plainly,
because the opposite is the most common thing people believe about it.

With **one request in flight**, all but one stage is idle at any instant. Stage 0 works
while stages 1, 2, and 3 wait; then stage 1 works while 0, 2, and 3 wait. That idle time
is called the **bubble**, and it is not a bug — it is what a sequential computation looks
like when you spread it over parallel hardware.

```text
  ONE REQUEST IN FLIGHT — the bubble is most of the machine
  time ──►
  stage 0  [███][   ][   ][   ]
  stage 1  [   ][███][   ][   ]
  stage 2  [   ][   ][███][   ]
  stage 3  [   ][   ][   ][███]
           └───┴───┴───┴───┘
           at any instant, exactly ONE stage is working.
           the other three are idle. That is the bubble.

  SEVERAL REQUESTS (or micro-batches) — the bubble fills
  time ──►
  stage 0  [A][B][C][D]
  stage 1  [ ][A][B][C]
  stage 2  [ ][ ][A][B]
  stage 3  [ ][ ][ ][A]
           └───┴───┴───┴───┘
           once the pipe is full, every stage is busy.
           A, B, C, D are four different requests (or four micro-batches
           of one request) at different points in the same stack.
```

What fills the bubble is **more work in flight**: multiple concurrent requests, or
micro-batches of one request split into chunks that pipeline through the stages. The repo
has the incremental decode machinery that a scheduler would build on
(`StepPipelineDecode`, `internal/model/pipeline.go:474`; `PipelineGenerateIncremental`,
`:514`), but the *scheduling* that fills the bubble — how many micro-batches, in what order
— is not a witnessed end-to-end path here. [Level 5](level-5-the-parallelism-frontier.md)
names that as an open edge.

**There is no measured pipeline-parallel speedup in this repo.** Not a small one, not a
qualified one — none. The repo witnesses that a cut stack gives the *same answer*; it does
not witness that the cut stack is *faster*. If you see a number on this page, it is a byte
count or a layer count, never a measurement.

There is also no `fak serve` flag for pipeline stages. The serve surface has
`--tensor-parallel` and `--expert-parallel` (Level 4), but the PP path is reached through
the library and the engine bridge, not a serve flag. That is an honest gap, not an
oversight to paper over.

## What is witnessed?

Every row below is a real test in the repo. The third column says what "matches" *means*
in that row, because "matches" is not one thing.

| Mechanism | Test that pins it | What the test actually asserts |
|---|---|---|
| Band split | `TestForwardBandTwoWayMatchesMonolithic` (`internal/model/forward_band_test.go:17`) | a two-way band split equals the monolithic forward — **bit-exact** (`max\|delta\| = 0`) |
| Separately-loaded stages | `TestPipelineSeparatelyLoadedStagesMatchMonolithic` (`internal/model/pipeline_test.go:27`) | stages loaded as *separate windowed models* equal the monolith — **bit-exact**, and each stage is asserted to hold no out-of-band layer weights |
| Incremental decode | `TestPipelineIncrementalMatchesMonolithicSession` (`internal/model/pipeline_test.go:118`) | stepwise pipeline decode produces the same **token ids** as a monolithic `Session.Generate` |
| Per-step logits | `TestPipelineIncrementalLogitsBitExactPerStep` (`internal/model/pipeline_test.go:159`) | the **full logits vector at every decode position** is bit-identical — stronger than the token-id gate, because a sub-argmax delta a near-tie hides would still trip it |
| Shared boundary refused | `TestPipelineIncrementalRejectsSharedBoundary` (`internal/model/pipeline_test.go:223`) | a band that starts on a GLM shared-indexer layer is **refused**, not silently served |
| Generate loop | `TestPipelineGenerateMatchesMonolithicSession` (`internal/model/pipeline_test.go:363`) | the full greedy generate loop equals a monolithic session |
| Cross-process transport | `TestTCPTransportMatchesLocal` (`internal/model/pipeline_test.go:534`) | `TCPTransport` over a real loopback socket equals `LocalTransport` — **equality, not throughput** |
| Stage worker | `TestPipelineEngineAdmitRunsServeBandWorker` (`internal/modelengine/pipeline_test.go:19`) | the engine admits a request and runs a real `ServeBand` worker over TCP, streaming the same tokens as the monolith |

Read the table as a spectrum of *what is proven*, not a checklist:

```text
  PROVEN                                    NOT PROVEN HERE
  ─────────────────────────────────────     ────────────────────────────────
  a cut stack gives the SAME ANSWER         a cut stack is FASTER
  the socket path equals the local path     the socket path is fast
  a bad plan is REFUSED                     a good plan is OPTIMAL
  a stage runs its own band                 a stage recovers from a crash
  incremental decode matches the monolith   the bubble is filled by a scheduler
```

The right-hand column is not a list of failures. It is the boundary of what the tests
claim, and knowing where that boundary is is the difference between using this machinery
and misusing it.

## Try it

```bash
# The band identity: a two-way split equals the monolithic forward, bit-exact.
go test ./internal/model/ -run TestForwardBandTwoWayMatchesMonolithic -v

# The real thing: stages loaded as SEPARATE windowed models, still bit-exact.
go test ./internal/model/ -run TestPipelineSeparatelyLoadedStagesMatchMonolithic -v

# Incremental decode: per-stage KV, one position per step, matches the monolith.
go test ./internal/model/ -run TestPipelineIncrementalMatchesMonolithicSession -v

# The wire: a real loopback socket equals the in-process transport.
go test ./internal/model/ -run TestTCPTransportMatchesLocal -v

# The worker: the engine admits a request and runs a ServeBand worker over TCP.
go test ./internal/modelengine/ -run TestPipelineEngineAdmitRunsServeBandWorker -v

# Declared memory accounting for a layer-band placement (no model loading).
fak model fit bands --input placement.json --manifest shards.json --manifest-sha256 SHA256
```

The first four are the correctness story in four commands. The last one is the *planning*
side: `fak model fit bands` checks a declared per-rank layer-band placement against an
authenticated shard manifest and reports whether each band fits its declared charge. It
performs no model loading and no hardware qualification — it is declared accounting, and
it says so.

## See also

- [Level 2 — The layer loop](level-2-the-layer-loop.md): the residual stream and the
  same-shaped layer stack this rung cuts.
- [Level 4 — Tensor and expert parallelism](level-4-tensor-and-expert-parallelism.md): the
  other axis — cutting *inside* a layer, where the seam is a collective rather than a
  tensor.
- [Level 5 — The parallelism frontier](level-5-the-parallelism-frontier.md): how the axes
  compose, and the honest fence around what is witnessed.
- [Multi-GPU tensor parallelism in fak](../multi-gpu-tensor-parallelism.md) — the
  architecture reference for the tensor axis, including the exact NCCL/RCCL swap-in point.
  This page cross-links to it rather than duplicating it.
- [The fak caching ladder](../caching/README.md) — the sibling ladder. Caching is about
  *not recomputing* work you have already done; inference is about *doing* the work. They
  meet at the KV cache, which the incremental decode path above keeps per stage.
- [Addressable KV cache](../addressable-kv-cache.md) · [Context](../context.md) — the
  cache and context layers the serving path sits on top of.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---
*[Full ladder (README)](README.md) · next → [Level 4 — Tensor and expert parallelism](level-4-tensor-and-expert-parallelism.md)*
