# Ordinary Qwen host chunkwise Gated-DeltaNet

## Delivered unit and selection

One request-local **prefill state/readout subsystem**, not a decode replacement:
ordinary Q8 and mixed-Q4_K Qwen hybrid `Session.Prefill` / `PrefillNoLogits`
produce projections and causal convolution using their existing paths, normalize
Q/K as before, then optionally consume the complete panel with this chunkwise
transition. The existing output gate, normalization, projection, cache positions,
full-attention layers, and next-token decode remain unchanged.

Default builds retain the old serial recurrence. The typed qualification build
tag is `fak_qwen35_gdn_chunked_host_qualification`. An ordinary untagged build restores the original recurrence. A package-private,
invocation-local typed force-scalar mode supplies the same-binary test oracle;
it cannot enable the candidate in an unqualified build. No new behavioral
environment variable or mutable global/session state is introduced.
The existing vectorized-decode operator opt-in takes precedence. Direct callback
wrappers never grant ordinary-producer provenance. Metal, MetalQ4K, HAL/backend,
Prism, capture, residual-hook and active-tap paths are excluded. The existing
Q4K preprojected device sequence gets its original first refusal/acceptance.

There is no global workspace, cache, background worker, or persistent owner.
No shader, AMX instruction, BF16 conversion, Torch/BLAS or new runtime dependency
is added. The portable F32 source is deliberately **not bitwise equivalent** to
serial decode: matrix/triangular reductions regroup floating-point operations.

## Pinned donor and exact adaptation

vLLM `5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`, Apache-2.0:

- `csrc/cpu/sgl-kernels/fla.cpp`, Git
  `0cf34acaac281918d1a976effaf80b47b462db03`, SHA256
  `ba86e35f3c7407f969d1e8fa203d6d2894bcb12a91ed36ee55a91f0a2a3f3343`:
  chunk intra solve/recompute (1035 onward), inter state/readout (1273 onward),
  and `chunk_gated_delta_rule_cpu` (2242 onward). Its header attributes SGLang.
- `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`, Git
  `0a4dd19bf5a2e0ab27b2f638c2231e15fd27b55f`: real causal-convolution,
  initial/final recurrent-state and ordinary prefill consumer.
- Root `LICENSE`, Git `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`,
  SHA256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
  Its complete text is already reproduced byte-exactly in THIRD_PARTY_NOTICES.
  No root NOTICE or separate sgl-kernels LICENSE/README was present at this pin.

The donor CPU implementation requires BF16 Q/K/V/beta, equal D/Dv, multiples of
32, chunk64 and Torch/BLAS/AMX machinery. This adaptation retains the chunk64
causal triangular solve, inter-chunk state and separate Q readout, but supports
FAK's native F32 rectangular keyWidth × valueWidth state. It applies the
triangular inverse directly to the value RHS rather than materializing W/U.
It packs one Q/K tile and forms causal KK/QK tiles; this is not an additional
copy of the unrelated FAK WYF helper, whose pre-decay projection and K readout
are different equations.

With scalar decay g and beta computed by the EXISTING FAK gate formulas,
define prefix_i=product(g_0..g_i), d_ij=product(g_(j+1)..g_i), and incoming S0:

```
Vnew_i = beta_i * (V_i - prefix_i * K_i*S0
                  - sum(j<i) d_ij * dot(K_i,K_j) * Vnew_j)
O_i    = prefix_i * Q_i*S0 + sum(j<=i) d_ij * dot(Q_i,K_j) * Vnew_j
Slast  = prefix_last*S0 + sum(j) d_last,j * outer(K_j,Vnew_j)
```

Rounded scalar-decay products deliberately replace exp(cumulative-log
differences). This admits a legitimate exact-zero reset without log(0), vanished
prefix division, or 0/0. Q remains distinct and already scaled; value-head to
key-head grouping is unchanged. Tail chunks only read their actual token count.
The initial state is never implicitly zeroed. Continuation consumes the supplied
final state; no intermediate state history is retained.

## Admission, ownership and refusal

Admitted geometry: 64..4096 tokens, 1..64 value heads, integral value/key head
grouping, key/value widths 1..256, exact input/state lengths. All products fit
32-bit indices under these bounds. Workers are clamped to min(requested, heads,
16), with one reused fixed workspace each. The complete candidate float payload
(output + copied final states + all worker scratch) must fit **64 MiB**.
This is not a total-process/RSS cap: existing projection panels, the caller's
preexisting core allocation, allocation headers and goroutine stacks are extra.
The scope is one synchronous layer invocation, not a global admission budget.

All inputs are borrowed read-only. Candidate output and per-head final states
are separately owned. Every head worker joins before publication. Finite-input,
geometry and budget refusals happen before candidate allocation; a later
nonfinite intermediate is discarded without publishing any candidate output or
state, and the caller executes the unchanged legacy recurrence. Therefore
NaN/Inf/overflow behavior is not replaced by fabricated finite values.
The existing causal-convolution history update is retained exactly before this
seam; fallback uses that same already-produced panel, not a second convolution.
There is no new asynchronous/cancellation boundary: existing session/layer
cancellation and error handling applies, and no worker outlives the call.

The selected phase key is `qwen35_linear_chunked_host`, replacing (not nesting
inside) `qwen35_linear_recurrent` for an accepted layer. A declined attempt plus
legacy fallback remains charged to the original recurrent bucket. The marker is
an actual selected-consumer witness only; it is not a measured gain.

## Qualification / rollback debt

Source baseline: published `0c8798d56a783799f0b035d1c9be2890119c86ba`.
Preserve its earlier NormGain, RoPE, one-panel workspace and tiled attention
compositions. Current-main dedup pin at intake:
`81f5f5464277cc176011db0fe19a789f7b003cd1`; main integration remains separate.

Focused software witness (commands recorded, **UNRUN**):

```sh
go test ./internal/model -run '^TestQwen35GDNChunkedHostContract$' -count=1
go test -tags fak_qwen35_gdn_chunked_host_qualification ./internal/model -run '^TestQwen35GDNChunkedHostContract$' -count=1
```

The compound witness covers nonzero initial state, rectangular dimensions,
distinct Q, grouped heads, two chunks plus tail, exact-zero decay, worker
partition, prefix continuation, refusal without mutation, returned ownership,
normal Q8/mixed-Q4K Prefill markers/logits/KV/conv/recurrent state and next decode.
It does not execute a real model, assert model quality or qualify a compiler,
device, complete suite or other tag combinations. Existing tests are unchanged.

The source gap is the still-serial ordinary host recurrence, **not** a claim it
is today's largest measured bottleneck. An older Mac profile reported about
0.5% in GDN; another historical NVIDIA diagnostic is not current wall ranking.
No new returned controlled measurement supports a speedup here.

Future repository hardware pickup must bind actual source/tree, build tags,
binary hash, model/config/shard hashes and tokenizer, device/driver/power/threads,
exact uncached prompt IDs, computed tokens and complete producer request pins.
Use matched baseline/qualification builds differing only by this build tag,
profile-OFF ABBA
(at least3 samples per arm), fixed threads/cache/residency budgets and P128/P512
plus a nonzero-prefix continuation. Record host prefill wall/TTFT, subsequent
decode, output/state tolerance and token agreement, allocations/RSS, and separate
profile-ON selected/fallback phase coverage. Keep all other opt-in tags fixed.
Report attempted and actually selected cells separately. The reduction may
regress: retain default-off and roll back to the untagged build unless returned evidence qualifies the
whole operation. No hardware request/job, build, test, performance or FULL
acceptance is claimed by this source packet.
