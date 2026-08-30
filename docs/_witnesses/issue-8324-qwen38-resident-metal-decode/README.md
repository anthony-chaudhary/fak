---
title: "Issue 8324 - exact-Qwen3.8 resident hybrid Metal decode HOLD"
description: "Typed HOLD packet for epic 10193 row 6; no performance, quality, or default credit."
---

# Issue #8324 - exact-Qwen3.8 resident hybrid Metal decode HOLD

Verdict: **HOLD_CAPACITY_AND_CLAIM_TOPOLOGY_UNPROVEN**. This packet is a
fail-closed route receipt, not performance evidence. It binds epic #10193 row 6
to base `8fbba932b8128700aef41dd52ab548664a919003` and awards no performance,
quality, topology, or default credit.

## Bound facts

- The exact `Qwen3.8-27B-Q4_K_M.gguf` artifact from
  `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
  is **AVAILABLE_VERIFIED**: 17,106,775,008 bytes, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
  Its private local path is intentionally omitted.
- The observed local host is Darwin/arm64 Apple M3 Pro with 36 GiB
  (`38,654,705,664` bytes), below this packet's sanctioned `>=64 GiB` run
  requirement. No public verified `>=64 GiB` Apple target is bound, and the
  scrubbed private bridge doctor is `NOT_READY`.
- `engine=fak-native`, `runtime=inkernel`, `backend=metal`, and zero fallback
  are mandatory. llama.cpp, MLX, and MLX-LM are prior-art references only; none
  may execute or supply a fallback.

## Mechanism boundary

The base contains #9486 at
`46fdd8a52fd70b3e29345cd311be3cc89443e8fc` and #9488 at
`99ea660ae222dd6a75dd661c54778f470904f9e7`. Together they cover a resident
linear-attention block through its MLP. They do **not** prove that periodic
full-attention layers and dense GDN layers remain device-resident across a
whole token or across all layers.

The current `cmd/modelbench` P=32/T=64 profile also remains pinned to its 36 GiB
M3 Pro envelope. Its claim-grade receipt does not export periodic
full-attention residency, exact multi-token cosine, or matching greedy-token
capture. Therefore a timing capture, dry run, estimate, or simulator cannot
promote this HOLD.

## Acceptance still missing

A replacement receipt must bind one exact matched A/B campaign to all of:

1. resident dense GDN plus periodic full attention across all layers for every
   measured token, with a logged decline for unsupported geometry;
2. exact multi-token cosine `>=0.9999` and matching greedy tokens;
3. the 2.9 tok/s full-prefill and 0.4-1.3 tok/s cached baselines, with a
   candidate at `>=5 tok/s` before any default change;
4. inclusive setup, recovery, prefill, first-token, steady-decode,
   verification, and teardown accounting; and
5. fak-native/inkernel/Metal execution with zero fallback.

## Sanctioned route probe

On an explicitly sanctioned Darwin/arm64 Apple node with at least 64 GiB, set
private paths without printing them and run:

```sh
FAK_SANCTIONED_APPLE_NODE=YES \
GGUF_PATH=/absolute/private/path/Qwen3.8-27B-Q4_K_M.gguf \
OUT_DIR=/absolute/private/output/issue-8324 \
./docs/_witnesses/issue-8324-qwen38-resident-metal-decode/sanctioned-rerun.sh
```

The script checks the host and exact artifact, runs focused mechanism and
receipt-validator tests, builds a VCS-bound `modelbench`, and prints the exact
six-arm P=32/T=64 profile, readback, steady-decode, and end-to-end commands. It
then exits 2 deliberately: the current receipt topology cannot prove the two
missing claim-grade properties above.

Validate this packet locally with:

```sh
cd docs/_witnesses/issue-8324-qwen38-resident-metal-decode
go test ./...
bash -n sanctioned-rerun.sh
```

Prior-art route: **BORROW invariants only** from pinned llama.cpp `ebd048f`, MLX
`43d2f06`, and MLX-LM `cc85215`: graph/state order, caller-owned Metal encoder
lifetime, and GDN recurrent-state semantics. No external runtime or backend is
accepted.
