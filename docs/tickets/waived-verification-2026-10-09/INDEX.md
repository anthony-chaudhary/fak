# Waived-verification follow-up (2026-10-09)

On 2026-10-09, 68 public commits landed under an operator deferred-verification waiver: source-only landing, with compiler, tests and hardware runs deferred. A clean checkout of origin/main `24033b4080f` builds and vets green. Running the touched packages' CPU tests left reds in `internal/model`, `internal/computebuild` and `internal/safecommit`. The follow-up is split by functional goal, not by commit, so separate workers can take one ticket each.

| Ticket | Functional goal | Baseline |
| --- | --- | --- |
| TICKET-01 | V4.1 attention, rotary and indexer numerics | red (parity drift, session panic, shader count) |
| TICKET-02 | V4.1 device kernels qualified on Strix Halo silicon | unwitnessed |
| TICKET-03 | Vulkan memory accounting and residency (#13668) | CPU green; Vulkan-tagged build and hardware not yet witnessed |
| TICKET-04 | Gateway deadline cache credit vs. the Halo long-chat fix | conflict: tests pass, behavior regressed |
| TICKET-05 | Pi launcher/router and garden launchd watchdog | green: 11/11 verified (501095431bb, 8fcf7b7e995) |
| TICKET-05a | Pi launcher qwen-prefix window row vs router-reported window (#13778) | open: launcher 131072 vs router-derived 163840 for cloud Qwen |
| TICKET-06 | safecommit and test-hygiene commits | red (real-git fixtures) |
| TICKET-07 | FP8 tiled-to-Q8 loader | red |
| TICKET-08 | V4.1 compressed-window composition + session fallback | red |
| TICKET-09 | V4.1 sparse-sink host contract | red |
| TICKET-10 | V4.1 mHC geometry: absent-mix refusal and lost layer-two stream (#13790) | red |
| TICKET-11 | V4.1 clamped SwiGLU failure-retry panic aborts internal/model, fix first (#13795) | red |
| TICKET-12 | V4.1 compressor/indexer projection fault-closure contract (#13796) | red |
| TICKET-13 | V4.1 layer-step, grouped-session and head parity drift, BF16 boundaries (#13797) | red |

Each ticket is closed by fail-before/pass-after tests landed on trunk that cite the covered commit shas.
