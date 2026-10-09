# Waived-verification follow-up (2026-10-09)

On 2026-10-09, 68 public commits landed under an operator deferred-verification waiver: source-only landing, with compiler, tests and hardware runs deferred. A clean checkout of origin/main `24033b4080f` builds and vets green. Running the touched packages' CPU tests left reds in `internal/model`, `internal/computebuild` and `internal/safecommit`. The follow-up is split by functional goal, not by commit, so separate workers can take one ticket each.

| Ticket | Functional goal | Baseline |
| --- | --- | --- |
| TICKET-01 | V4.1 attention, rotary and indexer numerics | red (parity drift, session panic, shader count) |
| TICKET-02 | V4.1 device kernels qualified on Strix Halo silicon | unwitnessed |
| TICKET-03 | Vulkan memory accounting and residency (#13668) | CPU green; Vulkan-tagged build and hardware not yet witnessed |
| TICKET-04 | Gateway deadline cache credit vs. the Halo long-chat fix | conflict: tests pass, behavior regressed |
| TICKET-05 | Pi launcher/router and garden launchd watchdog | targeted tests not yet run |
| TICKET-06 | safecommit and test-hygiene commits | red (real-git fixtures) |

Each ticket is closed by fail-before/pass-after tests landed on trunk that cite the covered commit shas.
