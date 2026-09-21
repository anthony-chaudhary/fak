# Fifteen small model-composition tickets

These ranked follow-ons turn the shared FFN pilot (#13435) into narrow, independently verifiable changes. Each owns 1-5 files and has 3-4 steps with deterministic software witnesses. The 34 points are planning estimates, not measured throughput.

Source inspection baseline: `7e70e0e5aa7c4037d326ac57f9b15af9df76bf0b`. Pilot implementation: `360d204b49ca19fa68305847751f19583538adf7`. New tests named here remain work to implement.

| Rank | Issue and local packet | Points | Prerequisite |
| --- | --- | ---: | --- |
| 1 | [#13446](https://github.com/anthony-chaudhary/fak/issues/13446) - [refactor(model): share gated row activation with batched MLP](TICKET-13446.md) | 3 | Ready |
| 2 | [#13447](https://github.com/anthony-chaudhary/fak/issues/13447) - [refactor(model): reuse Gated in GLM shared experts](TICKET-13447.md) | 1 | Ready |
| 3 | [#13448](https://github.com/anthony-chaudhary/fak/issues/13448) - [refactor(model): reuse Gated in Qwen shared expert fallback](TICKET-13448.md) | 2 | Ready |
| 4 | [#13449](https://github.com/anthony-chaudhary/fak/issues/13449) - [refactor(model): reuse Gated in routed expert host fallback](TICKET-13449.md) | 3 | Ready |
| 5 | [#13450](https://github.com/anthony-chaudhary/fak/issues/13450) - [refactor(model): share gated rows in host-batched experts](TICKET-13450.md) | 2 | #13446 |
| 6 | [#13451](https://github.com/anthony-chaudhary/fak/issues/13451) - [refactor(model): share gated rows in tensor-parallel FFN](TICKET-13451.md) | 2 | #13446 |
| 7 | [#13452](https://github.com/anthony-chaudhary/fak/issues/13452) - [refactor(model): share GLM5-Next SwiGLU composition](TICKET-13452.md) | 2 | Ready |
| 8 | [#13453](https://github.com/anthony-chaudhary/fak/issues/13453) - [test(model): benchmark the real V4.1 shared FFN adapter](TICKET-13453.md) | 2 | Ready |
| 9 | [#13454](https://github.com/anthony-chaudhary/fak/issues/13454) - [test(model): cover the FFN adapter activation matrix](TICKET-13454.md) | 2 | Ready |
| 10 | [#13455](https://github.com/anthony-chaudhary/fak/issues/13455) - [refactor(model): isolate exact scalar activation primitives](TICKET-13455.md) | 2 | #13454 |
| 11 | [#13456](https://github.com/anthony-chaudhary/fak/issues/13456) - [refactor(model): isolate the serial RMSNorm reference](TICKET-13456.md) | 3 | Ready |
| 12 | [#13457](https://github.com/anthony-chaudhary/fak/issues/13457) - [refactor(model): share stable scalar softmax outside model](TICKET-13457.md) | 3 | Ready |
| 13 | [#13458](https://github.com/anthony-chaudhary/fak/issues/13458) - [refactor(model): share ordered expert accumulation](TICKET-13458.md) | 3 | Ready |
| 14 | [#13459](https://github.com/anthony-chaudhary/fak/issues/13459) - [refactor(model): reuse ordered accumulation for shared expert merge](TICKET-13459.md) | 2 | #13458 |
| 15 | [#13460](https://github.com/anthony-chaudhary/fak/issues/13460) - [refactor(model): centralize gated FFN fusion eligibility](TICKET-13460.md) | 2 | Ready |

## Dispatch rules

Start with **#13446, #13447 and #13453**: the row primitive, GLM shared adapter, and real shared-adapter benchmark touch disjoint files. Use one bounded worker packet per ticket, with separate test and judge contexts where required. This index does not launch workers.

After #13446 lands, #13450 and #13451 are independent two-file adopters. #13455 waits for activation-matrix #13454. #13459 waits for accumulation #13458.

File overlap overrides the discovery tool's generic dispatchable classification:

- Serialize `moe.go`: #13447, #13448, #13449, #13457, #13458, #13460.
- Serialize `arch.go`: #13446 and #13455.
- Serialize `forward.go`: #13455, #13456 and #13457.
- Serialize `ffn_composition_test.go`: #13453 and #13454.
- #13452 (GLM5-Next) is otherwise independent. Recompute overlaps from actual ticket paths before every wave.

Acquire exact file-tree leases and rebase each later owner after the earlier owner lands. Sharing the model lane does not make overlapping files safe to edit concurrently.

## Small-agent packet

Read the ticket, its pinned source seams, and [the shared FFN contract](../../../../internal/model/ffn/README.md). Each packet fixes file ownership, change boundaries, the named consumer, steps and witnesses. Avoid repeating whole-model discovery. Reuse the existing components and fixtures.

Write independent original-arithmetic expectations. Run the leaf/adapter witness first and confirm every required test exists and executes, then run the owning-package suite. Preserve operation order, float conversions, alias/ownership behavior, typed errors and allocation invariants. Test-only packets retain production code unchanged.

For code-and-test changes, use distinct implementation, test and judge contexts. Follow the normal worktree, review, verification and direct-main landing protocol. Close an issue only after its implementation and acceptance evidence land.

## Why these fifteen

The first seven grow reuse of Gated across scalar, batched and sharded callers. The next five lower the cost of proving activation, normalization and softmax changes, including the missing benchmark and activation matrix. The final three reuse ordered accumulation and separate common fusion eligibility from arithmetic.

Existing onboarding, RoPE, hardware-fusion, full-token oracle and validation-budget issues retain their current ownership. A bounded duplicate review found no exact open matches for these fifteen; #11085 supplies architecture context and #13435 is the completed baseline.

The 10x easier-improvement and 3x composability goals remain targets. Track changed-file scope, duplicated arithmetic removed, live consumers, leaf feedback time versus importer validation, and matched allocation/benchmark results as these tickets land. Ticket counts and extracted file counts do not prove a runtime speedup.

## Planning verification

All fifteen bodies passed the issue schema validator, public scrubber and native discoverability audit before publication. GitHub read-back confirmed fifteen distinct open issues with matching submitted titles/bodies and resolved prerequisites. The creation CLI adds local tracking metadata after publication; that line is retained in each local spec.

This documentation change implements no model code. The tickets' future numerical tests and benchmarks remain unrun.
