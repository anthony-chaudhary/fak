# External Repository Study Report: `Mchicao/endless`

## Verdict

**Do not borrow code or open an implementation issue from the pinned revision.** The repository demonstrates one clear mechanism: it keeps an experimental Codex app-server turn open by repeatedly returning operator messages through a typed `wait_for_user_input` tool. That mechanism is interesting as an inspiration-only comparison, but the revision supplies no matched latency, token, quota, quality, or policy evidence. Its host also runs Codex with `danger-full-access` and `approval-policy: never`, so it cannot be transplanted into FAK's capability floor.

This refresh replaces the earlier halted-investigation denominator with a pinned exhaustive tree map and complete non-tree read-back. The machine-readable evidence is [`docs/research/inventory/mchicao-endless.json`](../research/inventory/mchicao-endless.json).

## Scope and provenance

- Repository: <https://github.com/Mchicao/endless>
- Pinned revision: `bbf449ca624aca90d8cda0bc21100f2a73a56bf4` (`chore: init`, 2026-08-21T04:21:48Z)
- Observation date: 2026-08-25
- License: MIT (`LICENSE` and `package.json`)
- Tree inventory: 59 files, 322,763 bytes, seven generated subsystems; `.git` alone was excluded.
- Registry issue: [#9004](https://github.com/anthony-chaudhary/fak/issues/9004), child of [#8936](https://github.com/anthony-chaudhary/fak/issues/8936).

The checked revision is intentionally retained. Remote `main` still points to it. A later `support-windows` commit (`27274d9e…`) exists but is outside the pinned denominator.

## Exhaustive source-class audit

| Source class | Evidence and result |
|---|---|
| README and docs | `README.md` is the entire user guide: install with Bun, run `els`, and accept an explicit warning that experimental Codex APIs may consume quota while waiting. No separate docs tree exists. |
| Architecture and design | Architecture was reconstructed from `src/session/endless-session.ts`, `src/session/prompts.ts`, `src/server/codex-host.ts`, protocol/transport modules, and the TUI. No ADR or design document exists. |
| Runtime source | All 29 files under `src/` were indexed. The session maps a fixed `wait_for_user_input` tool to a deferred inbox; the Codex host converts each response into a tool result and waits for the next tool call. |
| Tests and fixtures | All 13 files under `test/` were indexed. Tests cover JSONL, host events, worker transport, session settings/tabs/usage, install behavior, and TUI rendering/input. No performance, quota, long-idle, policy, or adversarial witness exists. |
| History, changelog, releases | The clone exposes two commits total; the pinned initial commit and one excluded Windows-support commit. GitHub returned zero releases and zero tags. There is no changelog. |
| Issues, PRs, discussions | Exhaustive GitHub read-back returned zero issues, zero pull requests, and zero discussions, open or closed. |
| Roadmap and unfinished-work markers | No roadmap file and no in-code unfinished-work marker was found in the indexed runtime/tests/docs. The README warning is the only explicit limitation statement. |
| License and provenance | Root `LICENSE` is MIT and `package.json` declares MIT. Runtime dependencies are zero; development dependencies are TypeScript and Bun types. Package metadata points at the earlier `maria-rcks/endless` identity while the studied repository is `Mchicao/endless`, so identity provenance should not be inferred beyond the pinned Git history. |
| FAK self-query | Three candidate-specific `fak capabilities` queries found FAK's capability floor, turn-tax avoidance, live-session control, persisted carry-forward, context reuse, and fleet monitoring. They found no native in-provider-turn user rendezvous. |
| Candidate matrix | Three candidates were adjudicated below and recorded in the inventory JSON. None survived as a borrow. |
| Completeness critic | The generated map accounts for every pinned tree file and records all non-tree queries and empty collections. Missing upstream artifacts are reported as absent, not silently skipped. |
| Issue tracking | #9004 owns this denominator refresh. No borrow follow-up was filed because no candidate passed evidence and policy gates. |

## Architecture and execution flow

1. `src/cli.ts` starts the Bun application and constructs an `EndlessSession`.
2. `src/server/endless-server.ts` and worker modules launch and supervise a Codex app-server process.
3. `src/server/codex-host.ts` configures the experimental app-server, registers `wait_for_user_input`, and starts one turn with a generated prompt.
4. `src/session/prompts.ts` instructs the model to call that tool whenever it would otherwise finish.
5. `src/session/endless-session.ts` resolves the pending tool request with the next operator message. The host sends the text back as the tool result, preserving the same provider turn.
6. Session, tab, usage, and TUI modules expose this loop as a terminal product; JSONL and worker protocols isolate process communication.

The causal mechanism is therefore narrow: **a deferred operator inbox is bound to a provider tool call, and the model prompt makes that call the continuation rendezvous**. The surrounding TUI, tab manager, Bun runtime, and worker process are delivery machinery rather than the mechanism.

## Candidate-borrow matrix

| Candidate | Classification | FAK alternative | Decision |
|---|---|---|---|
| `wait_for_user_input` provider-turn rendezvous | Inspiration | Live-session control, turn-tax measurement, stable-context reuse, and persisted session carry-forward | **Reject now.** No matched evidence shows a net gain after idle quota, provider behavior, quality, and policy are counted. The source warning says waiting may consume quota. |
| Bun TUI and tab manager | Direct port | Existing FAK operator and trajectory-control surfaces | **Reject.** Product shell and platform coupling, not a reusable performance mechanism. |
| Codex app-server JSONL host and worker transport | Adaptation | Native gateway/MCP plus guarded workers, leases, and independent witnesses | **Already owned.** FAK's boundary is broader and preserves policy/witness semantics. |

### Ablation that would be required before reconsideration

A future study would need a capability-gated FAK-native experiment, not a source port. Compare the same provider/model/task under:

- baseline multi-turn carry-forward;
- FAK stable-prefix/context reuse;
- one open turn with a typed operator rendezvous.

Count idle and active provider quota, all input/output tokens, latency to accept a follow-up, task quality, cancellation/recovery behavior, and every policy/adjudication event. Remove only the rendezvous while keeping prompt, model, tools, and task fixed. Until that witness exists, the candidate remains rejected rather than filed as speculative work.

## Risks and non-borrows

- `src/server/codex-host.ts` selects `danger-full-access` and `never` approval. FAK must not weaken its default-deny capability floor to reproduce the demo.
- The README explicitly warns that an idle open turn may consume quota; there is no accounting witness.
- The implementation depends on experimental Codex app-server behavior and Bun 1.4, and the pinned package excludes Windows.
- Parallel/tab support and terminal rendering enlarge the product but do not establish the core mechanism's value.
- No releases, issue history, discussions, roadmap, benchmarks, or long-duration tests establish maturity.

## Reproduction

```text
fak study-inventory --root <pinned-clone> --repository Mchicao/endless \
  --url https://github.com/Mchicao/endless \
  --revision bbf449ca624aca90d8cda0bc21100f2a73a56bf4 --json

gh api --paginate --slurp \
  'repos/Mchicao/endless/issues?state=all&per_page=100&sort=created&direction=asc'
gh pr list --repo Mchicao/endless --state all --limit 1000 --json number,title,state
gh api graphql -f query='query { repository(owner:"Mchicao", name:"endless") { discussions(first:100) { totalCount nodes { number title } } } }'

fak study-monitor --inventory-check --json
```

The registry row's `inventory` block binds these observations to the committed map and pinned revision.
