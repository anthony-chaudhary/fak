# Agent customization field index

**Machine source:** [`agent-customization-index.json`](agent-customization-index.json)  
**Observed:** 2026-08-17  
**Scope:** customization of the agent being authored, its runtime authority, interpretation of its trajectory, and the view presented above that trajectory.

## The important split

“Customize my agent” is not one settings page. It spans four contracts:

1. **Authoring** — identity, instructions, models, tools, skills, memory, and delegation.
2. **Execution** — runtime, authority, hooks, persistence, economics, and privacy.
3. **Interpretation** — source adapters, typed semantic events, derived signals, and redaction.
4. **Presentation** — audience-specific selection, visual language, live controls, and sharing.

The last two are easy to conflate. A parser determines what an event *means* and records provenance/confidence. A presentation chooses what a particular audience *sees*. Hiding a tool result, grouping retries, or calling a phase “investigation” must not mutate the canonical trajectory. The concrete contract is in [`trajectory-presentation.md`](../observability/trajectory-presentation.md).

## Coverage map

The machine index currently records **21 independent axes**. This table is a compact view; query the JSON for full user needs, examples, evidence, status, and disposition.

| Layer | Axes | Current fak gaps to prioritize |
|---|---|---|
| Authoring | instructions, routing, tools, skills, memory, delegation | Composition exists; projection into each native harness remains uneven. |
| Execution | runtime, authority, hooks, persistence, economy, privacy | Session portability and explicit retention/redaction controls are partial. |
| Interpretation | source adapters, semantic events, derived signals, redaction | Codex, AG-UI, Claude Code, and OpenAI export adapters now feed canonical typed events; deterministic derived signals carry source-event and digest receipts. |
| Presentation | audience, selection, visual language, live control, sharing | Declarative audience projections, pre-render redaction, control round trips, sharing bundles, and accessible renderer-neutral visual profiles now ship; full client applications remain outside this index spine. |

## Freshness check

```text
fak customization-index --as-of 2026-08-17
fak customization-index --json
```

The command validates schema shape, unique IDs, source references, status and disposition values, and required axis fields. It also groups axes by layer/status and reports sources older than the declared 30-day review interval. `--as-of` makes tests and historical witnesses deterministic; `--index` checks a candidate registry.
## Reproducible queries

PowerShell:

```powershell
$index = Get-Content docs/research/agent-customization-index.json -Raw | ConvertFrom-Json
$index.axes | Group-Object layer, fak_status | Select-Object Count, Name
$index.axes | Where-Object fak_status -eq absent | Select-Object axis_id, disposition, user_need
$index.axes | Where-Object layer -eq presentation | ConvertTo-Json -Depth 6
```

`jq`:

```bash
jq -r '.axes[] | [.layer,.axis_id,.fak_status,.disposition] | @tsv' docs/research/agent-customization-index.json
jq '.axes[] | select(.layer == "presentation")' docs/research/agent-customization-index.json
```

## Source ledger and honest limits

The source ledger pins repository observations to exact revisions where source is public. Canonical product documentation is dated when no immutable revision is exposed. Sources were selected for distinct evidence, not popularity alone:

- Claude Code, Codex, Cursor, Goose, OpenCode, and Copilot CLI expose broad authoring/runtime choices.
- OpenHands exposes the seam between runtime events and a separate frontend.
- LangGraph makes streaming modes, state, checkpoints, and interrupts explicit.
- AG-UI treats agent-to-UI traffic as a typed event protocol, including messages, tools, state, and activity.

This is a **capability taxonomy**, not a claim that every cited product implements every axis well. `fak_status` is a repository-level self-query classification: `present`, `partial`, or `absent`. `disposition` says whether fak should carry the capability as a default, optional module, recipe, watch item, or exclusion. It does not imply source-code copying. License metadata is retained so later deep-study work can distinguish direct ports from concept-only borrowing.

## Maintenance contract

1. Treat `axis_id` as the stable deduplication key; rename only with an explicit migration note.
2. Before adding an axis, prove it cannot be represented as an example under an existing user need.
3. On a meaningful source refresh, update its `observed_at` and `checked_revision` together; never advance the date without inspecting the source.
4. Every axis needs at least one source ID, a fak status, and a portfolio disposition.
5. When behavior ships, update status in the same commit as the implementation or witness. `present` requires a real path, not a plan.
6. Review every 30 days and whenever a major harness release changes configuration or event protocols.
7. Keep parser semantics and view preferences separate. A view may select and render facts; it may not silently redefine them.

## Value frame

- **For:** people assembling, operating, inspecting, or embedding agents.
- **Problem:** customization requests are scattered across prompt files, runtime config, event schemas, and UI preferences, so “support customization” hides major gaps.
- **Today:** fak has strong harness composition and trajectory recording, but no single maintained map connecting those surfaces and no explicit trajectory-view contract.
- **Better because:** one queryable taxonomy reveals which layer owns a request and prevents UI choices from contaminating evidence semantics.
- **Witness:** parse the JSON, enumerate all four layers, and query the absent presentation axes with the commands above.

