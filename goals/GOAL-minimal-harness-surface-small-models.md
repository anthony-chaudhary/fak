---
loop: goal
goal_slug: minimal-harness-surface-small-models
witness: "cd ..\\fak; go test -count=1 ./internal/agent/... ./internal/codetools/...; go run ./cmd/fak footprint-audit --help"
budget: { max_iters: 12 }
lane: harness
---
# Objective
Give `fak serve --native` a model-aware, budgeted harness surface so small local
models (1–4B class) keep a minimal system prompt + compact tool catalog +
on-demand context — instead of one fixed large surface for every model — reducing
the per-call harness floor (EST ~1K tokens today, tool schemas the largest lever)
and removing attention dilution on small windows. Trigger and report are automatic
via `ctxplan.EnvelopeForModel` tiers, not new flags. Full design, seams, and
cache-identity invariants: `fak-private: docs/notes/2026-09-10-minimal-harness-surface-small-models.md`.

# Non-Goals
- No behavior change on large models: standard tier ships today's exact prompt/catalog bytes.
- No new user-facing tuning flags; `FAK_HARNESS_PROFILE` is a diagnosis override only.
- Do not touch `metaFor`/vDSO read-only keys, kernel tool dispatch, or the frozen ABI.
- No task-quality claim without a live-model weak-tier witness; no `[HW-WITNESSED]` from mocks.
- No loose scripts; all changes are Go in `fak` (public core engine placement).

# Plan
- [ ] 1. **Profile resolution leaf** (`fak: internal/agent/harness_profile.go`)
  - `HarnessProfile{Tier, FloorBudget, ProfileSHA}` + `ResolveHarnessProfile(model)` keyed off `ctxplan.EnvelopeForModel`.
  - Wire model identity through to composition (`runConfig` carries it; unset = standard = byte-identical).
  - Witness: `go test ./internal/agent/... -run HarnessProfile -count=1` in `..\\fak`.
- [ ] 2. **L1 micro-core system prompt** (compact tier)
  - Numbered ~30–50 tok core, positive-only framing (identity / "take every value from a tool result" / stop rule); lint test asserts no do-not/never/avoid tokens; wrap via `BuildOwnedSystemBlock` unchanged.
- [ ] 3. **L2 compact tool catalog** (`fak: internal/codetools`)
  - `CompactCatalog()`: same names, same `ReadOnly` bits, same parameter semantics; required-args-only schemas; one-line descriptions.
  - Kernel-repaired/defaulted parameters keep a one-line mention (small models need the contract stated).
  - Witness: `go test ./internal/codetools/... -count=1` in `..\\fak`.
- [ ] 4. **L3 deferred memory/orientation surface**
  - Digest + AGENTS-style orientation out of the prefix; read-only query tool fetches slices on demand (microcontext posture).
- [ ] 5. **Meter + report**
  - Floor receipt field from `RequestFootprint(DeFoldSystemRequest(...))`; profile tier + reason surfaced wherever acceleration is reported.
  - Compact floor budget: ≤1500 est tok (system ≤150, digest ≤400, tools ≤1000), enforced at composition.
- [ ] 6. **Ablation witness (quality arm)**
  - `fak ablate --models weak,mid` with a no-prompt / minimal / full arm roster (ponytail methodology) on one bounded coding task; cold-vs-warm cache state stated per arm; P50/P90 over N>1 trials.
  - SW-VERIFIED wiring evidence until a live weak-tier model run witnesses task quality.

# Scoreboard & Target Metrics
- Compact-tier harness floor: EST ≤1500 tokens (from ~1400–2400 est today, catalog-dominated).
- Standard-tier drift: 0 bytes vs trunk (byte-identical composition when profile unset).
- Small-window occupancy delta on `*fable*`/`*haiku*` rows: reported, not asserted.
- Task quality: measured only; a regression holds the compact default for that model row (milestone rule 6).

# Done Condition
- Plan items 1–5 green with their package tests in `..\\fak`; boundary check exits 0.
- Ablation arm roster (item 6) filed as its own witnessed ticket if not run in-leaf.
- Report/receipt surfaces show active profile tier + reason.
