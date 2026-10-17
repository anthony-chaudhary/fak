# OpenCode free models — auto-update design note (2026-10-16)

## Problem

OpenCode's free model roster changes often (renames, retirements, new entries:
`ll-alpha-free`, `space-bunny-free`, `beam-free`, …). FAK ops hardcodes these ids in
three places:

1. `opencode.json` (tracked repo root) — the spawn config FAK ops workers run under.
2. `~/.fak/router/routing.json` (live, operational) — the FAK router routing table.
3. `internal/modelroute` (`OpenCodeGoOxAlphaModel`, `DefaultFreeTierModels`, `DefaultFreeTierRoster`) — the free-tier account roster.

When OpenCode renames or retires a model, every hardcoded reference drifts and FAK ops
fails with `provider.no-route: Model unavailable: fak/<model>` (observed 2026-10-16 for
`fak/ll-alpha-free`). The discovery spine below removes the staleness.

## Discovery (shipped)

Source of truth for the native `opencode` provider: the `opencode models` CLI, which
lists one `provider/model` id per line. The OpenCode Go gateway additionally exposes an
OpenAI-compatible `GET https://opencode.ai/zen/go/v1/models` (no key required) for the
`opencode-go` tier.

Discovery is implemented in:

- `internal/modelroute/opencode_free_models.go` — pure, testable:
  - `ParseOpenCodeFreeModels(lines, suffix)` — filters to `*-free` suffix, parses
    `provider/model`, runs the quality gate.
  - `SplitOpenCodeFreeModels` — partitions survivors vs rejected.
- `fak opencode free-models [--json] [--suffix] [--provider] [--all] [--check] [--write PATH]` — runs
  `opencode models`, applies the gate, prints survivors (plain) or full JSON.
  - `--check`: freshness gate; exit non-zero on any rejected free model or empty discovery.
  - `--write PATH`: persist the discovered survivor roster as JSON for downstream apply tools.
- Tests: `internal/modelroute/opencode_free_models_test.go` (PASS).

Quality gate reuse: `FreeQualityReason` (deny tokens `compound/whisper/tts/embed/1b/tiny/nano`,
tool-capable floor, context floor) — so a renamed-but-still-a-toy model is rejected loud.

## Auto-update approach (spine → apply → cadence)

1. **Discover** (shipped): `fak opencode free-models --json` emits the current
   quality-gated free roster. Machine-readable, suitable for piping.
2. **Apply** (fan-out): a follow-on `fak opencode free-models --apply` (not yet shipped)
   that rewrites the three hardcoded surfaces from the discovered roster:
   - `opencode.json` provider entries for `opencode`/`opencode-go` free models.
   - router `routing.json` routes (backend `opencode-go` for OpenCode Go models).
   - `internal/modelroute` constants/roster (regenerate `OpenCodeGoOxAlphaModel`-style
     bindings; keep the env-var-only credential indirection).
   Apply must be a diff the operator reviews before commit (no silent mutation of
   committed config).
3. **Cadence** (fan-out): hook discovery into the existing `fak-sync ops ... watch`
   pattern (e.g. `fak-sync ops opencode-free-models watch --apply`) or a cron lane,
   so the roster refreshes on a schedule and opens a PR/diff instead of failing at
   dispatch time. Reuse `fak opencode free-models --json` as the probe.
4. **Verification** (fan-out): a tracked test that pins the discovered roster against
   `opencode models` (skip in CI when the CLI is absent), so drift fails loud in
   the operator's environment before a spawn dies.

## Why not scrape HTML / hardcode a refresh list

- `opencode models` is the authoritative, version-matched source (it reflects what the
  installed CLI can actually reach). Scraping docs would lag the CLI.
- The gateway `/v1/models` is the fallback for the `opencode-go` tier and needs no key.
- Hardcoding a "latest known" list is exactly the drift we are fixing.

## Fan-out follow-ons (not in this spine)

- `--apply` flag + config rewriter (tracked `opencode.json` + router routing).
- Cadence wiring (`fak-sync ops` watch or cron).
- Tracked freshness test (discovered vs committed roster).
- Dashboard/ops surface showing the discovered free roster + last-refresh time.
- Migrate the stale `OpenCodeGoOxAlphaModel = "ox-alpha-free"` constant to the
  discovered canonical name (`ll-alpha-free`) once the apply step lands.

## Witness

- `go test ./internal/modelroute/ -run 'OpenCodeFreeModels' -count=1` → PASS
- `fak opencode free-models --json --provider opencode` → lists the live free roster
  (incl. `opencode/ll-alpha-free`) with quality verdicts.
EOF
echo written