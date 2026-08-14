# Concept study — the Reddit "token-saving tools for Claude Code" cluster (rtk · distill · claude-token-guard · 9router · Headroom) → witnessed borrows for fak

Study date **2026-07-16**. Trigger: a user pointed at a Reddit thread naming several "cut your Claude Code token bill" tools — and specifically flagged **rtk** as *"more like a joke/satire than a real thing,"* while raising the genuinely interesting question of whether the **output-style** idea (make the model itself terse via the system prompt) is worth adopting in fak, plus a request to study **Headroom** further for *coexistence* ("respect it and users") and *native* deepening.

Method: two fanned-out research subagents over pinned clones/APIs (rtk+distill+ctg+9router; Headroom CLI+desktop), each finding **code-grounded**, then every candidate **witnessed on-axis** against the fak seam (`fak_feature_query`/`fak index` + raw `Grep` + reading the source), classified at the property grain per the [`study-repo`](../../.claude/skills/study-repo/SKILL.md) discipline. All borrows below are **INSPIRE / clean-room Go** — no bytes vendored.

## Sources (pinned)

| Tool | Repo | Pinned | License | Real? |
|---|---|---|---|---|
| rtk (Rust Token Killer) | `rtk-ai/rtk` (a.k.a. `TaKO8Ki/rtk`) | `5d32d07` | Apache-2.0 | **Real** — 126 Rust files, Homebrew formula, tagged releases |
| distill | `samuelfaj/distill` | `4e2eaac` | MIT (CLI pkg) | **Real** — TS source, local 1.7B MLX model |
| claude-token-guard (`ctg`) | `kaviadigdarshan/claude-token-guard` | `8fa54df` | MIT (README) | **Real** — small JS "ESLint for your setup" |
| 9router | `decolua/9router` | `bc252ea` | MIT | **Real** — large Node proxy/router |
| Headroom (CLI) | `headroomlabs-ai/headroom` (was `chopratejas/headroom`, 301-redirects) | `718c8dc` | Apache-2.0 | **Real** — Python core + Rust proxy |
| Headroom Desktop | `gglucass/headroom-desktop` | `cbf68e0` | MIT shell (paid app) | **Real** — Tauri tray wrapper over the CLI |

### Satire verdict (the user's flag)
**rtk is genuinely real software, not a gag.** The "satire smell" is explained by its **wildly disproportionate ~71k star count + aggressive "-90%" marketing** (compare distill 649, ctg 12) — the *popularity signal* looks astroturfed, but the *code* is a full Rust codebase with per-command filter modules, a Bash `PreToolUse` hook, and releases. Treat rtk's popularity skeptically; treat its mechanism as real. (9router's 22k stars carry a similar grey-hat "unlimited free / multi-account" framing worth the same skepticism.)

## The one idea the field is really selling, and where fak already sits

Strip the theatrics and these tools attack the token bill on four distinct axes. fak already owns three of them; the fourth (output-style) is the user's actual question.

| Axis | Field tool | fak on-axis | Verdict |
|---|---|---|---|
| Tool-**output** compression (intercept oversized tool_result) | rtk (hook line-drop), 9router ("RTK Token Saver" at proxy), Headroom (`SmartCrusher`/CCR) | `internal/headroom/native.go` + the `Compressor` seam; rtk borrows already filed #5011/#5012/#5014; LLMLingua-2 #3204 | **PRESENT** — see rtk study note |
| **Output-style** (make the model emit fewer tokens via system prompt) | **distill `/distill` DSL** (`skills/distill/SKILL.md@4e2eaac`) | `internal/syspromptmmu/steering.go` (#3308) — **producer only, unrouted** | **PARTIAL → filed #5047/#5051** |
| Multi-account routing / tiered fallback | 9router (round-robin, tiered) | `internal/fleetaccounts/*` + `accounts/rotation.go` (capacity/reset-aware) + `internal/modelroute/*` | **PRESENT/ahead** (ToS-respecting; 9router's "unlimited free" is DIVERGENT) |
| Config token-bloat audit | **ctg `audit`** (10 anti-patterns) | `RequestFootprint` (`internal/agent/anthropic_footprint.go`) + `internal/mcpfootprint/*` measure it; no user-facing audit verb | **PARTIAL → filed #5050** |

## Filed this pass (5 leaves, all `inspire`)

| # | Title | Axis / seam | Parent | Prior art |
|---|---|---|---|---|
| **#5047** | Wire cache-safe terseness steering to the live request path | the deferred #3308 rung; `steering.go:86` producer → `overlay.go` splice | epic **#1258** | distill `/distill` |
| **#5051** | User-facing "style" selection surface (blocked by #5047) | named style→level UX; `steering.go` closed-set | epic **#1258** | `distill/skills/distill/SKILL.md@4e2eaac` |
| **#5048** | Detect a Headroom proxy in front & coexist (no double-compress, no base-URL clobber) | `bridge.go:155` `Reachable()` → fingerprint; `admit.go` coexist gate | seam (headroom) | `headroom/cli/doctor.py@718c8dc` `check_claude_routing()` |
| **#5049** | Preserve tool-schema deferral when repointing `ANTHROPIC_BASE_URL` (`ENABLE_TOOL_SEARCH` guard) | `guardInjectedEnv` (`guard_provider.go:286`) omits it; composes with `--defer-cold-tools` #3232 | epic **#3229** | Headroom **GH #746** |
| **#5050** | User-facing session-config token-bloat audit over `RequestFootprint` | thin aggregator over the floor + `mcpfootprint` + #5049 | epic **#2063** | `ctg audit@8fa54df` |

### The load-bearing new findings
1. **fak and Headroom are both `ANTHROPIC_BASE_URL` interposers** → they collide on who owns that env var, and if they chain, both compress the same tool_result (native gate + `SmartCrusher`) plus Headroom's per-request CCR injection. On short/code-heavy sessions the stacked overhead can *exceed* the saving — the code-grounded explanation for the field report that "Headroom **increased** my tokens ~30%" (Headroom's own `limitations.mdx@718c8dc`: <300-tok messages "overhead exceeds savings"; short exchanges median 4.8%). → **#5048**.
2. **Headroom's GH #746 is a bug fak is exposed to.** A custom `ANTHROPIC_BASE_URL` with `ENABLE_TOOL_SEARCH` unset makes Claude Code **materialize every MCP/system tool schema into context** — ~35.8k of the ~41k fresh-session floor (fak's own number, `guard.go:134`). fak repoints the base-URL (`guardInjectedEnv`) but sets no `ENABLE_TOOL_SEARCH`, and its server-side compensator `--defer-cold-tools` (#3232) is **default-off** (gated on #3537). So a default `fak manage -- claude` may silently pay the full floor. The env guard is the cheap, cache-safe, complementary fix. → **#5049**.
3. **The output-style idea has real prior art** (distill's `/distill` DSL) and fak already has the *cache-safe* producer for it (#3308) sitting **unrouted** — its own scope fence admits "no request-path caller today." fak's leveled prose terseness is deliberately the *calibrated* version of distill's format-changing DSL. → **#5047** (wire) + **#5051** (name it).

## DIVERGENT ledger — earned dismissals (recorded, not filed)

- **rtk hook command-rewrite + filter-trust model** — already earned-dismissed in the [rtk study note](CONCEPT-STUDY-RTK-2026-07-16.md): fak adjudicates in-loop, it does not rewrite commands, so rtk's static allow/deny lexer + SHA-trust gate are prerequisites fak's model doesn't need.
- **9router "unlimited FREE" via mass multi-account rotation** — the grey-hat core (rotate across free/subscription providers + mass-account auto-login helpers to evade limits) is a ToS posture fak explicitly does **not** take. fak's `fleetaccounts` is *legitimate* capacity/reset-aware rotation, not limit-evasion. DIVERGENT — drop the framing, keep nothing.
- **Headroom Desktop's paid Tauri tray + managed 2 GB Python runtime + watchdog** — an app-distribution concern (install/auto-update/watchdog the proxy), not a fak kernel concern. Its only fak-relevant bit — that it interposes on **port 6767** and *also* installs a Claude `PreToolUse` rtk hook — is folded into #5048's detector (recognize both the :8787 CLI and the :6767 desktop variant).
- **distill's local 1.7B MLX expert-model output compressor** — a heavier variant of the token-level compression #3204 already tracks behind the `Compressor` seam; not separately filed (the seam exists; a local-LLM plugin is a #3204-family choice, not a new axis).
- **ctg's unverifiable house rules** (P2 "no resume-after-rate-limit rule", P3 "no turn-counter hook") — opinionated config prescriptions, not measurable token facts; #5050 adopts only the anti-patterns fak can *measure or verify*, not these.

## PRESENT ledger — witnessed "already have it on-axis" (dropped)

- **Tool-output compression** — `internal/headroom/native.go` + `Compressor` seam; rtk/9router/Headroom-SmartCrusher borrows already filed (#5011/#5012/#5014, #3204). PRESENT.
- **Multi-account rotation + tiered model routing** (9router's core) — `internal/fleetaccounts/*` (apextier/capacity/authsignals/capstate), `internal/accounts/rotation.go`, `internal/modelroute/*`. fak's rotation is capacity- and reset-aware (`headroomtier.go` three-way offerability), strictly more principled than round-robin. PRESENT/ahead.
- **Usage-limit understanding** — `internal/accounts/headroomtier.go` (offerability tier + reset-soonness), `cmd/fak/usage.go`, `cmd/fak/cachevalue.go`. PRESENT.
- **Context hygiene** (clear at %, keep window small, fewer MCPs/skills) — managed-context planning + `internal/mcpfootprint` floor gate + capability paging. PRESENT.
- **Token accounting substrate** — `RequestFootprint` floor split (system/tools/history), provenance-labeled ESTIMATED (#2924). PRESENT; #5050 is the *user-facing audit* on top, not a re-measure.

## The "headroom" triple-overload (disambiguation — this bit people)

The word **headroom** names three unrelated things in fak; keep them straight when reading the tickets:
1. **Compression control surface** — `internal/headroom/*` (the tool-output compressor gate + the Headroom-proxy bridge). ← #5048 lives here.
2. **Usage/quota headroom** — `internal/accounts/headroomtier.go` (how much rate-limit room an account has). ← the rotation axis.
3. **Resume-attempt headroom** — `internal/resume/headroom` (retry budget). Unrelated to both.
The external **Headroom** product (`headroomlabs-ai/headroom`) maps onto sense (1) only.

## Honest limits

- Witness is lexical + a snapshot (substring rankers; verdicts true as of 2026-07-16). Every PARTIAL/ABSENT was confirmed by reading the fak seam, not a ranker miss.
- The "Headroom increased tokens 30%" cause (GH #746 + CCR overhead) is **code-grounded inference**, not an official Headroom post-mortem — no doc names the Cursor+increase case directly.
- Star counts are as the GitHub API returned them; not independently corroborated (and, per the satire verdict, not to be trusted as a quality signal).
- License reads are good-faith; every borrow is INSPIRE clean-room Go regardless, so no vendor decision rides on them.

## Companions

- Skill: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) → [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md).
- Sibling study: [`CONCEPT-STUDY-RTK-2026-07-16.md`](CONCEPT-STUDY-RTK-2026-07-16.md) (the rtk deep pass; borrows #5011/#5012/#5014).
- Prior Headroom pass: the 2026-07-08 headroom study (borrows #3307/#3308/#3339–#3343) — this pass adds the *coexistence* + *GH#746* axes that pass did not cover.
