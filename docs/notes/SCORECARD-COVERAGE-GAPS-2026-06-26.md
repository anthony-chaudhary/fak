# The top unscored surfaces — a scorecard-coverage gap map (2026-06-26)

**Kind:** coverage backlog (what to score next). **Lane:** `tools` (scorecard family).

fak measures itself with a *family* of ~24 deterministic scorecards (code-quality,
docs, doc-appeal, seo, demo-quality, demo-robustness, repo-hygiene, industry-parity,
agent-readiness, product, persona, stability, steerability, conflation, code-slop,
token-defaults, guard-rsi, dogfood, observability, learning, rsi-maturity, bench-dx,
cuda-dev, tooling-quality). Each is the same machine — a pure KPI set over the
git-tracked tree, cross-checked against reality, folded into a `*-debt` integer + an
A–F grade, retired worst-first by the paired RSI skill. The doctrine is
[`.claude/skills/scorecard/SKILL.md`](../../.claude/skills/scorecard/SKILL.md); the
2×-then-harden loop that drives any of them is
[`/score-2x`](../../.claude/skills/score-2x/SKILL.md).

This note maps the **high-value surfaces that are NOT yet scored** but COULD be, all
deterministically from the tree (no clock, no network, no runtime). The anchor —
**skill-effectiveness** — shipped this pass; the other ten are ranked, ready-to-build
specs. Build each by copying the closest instance (`agent_readiness_scorecard.py` for a
tree-reading card, `product_scorecard.py` for a catalog card) and re-pointing it.

## Anchor (SHIPPED 2026-06-26)

**0. skill-effectiveness** — is each `.claude/skills/*/SKILL.md` BUILT to be effective?
Nine KPIs over discover / operate / trust / economy; debt key `skill_debt`.
Tool: [`tools/skill_effectiveness_scorecard.py`](../../tools/skill_effectiveness_scorecard.py);
skill: [`/skill-score`](../../.claude/skills/skill-score/SKILL.md);
snapshot: [`docs/SKILL-EFFECTIVENESS-SCORECARD.md`](../SKILL-EFFECTIVENESS-SCORECARD.md).
Baseline at ship: 11 skill-debt over 31 skills, score 92.7 (grade A) — the
commit-discipline cluster + dead-reference + trigger gaps are the first `/skill-score`
targets. The ungameable anchor KPI (`refs_resolve`) cross-checks every cited path
against disk, the way `agent_readiness` resolves its links. This is the surface no
other scorecard graded: the pack measuring *itself*.

## The ranked backlog (10 unscored surfaces, highest value first)

Each row: **what it measures** · why it matters (evidence) · the deterministic signal ·
the proposed `*-debt` unit · shape (tree-reading vs catalog).

**1. type-contract-docs** (`contract_debt`, tree-reading) — does each stability-critical
`internal/<pkg>` document its interface contract (role, invariants, caller guarantees)?
*Evidence:* ~136 internal packages; `*contract.go` exists in a few (`browseraction/`)
but inconsistently — agents integrating fak need the API contract before calling. *Signal:*
top-of-package docstring present + ≥2 sentences + invariant/guarantee language; cross-check
against `*contract.go`/`*interface.go`. *Unit:* one package with no clear contract docstring.

**2. refusal-taxonomy-coverage** (`taxonomy_debt`, tree-reading) — is every refusal
ReasonCode defined, documented, tested, and invoked consistently? *Evidence:* fak's core
contract is a CLOSED refusal vocabulary (`internal/abi/reasons.go` consts +
`dos.toml [reasons.*]`); an undocumented/untested code breaks the agent+operator contract.
*Signal:* for each ReasonCode — in `coreReasonNames`? a `dos.toml` block? a witness test?
a public-doc example? *Unit:* one reason missing doc, test, or example. (Low effort, grep-based.)

**3. benchmark-catalog-completeness** (`benchcat_debt`, catalog) — are the ~40 `cmd/*bench*`
binaries discoverable, runnable, and documented as products, not orphaned code? *Evidence:*
"fak is fast" is empty without a runnable catalog; `BENCHMARK-AUTHORITY.md` + `bench-dx`
exist but which benchmarks are official/linked? *Signal:* declared roster row ⇒ `cmd/<name>`
dir + doc entry + runnable example + link from the authority doc; any undeclared `cmd/*bench*`
is a coverage gap. *Unit:* one benchmark missing a cmd dir, doc, example, or link.

**4. witness-test-completeness** (`witness_debt`, tree-reading) — does each correctness
claim have a witness? *Evidence:* fak claims "deterministic / fair / safe / invariant";
~35 `*_witness_test.go` exist, but `demo-robustness` scores witness USE, not COVERAGE.
*Signal:* parse CLAIMS.md `[SHIPPED]` lines mentioning determinism/correctness/fairness/safety;
each should map to a `*_witness_test.go`/`*_proof*.go` in its package. *Unit:* one such claim
with no witness test.

**5. integration-platform-coverage** (`integration_debt`, catalog) — is each platform
integration (MCP, Claude Code, Cursor, OpenAI, Anthropic) documented, exampled, and tested?
*Evidence:* adoption depends on integrations; `examples/mcp/`, `docs/integrations/`,
`DOGFOOD-CLAUDE.md` exist — but does each have a runnable example + an integration test +
a config (`.mcp.json`/`.cursorrules`)? *Signal:* per declared integration, resolve doc +
example dir + `*_integration_test.go` mention + config; live-code integration with no row =
gap. *Unit:* one integration missing doc, example, test, or config. (Distinct from
agent-readiness, which scores agent FRICTION, not integration completeness.)

**6. lint-rule-architectural-coverage** (`lintcov_debt`, tree-reading) — do the linters
(`codelint`, `toollint`, `urllint`, `boundarylint`, `pathlint`) enforce the documented
architectural invariants at lint time, not just at test time? *Evidence:* fak has strong
rules (`ARCH_LAYER_VIOLATION`, `OUT_OF_DIRECTION`); a rule only caught by `architest` is
slower, noisier feedback. *Signal:* extract each linter's rule set; cross-check against
documented invariants (`ARCHITECTURE.md`/`DIRECTION.md`/`PARTITION.md`/`dos.toml` lanes);
*Unit:* one documented invariant no linter enforces.

**7. performance-sla-docs** (`sla_debt`, tree-reading) — are latency/throughput SLAs stated
and witnessed? *Evidence:* "the gateway is fast" needs an explicit bar + a benchmark that
proves it; baselines exist (`docs/baselines/`, `docs/webbench-baselines.md`) but SLAs aren't
always stated/linked. *Signal:* detect SLA strings (`p99`, `< X ms`, `> Y req/s`) in
CLAIMS.md/docs; each should link a committed baseline or witness. *Unit:* one SLA claim with
no supporting baseline/benchmark.

**8. metrics-export-completeness** (`metrics_debt`, tree-reading) — are exported metrics
comprehensive and documented? *Evidence:* fak runs as a gateway operators must observe
(`internal/metrics/`, `internal/gateway/metrics.go`, `/metrics`); are all metrics in
`docs/.../observability`? *Signal:* parse metric definitions (name/type); cross-check against
the observability doc. *Unit:* one metric defined-but-undocumented OR documented-but-absent
(overclaim). (Complements the existing `observability` scorecard, which grades the dashboards/
alerts surface, not metric-definition parity.)

**9. example-persona-coverage** (`example_debt`, catalog) — does each key persona
(coding-agent, operator, policy-author, researcher, integrator) have a runnable end-to-end
example? *Evidence:* `examples/` has 25+ subdirs; `product`/`persona` grade concept maturity
and who-is-served, not whether each persona has a "my first example" path. *Signal:* per
(persona × example-type) cell, resolve an `examples/<name>/README.md` with a copy-pasteable
command + real use case; unmapped example dir = gap. *Unit:* one (persona, example-type) cell
with no runnable example.

**10. upgrade-migration-coverage** (`migration_debt`, tree-reading) — are version upgrades
and state migrations documented and tested? *Evidence:* fak carries state (session images,
leases, policies); `docs/migration-guide.md` + `docs/ROLLBACK.md` exist, but are upgrade
scenarios witnessed? *Signal:* detect migration/upgrade/rollback docs; each documented path
should map to a test/script + a breaking-change notice. *Unit:* one documented migration with
no test/example.

## Notes on method

- All ten are **tree-reading** or **catalog + tree-verification** — deterministic from the
  git tree alone, no runtime, so two clones at one commit score identically (scorecard law #1).
- None duplicate an existing card: type-contracts ≠ code-quality; refusal-taxonomy ≠
  code-slop; benchmark-catalog ≠ bench-dx (which scores benchmark *DX*); integration ≠
  agent-readiness (agent friction); example-coverage ≠ product (concept maturity);
  metrics-export ≠ observability (dashboards/alerts).
- Build order ≈ value order above; each new card folds into `tools/scorecard_control_pane.py`
  (`SCORECARDS`) and re-pins the baseline (doctrine step 7). `/score-2x` then drives each 2×.

**Status tags:** skill-effectiveness `[SHIPPED]`; areas 1–10 `[PROPOSED]` (designed,
not built). This note is the work-list; building any one is a self-contained `scorecard`
doctrine pass.
