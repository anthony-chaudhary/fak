# CONCEPT-STUDY — Sentry-LLM/SentryLLM (2026-08-08)

- **Source:** https://github.com/Sentry-LLM/SentryLLM
- **Pinned:** `@1ac6b81bfa73be2237e6219e3155d47b74b2b7a2` (single squashed commit,
  `chore: remove all platform-specific references`)
- **License:** MIT (borrow-compatible — moot, see verdict)
- **Lead:** idea-scout #5596, fresh lane, **score 75** — the highest-scoring lead in the
  60-issue open triage queue
- **Pass:** `/scout-loop` → `/study-repo`, one lead

## Verdict — no borrow; the repo is a scaffold

The README is a ~250-line product page (SDK, 10 detection categories, real-time dashboard,
OWASP LLM Top-10 coverage matrix, `[x] Prompt injection detection (v1)` marked shipped).
The tree implements none of it. This is precisely the "borrowed the pitch" failure
`/study-repo` exists to catch, caught by reading the code first.

Evidence, all at the pinned sha:

| Claim (README) | Ground truth in the tree |
|---|---|
| SDK `npm install sentryllm` | `registry.npmjs.org/sentryllm` → **HTTP 404** |
| Analysis engine, 10 detectors | grep `injection\|jailbreak\|exfiltrat\|threat` over all `.ts`/`.tsx`/`.yaml` → **0 source matches** |
| Real-time API | `artifacts/api-server/src/routes/health.ts:6` — `GET /healthz` → `{status:"ok"}`, the only route |
| PostgreSQL storage | `lib/db/src/schema/index.ts:19` — unedited Drizzle template, `export {}` |
| React dashboard | `artifacts/mockup-sandbox/src/App.tsx:118` — shadcn component-preview harness, "Component Preview Server" |
| API surface | `lib/api-spec/openapi.yaml:12` — `/healthz` is the only declared path |
| — | `replit.md:1` — untouched scaffold header, `# [Project name]` |
| CI badge | `.github/workflows/ci.yml:30` — `npm ci`/`npm test` in a **pnpm** workspace, no lockfile, no test script |

## Read coverage + completeness critic

Read: `README.md`, `LICENSE`, `replit.md`, `.github/workflows/ci.yml`, the whole
`artifacts/api-server/src/**` (`app.ts`, `index.ts`, `routes/*`), `artifacts/mockup-sandbox/src/App.tsx`,
`lib/api-spec/openapi.yaml`, `lib/db/src/schema/index.ts`, `lib/api-client-react/src/generated/api.ts`,
`scripts/src/hello.ts`, plus a whole-tree grep for detection logic.

Skipped, justified: `artifacts/mockup-sandbox/src/components/ui/*.tsx` (~55 files) — verbatim
shadcn/ui, generated from `components.json`, not authored here; `pnpm-lock.yaml`; template
`CONTRIBUTING.md` / `SECURITY.md` / issue templates. The generated orval client was checked
rather than skipped: it exports only `healthCheck`, confirming the spec has no other surface.

Critic result: **nothing material left unopened.** 118 tracked files, ~60 of them boilerplate.
No fan-out was warranted — there is one subsystem and it is a health check.

No worldview read was possible: no issues, no discussions, no ADRs, no design docs, and a
single squashed commit. There is no author intent to reconstruct.

## Candidate table

| Borrow | Source `path:line@sha` | Axis | Their-worldview reason | Witness | inspire/integrate | Filed |
|---|---|---|---|---|---|---|
| *(none)* | — | — | no implementing code exists to ground a candidate | n/a | n/a | — |

**Zero candidates is the honest result here** — not a proud read. The dismissal is not
"ours is better": it is that there is no `path:line@sha` in this repo implementing any
advertised technique, verified by whole-tree grep and a 404 on the published package.

## What the pass *did* produce — a fak-seam finding

The lead scoring **75** while being empty is a defect in fak's own crawler, and it is the
durable output of this pass:

- `tools/idea_scout.py:322-394` — `score_candidate()` sums term hits, recency, stars, HN
  points, push-freshness and star velocity. **No term reflects repo substance.**
- `tools/idea_scout.py:821-840` — `fetch_github()` / `fetch_github_fresh()` request
  `fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language`; no size
  or content field is fetched, so substance could not be weighed even if scored.
- Calibration: `NVIDIA/kvpress` (#5328, 1142 stars, real kernels) scored **30**. The hollow
  repo beat it by 45 points, on `very fresh` + `trending (34★/day)` alone.

Dogfood witness on the axis *"does a surfaced lead contain implementing code?"*:
`fak_feature_query` twice (two phrasings) → no card; raw `Grep` over `tools/ internal/ cmd/ docs/`
for `vaporware|substance.gate|readme.only|diskUsage|hollow.repo` → no gate. False-ABSENT
control: `idea-scout` matches 129x in the same index dump, so the index is populated.
Verdict **ABSENT-on-axis** → filed **#5891**.

Distinct from the post-file triage-closure family (#4072–#4076 under #3947), which governs
what happens to a lead *after* it is filed; this is pre-file precision.

## Filed

- **#5891** — `feat(idea_scout): substance gate — penalise README-only repos the fresh/trending lane over-ranks`
- **#5596** — closed *not planned* with the verdict above (studied, no borrow)

## Honest limits

- The witness is lexical and a snapshot — re-witness #5891's ABSENT before acting on it.
- `--depth 1` on a single-commit repo drops no history, but there is also no history to read:
  "they removed this" cannot be assessed.
- Repo size is a proxy for substance, not an oracle — see the sizing caveat in #5891.
- This says nothing about whether the *topic* (`prompt-injection-defense`) is worth mining;
  it says this one lead was not. The fresh lane remains the right lane to watch.

## Companions

[`/scout-loop`](../../.claude/skills/scout-loop/SKILL.md) ·
[`/study-repo`](../../.claude/skills/study-repo/SKILL.md) ·
[`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md) · `tools/idea_scout.py`
