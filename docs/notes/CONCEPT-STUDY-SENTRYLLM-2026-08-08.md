# CONCEPT-STUDY — Sentry-LLM/SentryLLM (2026-08-08)

- **Source:** https://github.com/Sentry-LLM/SentryLLM
- **Pinned:** `@1ac6b81bfa73be2237e6219e3155d47b74b2b7a2` (single squashed commit,
  `chore: remove all platform-specific references`)
- **License:** MIT (borrow-compatible — moot, see verdict)
- **Lead:** idea-scout #5596, fresh lane, **score 75** — the highest-scoring *repo-shaped*
  lead in the open triage queue (~116 open `idea-scout` issues; next repo-shaped is 58/#5375).
  Not the highest score overall — #4629 scores 77.
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
| PostgreSQL storage | `lib/db/src/schema/index.ts:20` — unedited Drizzle template, `export {}` |
| React dashboard | `artifacts/mockup-sandbox/src/App.tsx:104` — shadcn component-preview harness, "Component Preview Server" |
| API surface | `lib/api-spec/openapi.yaml:14` — `/healthz` is the only entry under `paths:` (`:13`) |
| — | `replit.md:1` — untouched scaffold header, `# [Project name]` |
| CI badge | `.github/workflows/ci.yml:28,37` — `npm ci`/`npm test` in a **pnpm** workspace, no lockfile, no test script; the root `preinstall` hook exits 1 for a non-pnpm agent, so `:28` fails first |

**"Scaffold-only", not "README-only"** — the tree has 81 `.ts`/`.tsx` files and ~7,000 lines.
It is a generated monorepo skeleton with none of the advertised behaviour, which is a
*harder* case for a size-based filter than an empty repo, and the reason #5891's floor is
scoped to catch scaffolds rather than to rank real repos.

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

- `tools/idea_scout.py:322-396` — `score_candidate()` sums six terms: term hits, recency,
  stars, HN points, push-freshness and star velocity. **No term reflects repo substance.**
- `tools/idea_scout.py:824` and `:836` — `fetch_github()` / `fetch_github_fresh()` request
  `fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language`; no size
  or content field is fetched, so substance could not be weighed even if scored.
- `tools/idea_scout.py:457-461` — `parse_github_repos()` copies a **three-key allow-list**
  into `extra` (`stars`, `pushed_at`, `language`, at `:458-460`). Widening the `--json` list
  alone is a silent no-op: a new field is dropped here before it can reach the scorer. This
  is the step a fix most easily misses.
- **The Go port must move with it.** `internal/ideascout/fetch.go:57,67` carries a
  byte-identical `--json` field list, and `internal/ideascout/testdata/source_corpus.json`
  cross-pins the two (read by `internal/ideascout/ideascout_test.go:960` *and*
  `tools/idea_scout_test.py:902`). A Python-only change is the #5549 divergence class.
- Calibration: `NVIDIA/kvpress` (#5328, 1142 stars, real kernels) scored **30**. The hollow
  repo beat it by 45 points, on `very fresh` + `trending (34★/day)` alone — 0 of the 75
  points came from stars.

Dogfood witness on the axis *"does a surfaced lead contain implementing code?"*:
`fak feature query` twice (two phrasings) → no card on the axis; raw `Grep` over
`tools/ internal/ cmd/ docs/` for `vaporware|substance.gate|readme.only|hollow.repo` → no
gate. False-ABSENT control (re-run 2026-08-08 against the 1.05 MB `-json` catalog dump):
`idea-scout`/`idea_scout` match **85×**, so the index is populated; the only substance-term
hit in the whole dump is this note's own `INDEX.md` line. Verdict **ABSENT-on-axis** →
filed **#5891**.

Scope note: this is *pre-file* precision — the scorer admitting a hollow repo — whereas the
triage-closure family (#4072–#4076 under #3947) governs what happens to a lead after it is
filed. The two are adjacent, not disjoint: a stronger post-file closure would also have
retired #5596, just later and by hand.

## Filed

- **#5891** — `feat(idea_scout): substance gate — a repo-size admission floor so scaffold
  repos cannot ride the fresh/trending lane`. Filed to the worker-ready contract (27/27
  template fields, `generation`/`gen/now`, milestone *Generation G0*, lanes `tools` +
  `ideascout`). It proposes an **admission floor** on `size` — not a `W_THIN_REPO` score
  penalty, which the six-summand arithmetic makes unsatisfiable: freshness + trending alone
  reach 75, so no bounded negative weight can sink a scaffold below a real repo without
  also sinking real ones.
- **#5596** — closed *not planned* with the verdict above (studied, no borrow)

**Two corrections to the first draft of #5891**, both caught by an adversarial
re-verification pass before dispatch and worth recording because they are the reusable
lesson: (1) the first draft named `diskUsage` as the field to add — that is a GraphQL
`Repository` field, **not** a `gh search repos --json` key; the correct key is `size`, and
the wrong name would have made `gh_json()` (`tools/idea_scout.py:815`) exit non-zero and
kill **both** GitHub lanes on the first run. (2) The draft carried the `research` label,
which `internal/issuecontract/contract.go:1305-1317` `isDispatchLeaf()` deny-lists — the
ticket was permanently undispatchable while reading as filed.

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
