# Study: Caveman — shape-specific context transforms with measured fallback

**Observed:** 2026-08-13  
**Source event:** upstream HEAD `c72984e4392c7a154e55c11dbf445f01ce5c35d4` committed 2026-08-13T17:27:59+02:00  
**Source state:** public GitHub repository, default branch `main`, exact pinned revision  
**Platform context:** static source study on Windows; no Caveman binary was executed and no upstream performance claim is adopted as a fak claim  
**Refresh trigger:** upstream engine license/change date, a material `engine/compressors`, `engine/pixel`, or `engine/ccr` release, or closure of filed issues #6668–#6670

- Repository: <https://github.com/juliusbrussee/caveman>
- Pin: [`juliusbrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`](https://github.com/juliusbrussee/caveman/tree/c72984e4392c7a154e55c11dbf445f01ce5c35d4)
- Release context: the pin is sixteen commits after `v2.0.0` (`2c67abb9833689b48c7abba88afaa77c39a18657`); releases from `v1.0.0` through `v2.0.0`, recent history, open/closed issues, PRs, and discussions were inspected.

## Value frame

- **For:** fak operators running long, tool-heavy agent sessions.
- **Problem:** generic budgeting and paging preserve correctness but can miss representation-specific token reductions before a body must be evicted.
- **Today:** fak owns deterministic page-out/page-in, CAS stubs, TOON work, schema deferral, image ingestion, and provider cache accounting; it does not yet choose among conservative output codecs or text-versus-image representations at those seams.
- **Better because:** one narrow transform can be admitted only when it is reversible or quality-witnessed and net-positive against fak's current path.
- **Witness:** fixture-level fidelity plus a real fak corpus ablation; no upstream headline percentage is repeated as fact.

Problem centrality is **Enabling**. P1 managed context is direct; P2 net-true efficiency requires tuned-baseline ablation; P3 adaptation stays bounded through typed gates and raw fallback; P4 operations require visible decisions and refusal reasons. The smallest spine for each surviving borrow is therefore an offline or one-shape witnessed gate, not a broad compression engine.

## What Caveman is optimizing

Caveman targets developers whose coding agents repeatedly carry large tool outputs and tool schemas. Its v2 engine treats compression as a portfolio of shape-specific transforms rather than one summarizer: structured data, search results, terminal/log output, code, and dense text each receive a separate applicability decision. Tests emphasize adversarial retention, round-trip behavior, explicit thresholds, and fallbacks. The repository also exposes memory, cache, proxy, browsing, MCP, UI, and packaging surfaces around that engine.

That worldview differs from fak's primary one in degree, not kind. Caveman optimizes immediate provider-token reduction across popular harnesses; fak optimizes a kernel-owned, auditable context lifecycle where cache survival, reversibility, and net value are first-class. Borrowing therefore means adding measured transforms at fak's existing ownership seams, not importing Caveman's engine or its claims.

## Coverage and completeness critic

The study read the README and honest-number/benchmark docs as maps, then inspected implementation and tests across:

- `engine/compressors`: JSON/TOON dispatch, tool-schema annotations, search-result, terminal, log, code, tabular, HTML, repetition/redundancy, relevance, and adversarial tests;
- `engine/pixel`: applicability, density gate, glyph/atlas rendering, provider transforms, schema stripping, history handling, pricing, golden and regression tests;
- `engine/ccr`: exact-response storage, keying, expiration, SQLite/WASM stores, permission and concurrency tests;
- `engine/evals`, `contextwindow`, `safety`, `tokens`, `image`, and command wiring;
- standalone `cacheengine`, `rewriter`, `shrink`, `mem`, `proxy`, `browse`, and MCP packages;
- agents, skills, installers, extension/UI, integrations, package SDKs, benchmarks/evals, release metadata, issues/PRs/discussions, security/contribution docs, root and per-directory licenses, notices, provenance tests, and submodule state.

No load-bearing top-level subsystem was left unopened. Generated docs/site assets, font binaries, screenshots, and distribution archives were inventoried but not byte-reviewed because they cannot change the borrow decision; their licenses/notices were checked. Open issues and PRs were treated as direction, never shipped proof.

## License and provenance gate

The repository is mixed-license. Root/open interface surfaces include MIT-licensed portions, while the engine and several runtime packages use **Business Source License 1.1** with an Apache-2.0 change date of 2028-08-11; pixel assets carry additional font notices. Public visibility is not permission. All three filed borrows are therefore **INSPIRE-ONLY**: independently implement behavior against fak-owned fixtures, copy no expressive code/tests/comments/assets, and retain the pinned source only as prior-art provenance.

## Candidate cards

| Borrow | Source anchor | Axis | Their-worldview reason | fak witness on-axis | Route | Filed |
|---|---|---|---|---|---|---|
| Shape-specific terminal/search/log result codec with raw fallback | `engine/compressors/searchresult.go:28-67@c72984e`; `terminal.go:26-66@c72984e`; `log.go:22-71@c72984e` | Resident tool-result tokens while preserving anchors and deterministic recovery | Coding agents repeatedly carry semi-structured command output; format knowledge saves more than generic truncation | **PARTIAL:** `internal/ctxmmu/mmu.go`, `toolpages.go`, and `capbody.go` own budgets/CAS/paging but have no shape codec; TOON #3064 owns tabular JSON only | INSPIRE-ONLY | [#6670](https://github.com/anthony-chaudhary/fak/issues/6670) |
| Provider-aware text-to-image density gate | `engine/pixel/applicability.go:40-112@c72984e`; `density.go:16-91@c72984e`; `transform_openai.go@c72984e` | Choose text versus image representation only when provider-token value is positive | Dense monospace output can be cheaper as vision input, but only for supported models/content/geometry | **ABSENT:** fak has image geometry, ingestion, screenshot dedup, and VLM epic #4033, but no outbound tool-result renderer or density decision | INSPIRE-ONLY | [#6668](https://github.com/anthony-chaudhary/fak/issues/6668) |
| Annotation-aware minimization for hot tool schemas | `engine/compressors/toolschema.go:26-118@c72984e`; `toolschema_annotations.go:18-106@c72984e` | Remove only prose structurally redundant with type/requiredness while retaining semantic constraints | Harnesses resend schema prose every turn; blanket deletion is unsafe, so redundancy is field-local and fail-closed | **PARTIAL:** #3229/#3231/#3232 defer cold schemas, but the remaining hot schemas have no annotation-aware minimizer | INSPIRE-ONLY | [#6669](https://github.com/anthony-chaudhary/fak/issues/6669) |
| TOON/tabular JSON encoding | `engine/compressors/toon.go@c72984e`; `json_strategy.go:14-47@c72984e` | Reversible tabular JSON token density | Repeated objects are cheaper in a compact tabular wire form | **PRESENT-on-axis / already tracked:** `internal/toon` and epic #3064 with open scorecard #3068 and governed lossy follow-on #3343 | INSPIRE-ONLY; do not refile | #3064 |
| Cold tool-schema deferral | Caveman tool-schema compression neighborhood; fak comparison at `internal/gateway` tool-search seams | Avoid sending unused tool definitions at all | Schema tokens are an always-sent floor | **PRESENT-on-axis:** shipped #3231/#3232 under epic #3229 defer cold schemas; this dominates minification for cold tools | Stay with fak design | #3229 |
| Exact response cache | `engine/ccr/store.go:10-41@c72984e`; `store_sqlite.go:90-180@c72984e` | Return a stored completion for an exact repeated request | Repeated deterministic requests should avoid provider work entirely | **DIVERGENT:** fak deliberately centers provider prefix/KV reuse plus deterministic replay/audit rather than silently substituting a prior model response. Exact-response substitution changes freshness and stochastic semantics; no leaf filed without a named safe workload | Note only | — |
| Cross-session text memory | `mem/README.md@c72984e` and `mem/*.go@c72984e` | Persist/retrieve working facts across sessions | Coding users want continuity without replaying full transcripts | **PRESENT-on-axis:** fak memory/recall core images, `internal/memq`, and session images already provide governed persistence and retrieval | No borrow | — |
| Generic token-level compressor | compressor/eval harness and `engine/evals/harness.go@c72984e` | Semantic compression under quality thresholds | Different content needs measured quality, not one ratio | **PRESENT/on backlog:** Compressor work and closed #3204 already establish the plugin/eval direction; no Caveman-specific leaf survives | No duplicate | #3204 |

## Negative knowledge and direction

- Caveman's own `docs/HONEST-NUMBERS.md` says small outputs and cache-friendly stable prefixes can make transformation net-negative. This supports fak's existing net-true gate: transform cost, cache disruption, and quality loss must be included.
- The repository keeps explicit safe fallbacks and adversarial fixtures. A successful fak borrow should expose typed skip reasons instead of silently degrading to a different representation.
- The exact-response cache is useful for deterministic workloads, but using it as a general model-response cache would trade away freshness and sampling semantics. That divergence is intentional until a safe workload and audit contract are named.
- Pixel compression is a research direction, not a shipped fak performance claim. Provider image-token formulas, rendering overhead, and readability must all be witnessed together.

## Filed backlog

- #6668 — one-provider offline visual-token density gate with a captured render and `not-yet` fallback.
- #6669 — one conservative annotation redundancy class over fak's own hot schemas, scorecard-only first.
- #6670 — one format-specific codec at the existing context page-out seam, with reversible/raw fallback.

Each issue carries both the pinned upstream anchor and the fak seam, the on-axis verdict, the BSL-driven INSPIRE-ONLY route, a first checkable step, scope guard, and witness. The issues were deduplicated against all open/closed fak issues and received the derived `class:dev` label through `tools/issue_lane_router.py`.

## Companions

- `field-borrow` witness discipline, applied through `fak-dev feature query` plus raw source inspection
- Existing epics: [#1217](https://github.com/anthony-chaudhary/fak/issues/1217), [#3229](https://github.com/anthony-chaudhary/fak/issues/3229), [#4033](https://github.com/anthony-chaudhary/fak/issues/4033)
- Existing TOON track: [#3064](https://github.com/anthony-chaudhary/fak/issues/3064)

