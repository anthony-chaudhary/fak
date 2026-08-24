# Top-50 documentation refresh audit — 2026-08-24

Issue: [#8786](https://github.com/anthony-chaudhary/fak/issues/8786)

## Verdict

All 50 selected pages were read against current source, generated-command, version,
claim, benchmark, or operating-procedure authorities. The sweep refreshed 21 pages
across five explicit-path commits and found 29 pages current. No identified semantic
defect remains in the selected set.

Selection prioritized front-door reachability, authority status, operator criticality,
and fak's current native-inference/context focus. A structural score alone was not
accepted as semantic evidence: the initial core scorecard covered only 25 pages and
was already green while version, command-owner, refusal-count, and implementation
status drift remained.

## Exact manifest and final disposition

| # | Page | Authority checked | Final disposition |
|---:|---|---|---|
| 1 | `README.md` | README freshness auditor + benchmark authority | Current; presentation notices were advisory only |
| 2 | `START-HERE.md` | current front-door map | Current |
| 3 | `INDEX.md` | `cmd/fak-dev`, issue-contract registries | Refreshed command ownership in `f8ef9f486f` |
| 4 | `INSTALL.md` | install command and release surfaces | Current |
| 5 | `GETTING-STARTED.md` | current CLI/build entry points | Current |
| 6 | `docs/index.md` | reachable-doc map | Current |
| 7 | `docs/fak/tutorial.md` | `VERSION` + captured-output provenance | Historical capture labeled in `c32f513eb2` |
| 8 | `docs/repro-packet.md` | current preflight/agent commands | Current |
| 9 | `cmd/simpledemo/README.md` | simpledemo command source | Current |
| 10 | `docs/FAQ.md` | `internal/abi.CoreReasonCount` + structured-data copy | Refreshed 17-code vocabulary in `f8ef9f486f` |
| 11 | `llms.txt` | `VERSION` + command registries | Refreshed version and issue commands in `c32f513eb2`, `f8ef9f486f` |
| 12 | `CONTRIBUTING.md` | shared-tree contributor commands | Current |
| 13 | `SECURITY.md` | policy and disclosure authorities | Current |
| 14 | `ARCHITECTURE.md` | `internal/model`, `internal/polymodel`, tree existence | Refreshed current/hypothetical package status in `0af67bf1d7` |
| 15 | `POLICY.md` | `internal/abi/reasons.go` | Refreshed full refusal vocabulary in `b5dc36be77` |
| 16 | `CLAIMS.md` | command registries, reason count, current implementation | Refreshed in `f8ef9f486f` |
| 17 | `STATUS.md` | current status authorities | Historical snapshot labeled and redirected in `c32f513eb2` |
| 18 | `EXTENDING.md` | extension registry/source | Current |
| 19 | `BENCHMARK-AUTHORITY.md` | current speculative-decode implementation + frozen artifact | Refreshed provenance/mechanism split in `0af67bf1d7` |
| 20 | `docs/project-orientation.md` | current portfolio authorities | Current |
| 21 | `docs/problems-we-solve.md` | P1–P4 source doctrine | Current |
| 22 | `docs/native-inference-goal.md` | native engine invariant | Current |
| 23 | `docs/benchmark-methodology.md` | benchmark gates/receipts | Current |
| 24 | `docs/supported/features.md` | feature sources + refusal count | Refreshed in `f8ef9f486f` |
| 25 | `docs/ROLLBACK.md` | current tags, `fak garden`, Makefile | Refreshed commands/version examples in `c32f513eb2` |
| 26 | `docs/dev-tooling.md` | `fak-dev` command source | Current |
| 27 | `docs/fleet-compute-nodes.md` | sanctioned-node routing contract | Current |
| 28 | `docs/private-comms-channel.md` | public/private boundary | Current |
| 29 | `docs/releases-channel.md` | release skill and CLI | Current |
| 30 | `docs/spine-first-defaults.md` | `cmd/fak-dev` issue commands | Refreshed in `f8ef9f486f` |
| 31 | `docs/generated-output-defaults.md` | dispatch registry | Refreshed `fak dispatch tick` in `f8ef9f486f` |
| 32 | `docs/integrations/README.md` | integration directory/current adapters | Current |
| 33 | `docs/integrations/adopter-playbook.md` | benchmark authority + MCP discovery | Refreshed decision baseline/tool discovery in `0af67bf1d7` |
| 34 | `docs/integrations/claude.md` | Claude integration source | Current |
| 35 | `docs/integrations/openai-codex.md` | Codex integration source | Current |
| 36 | `docs/integrations/cursor.md` | Cursor integration source | Current |
| 37 | `docs/integrations/mcp.md` | MCP serve/tool discovery source | Current |
| 38 | `docs/explainers/policy-in-the-kernel.md` | policy source + all local links | Current; visual-content suggestion remained advisory |
| 39 | `docs/explainers/addressable-kv-cache.md` | cache implementation/authority | Current |
| 40 | `docs/explainers/sota-optimizations.md` | native goal + model/routing/vision sources | Refreshed native ownership and implementation status in `0af67bf1d7` |
| 41 | `docs/explainers/cache.md` | current cache sources | Current |
| 42 | `docs/explainers/gateway.md` | gateway source | Current |
| 43 | `docs/explainers/context.md` | context-management sources | Current |
| 44 | `docs/explainers/context-shedding.md` | context MMU sources | Current |
| 45 | `docs/explainers/routing.md` | route registries | Current |
| 46 | `docs/explainers/agent-runtime.md` | runtime source | Current |
| 47 | `docs/explainers/one-binary-one-surface.md` | refusal ABI + native architecture | Refreshed in `f8ef9f486f` |
| 48 | `docs/model-routing.md` | route selection source | Current |
| 49 | `docs/kv-capacity-normalization.md` | `internal/modelperfobs` implementation/test | Current; missing-path report was independently refuted |
| 50 | `docs/cli-reference.md` | `VERSION`, command registries, flag definitions | Refreshed release and `git-daily` flags in `d11ea699b3` |

## Shift-left result

`tools/docs_scorecard.py` now treats an explicit current-version declaration as a
hard freshness contract and compares it with `VERSION`, while preserving historical
examples and snapshots. Its focused regression suite covers exact, missing-patch,
near-current, historical, and install-pin cases. This catches the class that allowed
`llms.txt`, the tutorial, and status prose to score clean while asserting old releases.

Command ownership and generated FAQ copies remain source-derived review targets; adding
a broad second command-doc generator in this sweep would duplicate existing registries
rather than provide a smaller complete fix.

## Final evidence

- `python tools/docs_scorecard.py --scope core --json`: `OK`, 25 docs, mean 97.0/100,
  100% coverage, zero doc debt.
- `python tools/check_links.py --audit-tree`: front-door link set clean.
- `python tools/readme_freshness_audit.py --json`: 100/A, zero failures.
- `python -m pytest -p no:cacheprovider -q tools/docs_scorecard_test.py -k freshness`:
  12 passed, 26 deselected.
- Search readback found none of the witnessed stale forms: old current-release language,
  `fak issue contract|cohort|fanout`, bare `fak tick`, 12-code refusal claims, obsolete
  `git-daily --emit/--install`, or current-mechanism claims bound only to `internal/spec`.

Peer-dirty files outside this manifest were neither claimed nor swept into these commits.
