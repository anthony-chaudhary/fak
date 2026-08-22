# Default documentation self-index dogfood, 2026-08-22

Issue [#8255](https://github.com/anthony-chaudhary/fak/issues/8255) asked for one
real run of the default documentation lookup shipped by
`8db26221bb495cbd321712a46e20c453786c8040`. This readout records the run against
the live repository at committed HEAD
`7c48bc047cb421f1411a5256b0bb7743ca3744e0` before any witness-file edit.

## Verdict

**PARTIAL:** the shorthand executes successfully, stays bounded, and is byte-equivalent
to the explicit `index docs` spelling. It does not yet narrow this real task to its
owning documentation, and the bounded result set is not path-unique. Both defects were
marker-deduped and filed; fixing them is outside this dogfood leaf.

## Live task and captured output

The real task was the issue's own work: dogfood the default documentation self-index.
`INDEX.md` had no working-tree change when the run began. The command was:

```text
fak-dev index "documentation self-index dogfood" --limit 10
```

It exited 0 and returned:

```text
docs/notes/DOS-FRESH-INSTALL-VALUE-DOGFOOD-LAPTOP-2026-06-25.md  DOS fresh-install value dogfood, from the laptop (2026-06-25)
docs/notes/MODELOPS-TOP3-DOGFOOD-2026-07-15.md                   Exact-model canary gate dogfood (2026-07-15)
docs/notes/LOOKAHEAD-RESET-DOGFOOD-TRACKER-2026-07-17.md         Look-ahead reset dogfood tracker: P0 → P1 → P2 runbook + witnessed tracking (2026-07-17)
docs/notes/ULTRACODE-DOGFOOD-QWEN36-PARITY-2026-06-28.md         Ultracode dogfood — a 9-agent fleet advancing Qwen3.6 parity (2026-06-28)
docs/notes/GIT-DAILY-DOGFOOD-2026-08-06.md                       Daily Git hygiene dogfood — 2026-08-06
docs/notes/GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md               GLM-5.2 GCP dogfood bring-up, 2026-07-05
docs/notes/CODEX-WORKFLOW-DEFAULT-DOGFOOD-2026-08-17.md          Guarded Codex workflow-default dogfood — 2026-08-17
docs/notes/MICRO-CONTEXT-DOGFOOD-2026-08-08.md                   Micro-context dogfood readout — 2026-08-08
docs/notes/MYSTERY-FREE-DOGFOOD-ARCHITECTURE-2026-08-21.md       Mystery-free architecture dogfood
docs/notes/MYSTERY-FREE-DOGFOOD-ARCHITECTURE-2026-08-21.md       Mystery-free dogfood: repository architecture — 2026-08-21
```

The explicit control was:

```text
fak-dev index docs "documentation self-index dogfood" --limit 10
```

The two ten-line outputs were equal. Their LF-joined stdout SHA-256 was
`2dc643aa064c43a283e9a96783aae1bfdc4f9e724d31aa2e1f27e907572f8574`.
This proves the new default dispatches to the existing documentation search rather than
a separate path.

## Outcome checks

| Check | Observed result | Verdict |
|---|---|---|
| Invocation | Default and explicit commands exited 0. | PASS |
| Default equivalence | Ten stdout lines were byte-equal under the stated LF normalization. | PASS |
| Bound | `--limit 10` returned ten rows. | PASS |
| Task relevance | `docs/dev-tooling.md`, which `INDEX.md` names as the owner of documentation lookup, ranked 24th of 42 matches and was absent from the bounded result. | FAIL: [#8537](https://github.com/anthony-chaudhary/fak/issues/8537) |
| Documented canary | The documented `shared-tree commit` query ranked `docs/dev-tooling.md` 14th of 52 matches, outside its advertised top-five example. | FAIL: [#8537](https://github.com/anthony-chaudhary/fak/issues/8537) |
| Path uniqueness | The ten task rows contained nine distinct paths; the Mystery-Free note occupied two slots. | FAIL: [#8538](https://github.com/anthony-chaudhary/fak/issues/8538) |

This scope is intentionally the default documentation search and its explicit control.
Other named index subcommands are separate surfaces, not defect generators for this leaf.

## Defect ledger and marker-key dedupe

Before creation, exact marker searches and semantic searches over this repository returned
no existing issue for either finding.

| Marker key | Filed issue | Done condition |
|---|---|---|
| `dogfood-devcmd-docindex-multiterm-relevance-v1` | [#8537](https://github.com/anthony-chaudhary/fak/issues/8537) | Specific multi-term queries rank their owning document ahead of unrelated single-token hits while preserving exact and fuzzy behavior. |
| `dogfood-devcmd-docindex-path-dedupe-v1` | [#8538](https://github.com/anthony-chaudhary/fak/issues/8538) | Default and explicit searches return at most one row per repository-relative Markdown path with deterministic metadata precedence. |

No other defect was surfaced by the bounded default-search run. The follow-up issues own
the fixes; this issue owns only the live readout and complete filing record.

## Reproduce

From the repository root at the captured revision:

```bash
fak-dev index "documentation self-index dogfood" --limit 10
fak-dev index docs "documentation self-index dogfood" --limit 10
fak-dev index "shared-tree commit" --limit 5
fak-dev index "documentation index" --limit 10
```
