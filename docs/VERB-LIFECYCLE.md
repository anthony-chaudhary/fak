---
title: "Verb lifecycle — create, classify, index, expose, review"
description: "The full lifecycle of a fak verb: how one is created from a dispatch switch, classified on three axes, cataloged, regenerated into model- and human-facing surfaces, reviewed, and kept drift-free, with the exact refresh runbook and gates."
---

# Verb lifecycle

**Primary audience:** anyone adding, renaming, curating, or reviewing a `fak` / `fak-dev`
verb. A **verb** is a dispatched subcommand (`fak route`, `fak-dev index`, …). The dispatch
switch is the source of truth; every catalog, help wall, and model surface below is derived
from it and re-derivable with one command.

The lifecycle is: **create (dispatch) → classify (3 axes) → catalog (curated overlay) →
regenerate derived surfaces → model exposure + human review → docs chain**, with a hard
gate at every arrow. A verb that only exists in the dispatch switch is still legal and
visible — it appears with a fallback synopsis — but it fails the classification gates until
curated.

## 1. Create — the dispatch switch

Add the case to the dispatcher that owns the binary:

| Binary | Dispatcher | Notes |
| --- | --- | --- |
| `fak` | [`cmd/fak/main.go`](../cmd/fak/main.go) | user-facing runtime verbs |
| `fak-dev` | [`cmd/fak-dev/main.go`](../cmd/fak-dev/main.go) | maintainer/developer verbs |

`internal/devindex.Verbs()` is a **live view** over those dispatch switches: a new
dispatched verb auto-appears with the fallback synopsis `not yet cataloged — \`<binary>
<verb> -h\` for usage`. The `Binary` field (`fak` / `fak-dev`) is **derived, never
authored** — binary truth is pinned by `TestVerbBinaryMatchesRealDispatch` and
`TestFallbackVerbBinaryMatchesRealDispatch` (#13093). Coverage is total by construction
(`TestVerbManifestCoversEveryDispatcherVerb`, `TestVerbsDeriveCoversUncuratedDispatch`,
`TestVerbsIncludeDevOnlyDispatch`): nothing can dispatch without appearing in the catalog.

## 2. Classify — three orthogonal axes

Each axis is a separate file and a separate hard gate. A verb must be classified on all three.

| Axis | Where | Values | Gate |
| --- | --- | --- | --- |
| Tier | `internal/devindex/tiers.go` (`verbTiers`) | `frontdoor` \| `dev` \| `hidden` | `TestVerbTierCoverageIsTotal`, `TestVerbTiersNameOnlyLiveVerbs`, `TestCurrentDispatchedVerbTiersStayClassified`, `TestFrontdoorTierStaysSmall` (frontdoor ceiling) |
| Reuse class | `internal/devindex/devreuse.go` | `portableDevPatterns` \| `maintainerDevCommands` \| `labDevCommands` — exactly one | unclassified verbs fail the devreuse test ("add it to exactly one of …") |
| Lane | `dos.toml` `[lanes]` + `[lanes.trees]` | the lane that owns the verb's code tree | undeclared-leaf gate (`TestLiveInternalGoPackagesResolveToExplicitLaneTrees`), `dos` arbitration, commit stamps |

**Lane declaration** (needed when the verb ships a new package/leaf): add the name to
`[lanes].concurrent` **and** `[lanes].autopick`, and a `[lanes.trees]` glob entry — the
same three-part shape `fak new-leaf` generates (its `# new-leaf:lane` markers show the
insertion points). `cmd/<verb>.go` shims stay in the `cmd` lane so leaf trees stay
pairwise-disjoint. A committed package with no lane declaration is *drift*: the
undeclared-leaf detector reports it (`fak-dev index freshness`).

## 3. Catalog — the curated overlay

`internal/devindex/verbs.go` holds `verbManifest`, a quality overlay on the live view:
curated **synopsis** (≤110 runes) and **Lane** per verb. The overlay never invents coverage
— it only upgrades the fallback text. Style gates: `TestVerbCatalogStyleCleanAtHead`,
`TestVerbStyleBaselineShrinksOnly` (the baseline may only shrink),
`TestVerbStyleGateGrandfathersButRedsOnNew` (existing debt grandfathered, new violations red).

## 4. Regenerate — derived surfaces

Three generated artifacts are checked in and gate-locked. After any dispatch or manifest
change, regenerate all three:

| Artifact | Regenerate with | Staleness gate |
| --- | --- | --- |
| `internal/devhandoff/commands_gen.go` (per-verb binary/command inventory) | `go run ./cmd/fak-dev index ownership --write-manifest` | handoff fallback tests (`TestFallbackVerbBinaryMatchesRealDispatch`) |
| `docs/generated/verb-surface.md` (verb + refusal surface page) | `go run ./cmd/verbsdoc -write` | `VERB_SURFACE_SYNC` / `TestCheckedInPageIsFresh` (check mode is the default: `go run ./cmd/verbsdoc`) |
| `cmd/fak/verbs_gen.go` (per-verb help wall data) | `go run ./cmd/fak-dev index verbs --write-usage` | feeds `fak help <verb>` / `fak help --full` (`cmd/fak/usage.go`) |

## 5. Model-facing surfaces

How a model or agent discovers verbs — all read-only derivations of the same catalog:

- **Gateway MCP tools** (`internal/gateway/mcp.go`): `fak_index_verbs`, `fak_feature_query`,
  `fak_capabilities`, `fak_tools_search`.
- **`fak-dev index <query>`** and its subverbs (`lanes`, `leaf`, `claims`, `reconcile`,
  `verbs`, `generation`, `work`, `refs`, `ctxknobs`, `knobs`, `freshness`, `execaudit`,
  `agents`, `graph`).
- **llms.txt** is the source of truth for `llms-full.txt` (generated by
  `tools/gen_llms_full.py`, checked by `fak llms-full`). Front-door doc changes flow into
  llms.txt so model context stays current.
- **Private side** (`fak-private`): `fak-featureindex orient|refresh|watch`,
  `platform/verbhint` (did-you-mean), skills + AGENTS.md router tables, and the
  `fak-sync skill-tool-surface` CI audit. See the private repo's verb-index note under
  `fak-private/docs/notes/`.

## 6. Human-review surfaces

- **`fak help <verb>`** and **`fak help --full`** — the authored help wall in
  `cmd/fak/usage.go` (carved per verb; `verbs_gen.go` supplies the generated blocks).
- **[`docs/cli-reference.md`](cli-reference.md)** — the human verb index and flag-level
  help; **[`docs/cli/verbs.md`](cli/verbs.md)** and
  **[`docs/cli/verbs-extended.md`](cli/verbs-extended.md)** — per-verb deep contracts.
- **`fak-dev index verbs|lanes|leaf|claims|reconcile`** — the same taxonomy a reviewer
  needs to judge a verb change.
- **Root [`INDEX.md`](../INDEX.md)** — every doc must be reachable from here
  (`python tools/check_index_sync.py --audit-tree`).

## 7. Docs chain and drift gates

- **`go run ./cmd/fak-dev index freshness`** — the drift report: undeclared verbs,
  undeclared leaves (packages with no `dos.toml` lane), orphan dated notes (a
  `docs/notes/` file with no `INDEX.md` row), and dead/missing llms.txt links. The same
  detectors are pinned green at HEAD by `TestCheckFreshnessGreen`, `TestOrphanNotes`,
  `TestDeadLLMSLinks`, `TestMissingLLMSLinks*`,
  `TestLiveInternalGoPackagesResolveToExplicitLaneTrees`.
- **Commit gates** (`internal/hooks`, one staged-diff read; `fak hygiene --gates …`):
  `PUBLIC_LEAK`, `SECRET_SHAPE`, `DOC_PLACEMENT`, `BROKEN_LINK`, `FILE_ADMISSION`,
  `INDEX_SYNC`, `PROVENANCE_LABEL`. `DOC_PLACEMENT`/`INDEX_SYNC` require an `INDEX.md`
  line in the same commit as a new doc.
- **Private CI**: `fak-sync skill-tool-surface --harness pi` — skills must match the live
  tool surface; a renamed verb that leaves a stale skill form behind fails this audit.

## 8. Refresh runbook — adding or changing a verb

```text
1. dispatch    add the case in cmd/fak/main.go or cmd/fak-dev/main.go
2. classify    tiers.go (tier) + devreuse.go (reuse class)
               dos.toml [lanes].concurrent + [lanes].autopick + [lanes.trees]   # if a new leaf
3. catalog     verbs.go verbManifest  (synopsis <= 110 runes)
4. regenerate  go run ./cmd/fak-dev index ownership --write-manifest
               go run ./cmd/fak-dev index verbs --write-usage
               go run ./cmd/verbsdoc -write
5. gate        go test ./internal/devindex/ -count=1
               go run ./cmd/verbsdoc                        # check mode
               go run ./cmd/fak-dev index freshness
6. docs        docs/cli-reference.md / docs/cli/verbs*.md as needed
               root INDEX.md row (same commit) + llms.txt if front-door
```

For a **rename**, keep the canonical name managed and compatibility aliases explicit
(`TestManageIsCanonicalWithCompatibilityAliases`), then update every skill/prompt that
still shows the old verb form — private-side, `skill-tool-surface` is the audit that
catches the stale form.

## Worked example (2026-11-05)

`decode-anatomy` had dispatched and landed its tier, but was still uncataloged: a
`verbManifest` synopsis (Lane `cmd`, ≤110 runes), a `maintainerDevCommands` entry, and
`index ownership --write-manifest` + `verbsdoc -write` regeneration closed it. The
`verbsdoc` check was also stale because peers had landed new `cmd/fak` files without
regenerating the page — the refresh runbook's step 5 is what catches that. `index
freshness` surfaced two *other* leaves (`pkg/billing`, `pkg/routerlaunch`) that had landed
without a lane declaration; they were reconciled with the three-part lane declaration
above. A sibling verb (`layout-contract`) whose dispatch and implementation were still
**uncommitted** WIP was deliberately *not* curated ahead of its wiring:
`TestVerbTiersNameOnlyLiveVerbs` rejects tier rows for verbs that do not yet dispatch, so
steps 2–5 belong in the commit that wires the verb (or the curation commit right after
it) — never before.
