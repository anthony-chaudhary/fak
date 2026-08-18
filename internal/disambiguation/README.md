# Disambiguation entry schema

`internal/disambiguation` owns the public, machine-readable terminology-entry
contract. Agents can inspect the pinned contract through the public binary:

```text
fak disambiguation schema --json
fak disambiguation schema --self-test --json
```

Version 1 is `fak-disambiguation-entry/1`. Its reader rejects unknown versions,
unknown fields, trailing JSON values, missing required fields, invalid enum
values, and non-public source locators. `identity.aliases` must be present as an
array but may be empty (`[]`). Adding, removing, renaming, or changing the type
of a field requires a new schema version.

The package performs no filesystem or network writes and imports no private
source. `GeneratePublicIndex` is the sole byte writer for the tracked public
artifact; `cmd/fak` only routes those bytes to disk. `testdata/entry-v1.json` is
a complete public-source fixture; `RunSelfTest` accepts it in equivalent typed
form and independently removes and rejects every required JSON path declared by
the schema descriptor.

## Canonical and declared-alias query

The public read seam performs an exact, case-sensitive lookup across canonical
terms and declared aliases; it never normalizes whitespace or case. Canonical
input returns the canonical record directly. Alias input returns that same
complete canonical record and adds `matched_alias` with the exact declared
alias used, so canonical identity and ownership remain visible:

```text
fak disambiguation query "agent kernel" --json
fak disambiguation query "fused agent kernel" --json
fak disambiguation query --self-test --json
```

`Query` remains the package-level canonical-only seam; `Resolve` is the additive
alias-aware seam used by the CLI. Index construction rejects duplicate canonical
terms, duplicate aliases (including repeats under one owner), and aliases that
collide with any canonical term. These checks are the generation/index boundary:
ambiguous identity cannot become queryable.

The response contract is `fak-disambiguation-query/1` and names the public seed
version separately. Its nested `entry` is the complete strict schema record,
including meaning (`definition`), scope, contrasts, owner, public sources, and
freshness. The reader is stdlib-only and read-only; it is not a second index
writer and has no private source dependency.

## Ranked search and explicit ambiguity

Use `fak disambiguation search <term> --json` when discovery, rather than exact
resolution, is intended. The `fak-disambiguation-search/1` response keeps exact
canonical, exact alias, and prefix matches in separate ranked groups. Its typed
`verdict` is `exact`, `alias`, `prefix`, `ambiguous`, or `not_found`; multiple
owners produce `ambiguous` instead of silently selecting the first candidate.
The human form prints the same groups, and exits 3 for ambiguity so automation
cannot mistake a candidate list for a resolved identity.

```text
fak disambiguation search "agent" --json
fak disambiguation search "kernel"
```

## CLI terminology source

`fak disambiguation cli-source --json` derives command, subcommand, and long-flag
terms directly from the public runtime help synopsis. Each flag retains its
invocation context, so shared spellings do not lose ownership. The read-only
`fak-disambiguation-cli-source/1` report can compare a prior snapshot and lists
removed terms under `stale`; `--self-test` captures one automatically added CLI
term and one removed verb. This projection is not a second index writer.

## Pairwise contrasts and forbidden conflations

Each contrast names another canonical entry, carries a non-empty explanation,
and explicitly records `required_pair` and `forbidden_conflation`. A required
pair must be declared in both directions and must agree on whether conflation is
forbidden. Index admission rejects self-contrasts, duplicate or unknown targets,
and asymmetric required pairs. Querying either a canonical term or an alias
returns the canonical entry with these contrast fields unchanged; for example,
`fak disambiguation query "fused agent kernel" --json` exposes the required,
forbidden `agent kernel` / `compute kernel` distinction.

These are reader-side admission checks over the same public index. They add no
writer, filesystem access, network access, or private dependency.

## Fuzzing the public read seams

The normal Go test gate discovers five native fuzz targets in `fuzz_test.go`:
`FuzzParseEntryMalformed`, `FuzzContrastGraphCycles`,
`FuzzDuplicateIdentities`, `FuzzUnicodeConfusables`, and
`FuzzPathologicalAliasSets`. Their deterministic seed corpus runs during every
ordinary `go test`; bounded mutation can be invoked with, for example,
`go test ./internal/disambiguation -run '^$' -fuzz FuzzUnicodeConfusables -fuzztime 5s`.

The targets import the package externally (`disambiguation_test`) and exercise
only the exported `ParseEntry`, `Entry.Validate`, `NewIndex`, and `Resolve`
seams. They add no corpus writer, generated index, filesystem/network access, or
private-source dependency. Alias and graph inputs are bounded so a routine fuzz
smoke cannot turn cardinality or string width into an accidental resource test.


## Ownership and dispatch admission

Every entry's `owner.leaf` and `owner.lane` fields are admission data, not free-form
labels. Agents discover the authoritative public contracts in these existing places:

- **owner leaf registry:** real public directories under `internal/<leaf>/` and
  `cmd/<leaf>/`, matching the ship-stamp contract in `AGENTS.md` and
  `tools/commit_stamp_doctor.py`;
- **dispatch lane registry:** `dos.toml` `[lanes]` (`concurrent`, `exclusive`,
  `autopick`) plus `[lanes.trees]`, the workspace manifest used by dispatch and
  lease tooling.

Generation calls `LoadPublicManifests` and `NewAdmittedIndex`, rejecting an entry
before output when either target is absent. This reads only public repository
paths—never private repositories, host configuration, or a second generated
registry. Run `fak disambiguation ownership --self-test --json`; its report proves
one accepted fixture and separate rejected leaf/lane fixtures through the public
CLI seam.

The seed distinctions retain their accountable `kernel` or `disambiguation`
owner leaves. Their dispatch lanes are `kernel` for the kernel-owned distinction
and the declared public terminology lane `canon` for disambiguation-owned
terminology distinctions.

## Scoped overloaded terms

A canonical token may have multiple entries only when each entry has a distinct, required `scope.kind` + `scope.value` qualifier. Unscoped queries of an overloaded token return `scope required`; callers select an owner with `--scope-kind` and `--scope-value`, and JSON returns the stored scope unchanged. For example:

```text
fak disambiguation query kernel --scope-kind package --scope-value internal/disambiguation --json
```

The index remains a single public-source writer. Scope does not broaden source admission or permit private repository, host, credential, or local-path material.


## Deterministic generated index

`fak disambiguation generate` validates the declared public entries, sorts entries,
aliases, contrasts, and source witnesses into canonical order, and writes
`docs/generated/disambiguation-index.json`. The artifact records the source-set
SHA-256 as `source_revision`; the command report records the final byte digest.
Run `fak disambiguation generate --check --json` at a committed tip to prove
regenerate-and-diff is clean. `--check` is read-only and exits nonzero on missing
or stale output. No repository crawl, network read, private source, or second
serializer participates in generation.

`fak disambiguation version --json` exposes the entry schema, generated-index
schema, source revision, entry count, and ordered-content SHA-256 for citations.
Semantic content changes move both digests; source-order-only changes move
neither because versioning consumes the same canonical bytes as generation.

`fak disambiguation committed-freshness --json` archives `HEAD` into an isolated
temporary checkout, runs the committed generator check there, and compares that
artifact with generation from the invoking binary. It reports `committed-clean`,
`own-overlay-drift`, or `unavailable`; a dirty peer file cannot make a committed
tip look stale, and an unavailable git/toolchain probe never looks clean.

The regenerate-and-diff check is a hard gate in all canonical CI entry points:
`make ci`, `scripts/ci.ps1`, the unchanged `build · vet · test · claims-lint`
GitHub job, and `fak-dev ci-preflight` over its isolated committed archive. The
workflow job/check name is unchanged; only a step was added. Rollback removes
that step/target/preflight call together and leaves the generator artifact intact.

## Strict public provenance

Every `sources[]` entry is immutable provenance returned unchanged by
`fak disambiguation query <term> --json`:

- `kind` is one of `document`, `go-source`, `test`, `generated-index`, or
  `github-metadata`; private repositories, local files, arbitrary URLs, and
  other unverifiable kinds are refused.
- `locator` is a normalized slash-separated repository-relative path with an
  optional `#fragment`; absolute Windows/Unix paths, backslashes, and `..`
  escapes are refused.
- `revision` pins the public source state, `checked_at` is canonical UTC RFC
  3339, and `probe` is the stable lowercase public probe identity.

Provenance admission failures are `ValidationError` values with stable `code`
and `field` members; use `ValidationCode(err)`, not message parsing. Validation
walks entries and sources in input order, making the first error deterministic.
This adds no writer: the existing entry/index writer remains authoritative and
the dependency boundary remains public fak repository/GitHub evidence only.
Run `fak disambiguation provenance --self-test --json` for the hermetic CLI
round-trip plus absolute-path, escaping-path, and private-kind rejection witness.

## Public-safety admission

`Entry.Validate` also scans every string field before an entry can reach the
index. It rejects credential-shaped text, absolute local paths, private
repository names, private hostnames or RFC 1918 addresses, and terminal control
bytes with stable `DISAMBIGUATION_PUBLIC_SAFETY_*` codes. Repository-relative
public locators and ordinary public documentation remain valid. The recursive
scan covers nested fields and future string fields by default, so adding a field
cannot silently bypass the publication boundary.

## Freshness verdict contract

Agents and JSON consumers get exactly four freshness states. Every state has one
stable reason code; do not infer freshness from timestamps or probe transport
success alone.

| Verdict | Stable `reason_code` | Meaning |
|---|---|---|
| `fresh` | `SOURCE_CURRENT` | The public probe was available and returned valid, current evidence. |
| `stale` | `SOURCE_OUTDATED` | The public probe was available and returned valid evidence that is not current. |
| `unknown` | `PROBE_UNAVAILABLE` | The probe could not supply evidence. Unavailability never degrades to `fresh`. |
| `invalid` | `EVIDENCE_MALFORMED` | Probe metadata or returned evidence was malformed. Malformed evidence is never classified as merely `stale`. |

`EvaluateFreshness` is the single read-only classifier. Its inputs are public
probe observations; it performs no I/O, reads no private source, and does not
write or regenerate the index. Existing schema, query, alias, contrast, scope,
and ownership behavior is unchanged. A `Freshness` value is valid only when its
verdict and reason-code pair match this table and `checked_at` is canonical
RFC3339.

Run the package witness with `go test ./internal/disambiguation` and the public
CLI JSON witness with:

```text
fak disambiguation freshness --self-test --json
```


## Public reference freshness

A source witness may add a `reference` object to cite a repository-visible
contract inside its existing public provenance locator:

- `go-symbol`: an exported top-level Go declaration in the cited `.go` file;
- `cli-verb`: a literal public verb dispatched by the cited Go source;
- `reason-code`: a stable uppercase reason-code literal in cited Go source;
- `doc-anchor`: a GitHub-style Markdown heading anchor in the cited document.

`ProbePublicReferences(repoRoot, entry)` reads only those repository-relative
public sources. A present reference is `fresh/SOURCE_CURRENT`; removing it is
`stale` with `PUBLIC_SYMBOL_MISSING`, `CLI_VERB_MISSING`,
`REASON_CODE_MISSING`, or `DOCUMENT_ANCHOR_MISSING`. A source that cannot be
read for reasons other than absence remains `unknown/PROBE_UNAVAILABLE`. Invalid
entry/provenance/reference evidence remains `invalid/EVIDENCE_MALFORMED` and
outranks unavailable or stale results, preserving the four-state #6280
precedence. This is a reader/probe over #6281 provenance, not a writer, registry,
or private-source integration.

Agents can capture the deterministic package/CLI transition witness with:

```text
fak disambiguation stale-symbols-self-test --json
```

The JSON emits both the initial `fresh` result and the result after deleting the
fixture declaration (`stale/PUBLIC_SYMBOL_MISSING`).


## Coverage inventory

`InventoryCoverage` is the deterministic orphan-term check for agents adding public
terminology. Call it with the repository's canonical `Index`, an explicit list of
`PublicTerminologySurface` declarations, and explicit `IncidentalTerm` classifications.
The first supported surface kind is `go_package`: it inventories exported type, function,
constant, and variable names from non-test Go files beneath exactly the declared package
directory. It does not crawl the repository, read private inputs, use the network, or write
an index.

Every exported candidate must resolve by one of these paths:

- **canonical**: its normalized words are a canonical term/alias, or the canonical entry's
  public-source provenance names the exact exported Go symbol;
- **incidental**: the caller supplies the exact symbol and a non-empty reason; or
- **finding**: JSON reports `MISSING_TERM_CLASSIFICATION`, the declared surface, exact symbol,
  and normalized candidate term.

Surfaces and findings are sorted, so identical public inputs produce identical JSON. Keep
surface declarations narrow and reviewed; adding a broad repository heuristic defeats the
contract by turning implementation churn into terminology noise.

Run the hermetic public CLI witness before changing coverage plumbing:

```text
fak disambiguation coverage-self-test --json
```

The witness introduces `NewlyExportedTerm`, observes the stable missing-classification
reason, classifies it as incidental, and verifies the finding clears. Package callers can
use the same `CoverageSelfCheck`; neither path introduces a second writer.


## Incidental classification contract (#6284)

`TermClassification` is the strict public contract for exported local implementation tokens that must count in terminology coverage but must never become canonical glossary/query rows. Build the normal index with `NewClassifiedIndex(entries, classifications)`; this is the same `Index` and the same `InventoryCoverage` writer used for canonical entries, not a second registry or identity path.

An incidental classification has deterministic machine-readable fields:

```json
{
  "schema_version": "fak-disambiguation-classification/1",
  "term": "LocalRetryToken",
  "classification": "incidental",
  "reason": "LOCAL_IMPLEMENTATION_TOKEN"
}
```

Validation is closed and deterministic: the term must be an exported Go identifier; schema, classification, and reason must be the constants above; duplicate terms and unknown values fail. `Index.Query` remains canonical/alias-only, while `InventoryCoverage` reports the classification and increments `incidental`. Agents can witness both sides with:

```text
fak disambiguation coverage-self-test --json
```

A passing report has `covered: true`, `absent_from_query: true`, and the stable classification/reason fields. The older `[]IncidentalTerm` coverage argument remains accepted for compatibility with #6283, but new declarations should use `NewClassifiedIndex` so classification belongs to the shared index.

## Generated documentation pages (#6320)

`fak disambiguation docs` renders the human-facing canonical-term page and reverse contrast index from the same typed `Entry` records used by `generate`, `query`, and `search`. The tracked pages live under [`docs/generated/disambiguation/`](../../docs/generated/disambiguation/); each row links to a scoped identity and prints the exact public query command.

```text
fak disambiguation docs
fak disambiguation docs --check
fak disambiguation docs --json
```

`--check` is the drift gate: it fails if either page differs and never rewrites it. Rendering is sorted and byte-identical, so documentation does not become a second terminology writer.

## Reverse lookup from repository evidence (#6321)

Agents that start from evidence rather than a glossary term can use `fak disambiguation reverse` to find the canonical distinction that owns it. The lookup reads the same immutable `Entry` index as `query` and `search`; it does not crawl the tree, infer fuzzy matches, or maintain a second registry.

```text
fak disambiguation reverse --kind source-path internal/disambiguation/query.go
fak disambiguation reverse --kind symbol Query
fak disambiguation reverse --kind cli-token disambiguation
fak disambiguation reverse --kind reason-code SOURCE_CURRENT
fak disambiguation reverse --self-test --json
```

Supported kinds are exact and closed: `source-path` matches a source locator (including the path portion before a document anchor), while `symbol`, `cli-token`, and `reason-code` match typed public references. Freshness reason codes are also indexed. A locator may return multiple evidenced owners; unknown input returns `reverse locator not found` with an empty `matches` array rather than fabricating an owner.
