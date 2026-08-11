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

The package is deliberately read/validation-only: it performs no filesystem or
network writes and imports no private source. A later index generator remains
the sole writer. `testdata/entry-v1.json` is a complete public-source fixture;
`RunSelfTest` accepts it in equivalent typed form and independently removes and
rejects every required JSON path declared by the schema descriptor.

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

## Scoped overloaded terms

A canonical token may have multiple entries only when each entry has a distinct, required `scope.kind` + `scope.value` qualifier. Unscoped queries of an overloaded token return `scope required`; callers select an owner with `--scope-kind` and `--scope-value`, and JSON returns the stored scope unchanged. For example:

```text
fak disambiguation query kernel --scope-kind package --scope-value internal/disambiguation --json
```

The index remains a single public-source writer. Scope does not broaden source admission or permit private repository, host, credential, or local-path material.
