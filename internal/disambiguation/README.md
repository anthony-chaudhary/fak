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
