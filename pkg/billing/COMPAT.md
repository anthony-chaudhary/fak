# pkg/billing compatibility policy

`pkg/billing` and its subpackages are a public disclosure surface: their
exported Go surface and the on-disk formats they produce are consumed outside
this repository. They evolve under the contract below.

## Scope

- The exported Go surface, centered on `UsageRecord` v1 and the `Policy`
  interface.
- The versioned on-disk artifacts: keystore `fak-billing-keystore/v1`,
  manifest `fak-billing-manifest/v1`, ledger `SchemaVersionV1`, and the JSONL
  usage-report records.

## Rule: additive evolution only

`UsageRecord` v1 and `Policy` evolve additively.

Additive (compatible):

- A new optional `UsageRecord` field whose zero value preserves v1 reader
  behavior, together with the constructor or helper that populates it.
- A new method, constructor, option, sink, or exported error value.

Breaking (requires a new major version):

- Removing or renaming an exported symbol, or changing a signature.
- Retyping a field, or changing the semantic meaning of an existing field,
  default, or format element.
- Changing a versioned on-disk format in a way a v1 reader cannot parse.

A breaking change ships under a new major version; until one exists, every
change to this surface must be additive.

## Reader obligation

Readers of versioned artifacts tolerate unknown optional fields and reject
only what they cannot parse.
