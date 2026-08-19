# Provider cost reconciliation

`fak provider-cost` imports rows produced by an authoritative provider billing export. It does **not** estimate USD from tokens.

```text
fak provider-cost import --ledger .fak/provider-cost.jsonl --input provider-export.jsonl
fak provider-cost report --ledger .fak/provider-cost.jsonl --registry .fak/session-registry.jsonl
fak provider-cost reconcile --ledger .fak/provider-cost.jsonl --provider openai --expected-rows 120 --expected-micro-usd 43120000
```

Each input line uses `fak-provider-cost-ledger/1` and must contain `provider`, the provider's immutable `provider_row_id`, opaque `session_id`, interval bounds, `export_id`, `exported_at`, and `source`. `billed_micro_usd` and `currency: USD` are optional together: absence means **unknown**, not zero. Preserve the original provider export outside fak according to the organization's retention policy; `export_id` and `source` identify that evidence without copying customer content.

Imports deduplicate retries by `(provider, provider_row_id)`. Reports join `session_id` to `sessionregistry.Record.Identity.SessionID` only when every matching registration names one root. Zero matches are `missing`; multiple roots are `ambiguous`; neither contributes attributed USD. Retry registrations under the same root remain one valid root match. Coverage reports row and known-amount attribution independently.

Reconcile each imported provider/export interval against the provider console's exported row count and, when the export supplies one, billed total. A missing provider total remains absent. Investigate any row or amount mismatch before publishing root-goal efficiency claims.
