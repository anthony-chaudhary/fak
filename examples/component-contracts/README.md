# Publish component compatibility contracts

A kernel, cache, runtime, or other component can publish a standalone
`fak-component-contract/1` JSON file. Consumers compose files from different authors; nobody
has to copy the declarations into one central catalog.

```powershell
go run ./cmd/fak component check `
  --contract examples/component-contracts/radix-cache.json `
  --contract examples/component-contracts/paged-attention-kernel.json `
  --contract examples/component-contracts/cuda-runtime.json `
  --root cache.kv.radix@1
```

The result is an evidence-bearing compatibility receipt. In this example the cache's hard
kernel requirement and the kernel's hard CUDA-runtime requirement resolve. The cache's CUDA
graph recommendation is absent, so the stack is allowed with a warning. Remove
`cuda-runtime.json` and the same command refuses with exit code 3 and identifies
`runtime.cuda.12` as the missing transitive requirement. Add `--json` for automation.

## Contract versus hint

Each relation has one of four meanings:

| Kind | Meaning |
|---|---|
| `requires` | Hard compatibility contract. The stack refuses when no component ID or provided capability satisfies it. |
| `conflicts` | Hard incompatibility contract. The stack refuses when the target is selected. |
| `recommends` | Soft performance, quality, or operational hint. Its absence produces `MISSING_RECOMMENDATION`, not refusal. |
| `optional` | Soft integration hint with no warning when absent. |

Targets are exact component IDs or exact provided capabilities in this first schema. A
relation may list ordered `substitutes`; the receipt records when one is used. Every component
and relation carries `authority` and `source`, with optional proof `tier` and `freshness`, so a
resolver decision points back to its publisher's evidence rather than presenting compatibility
as an unexplained boolean.

## Authoring shape

```json
{
  "schema": "fak-component-contract/1",
  "component": {
    "id": "cache.example@1",
    "kind": "cache",
    "version": "1.0.0",
    "provides": ["cache.example"],
    "relations": [
      {
        "kind": "requires",
        "target": "kernel.example",
        "evidence": {
          "authority": "cache maintainer",
          "source": "release compatibility test"
        }
      }
    ],
    "evidence": {
      "authority": "cache maintainer",
      "source": "release 1.0.0"
    }
  }
}
```

Use globally meaningful IDs and capability names, pin the component's own version in `id` and
`version`, and cite the narrowest reproducible evidence for each relation. Schema mismatches,
unknown fields, malformed relations, and duplicate component IDs fail before resolution.

This is the publication seam, not a registry or package manager. Remote discovery, signatures,
version ranges, and measured preference ranking can build on the contract without changing the
hard-versus-hint distinction witnessed here.
