# Typed harness asset composition

`fak harness compose` compiles the ordered layers emitted by contextual selection into one inert, provenance-bearing effective asset set:

```text
fak harness select  ... > selection.json
fak harness compose --assets harness-assets.json --selection selection.json
```

The asset manifest schema is `fak.harness-assets/v1alpha1`. Its initial closed kinds are `instruction`, `tool`, `memory`, `policy`, `route`, `secret`, `workflow`, and `ui`. Each declaration carries layer/source identity and one explicit operation: `add`, `replace`, or `remove`.

## Merge rules by kind

| Kind | Duplicate/add | Replace | Remove | Additional floor |
|---|---|---|---|---|
| instruction | identical retains; different conflicts | explicit only | allowed unless locked/mandatory | ambiguous activation never silently wins |
| tool | identical retains; different conflicts | explicit only | allowed unless locked/mandatory | ambiguous tool implementation never silently wins |
| memory | same namespace/boundary only | explicit within same boundary | allowed unless locked/mandatory | project/domain/task memory boundary must equal owning layer |
| policy | later declarations may add denies/remove grants | widening refused | allowed unless locked/mandatory | an existing deny or absent parent grant cannot become a later grant |
| route | identical retains; different conflicts | explicit only | allowed unless locked/mandatory | route changes are visible replacements |
| secret | identical reference retains | refused | allowed unless locked/mandatory | opaque reference required; inline value forbidden |
| workflow | identical retains; different conflicts | explicit only | allowed unless locked/mandatory | mandatory witness/workflow cannot be removed |
| UI | identical retains; different conflicts | explicit only | allowed unless locked/mandatory | UI preference never gets policy semantics |

Any kind can be locked by its source layer. A locked asset cannot be changed or removed by a narrower layer. Mandatory assets cannot be removed. Every effective asset names its source layer, and every action emits an add/retain/replace/remove/narrow trace.

This is deliberately not generic last-writer-wins. Instructions and tools require an explicit replacement when names collide. Memory cannot silently cross a project/domain/task boundary. Policy can become narrower but not wider. Secret references cannot be swapped by a lower layer, and raw secret values are invalid.

## Witness and boundary

The mixed company/person/project/legal/task witness composes all eight kinds deterministically. The 8-kind × 3-operation matrix covers every kind with add, replace, and remove. Adversarial tests seed policy widening, secret replacement, mandatory witness removal, instruction/tool ambiguity, and memory cross-contamination; each fails before launch with the source layer, kind, and asset ID.

Selection order is consumed directly from `selection.json`; discovery and classification are not reimplemented. This leaf resolves typed overlap semantics from #6904. It does not yet solve version/dependency ranges, cycles, provider ambiguity, compatibility budgets, or lock digests; those remain the broader deterministic resolver #6792. It also emits inert assets rather than projecting into Codex/Claude/Copilot or launching `pkg/harnesskit` (#6901).
