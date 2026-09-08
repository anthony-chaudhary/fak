# First-class tracing portfolio

This portfolio makes transformations and agent-visible causality queryable without creating parallel trace dialects. The portable contract lives in `pkg/traceabi`; domain recorders and engines adapt to it.

## Foundation

- [#12258](https://github.com/anthony-chaudhary/fak/issues/12258): typed causal envelope.
- [#12259](https://github.com/anthony-chaudhary/fak/issues/12259): many-to-many value lineage across split, fuse, unpack, repack, expand, dequantize, and contract.
- [#12260](https://github.com/anthony-chaudhary/fak/issues/12260): bounded privacy and evidence policy.
- [#12261](https://github.com/anthony-chaudhary/fak/issues/12261): bounded query, explain, and canonical export.
- [#12262](https://github.com/anthony-chaudhary/fak/issues/12262): deterministic reconstruction and first divergence.

The shared `pkg/traceabi/**` work is serialized in that order.

## Domain tracers

- [#12264](https://github.com/anthony-chaudhary/fak/issues/12264): authoritative packed, storage, logical, and expressed tensor views; follows #12259 and #10380.
- [#12263](https://github.com/anthony-chaudhary/fak/issues/12263): one live GGUF tensor from loader bytes to kernel consumption; follows #12258–#12261 and #12264.
- [#12265](https://github.com/anthony-chaudhary/fak/issues/12265): first numeric divergence; follows #12263.
- [#12266](https://github.com/anthony-chaudhary/fak/issues/12266): native context/KV lifecycle; follows #12265.
- [#12267](https://github.com/anthony-chaudhary/fak/issues/12267): model/tool turn trace; follows #12258–#12261 and is tree-disjoint from the model chain.

The useful agent-facing path is: packed bytes → unpack → repack → expand/dequantize → contract → bounded query/explain → replay/first divergence.
