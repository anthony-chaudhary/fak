# Kernel-served verbalizable-direction witness

This is the captured live witness for #4437. On 2026-07-15, a sanctioned L4 compute node ran the checked-in `fak serve` kernel against `qwen2.5-0.5b-instruct-q8_0.gguf`. The bidirectional hidden hook was exercised through the resident-Q8 CPU forward (the CUDA HAL currently has a separate device-resident loop); host and session identifiers are scrubbed.

## Method

- Concept: positive versus negative instruction framing.
- Contrastive prompts differ only in the final concept word and have the same 18-token absolute position.
- `direction.f32` is the unit-normalized layer-8 positive-minus-negative residual.
- A neutral matched prompt was steered at layer 8, position 18 with coefficients `-2,-1,0,1,2`.
- Residuals were read at layers 8, 12, 16, 20, and 23. `sweep.json` records decoded text and every projection.
- The decoded one-word answer stayed `neutral`; the residual projection nevertheless rose strictly with the coefficient at every captured downstream layer. This is an observed broadcast result, not an overclaim of a token flip.
- A separately launched unarmed serve and the alpha-zero serve produced byte-identical layer-23 dumps (`sha256` in `meta.json`).

`positive/`, `negative/`, and `sweep/` contain little-endian f32 residuals (896 values each). The repository test `TestVerbalizableDirectionLiveArtifact` independently re-reads these bytes, checks the direction against the captured diff-of-means, and proves monotonic broadcast at two downstream layers.
