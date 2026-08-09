# Context-compression comparison runner

`fak headroom compare` is the shared-corpus local spine for #6064:

```text
fak headroom compare --via none,native,lingua --json
```

The command runs every requested Compressor plugin over exactly the same frozen `headroom.BenchCorpus`. It emits each arm's byte savings and local transformation time. Missing integrations are explicit and produce exit 3; while #3204 remains unshipped, the default command reports `MISSING lingua` rather than silently comparing only native against itself. Even when all local arms exist, `complete` remains false until the report carries task success, retained-fact recall, provider tokens, TTFT, regrowth tax, and total cost; these appear as typed `pending` metrics instead of being buried in prose.

This local runner is **not** the completed product benchmark. It cannot establish retained-fact recall, task success, provider input tokens, cache behavior, TTFT, regrowth/re-read tax, or total cost. Those outcomes require #6064's end-to-end same-model witness. The `none` arm is only local pass-through here; its provider prefix-caching behavior belongs in that live witness.
