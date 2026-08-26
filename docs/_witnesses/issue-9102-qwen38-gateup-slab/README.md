# Issue #9102 — Qwen Q4_K gate/up slab Mac REJECT

Verdict: **REJECT the Mac promotion; ship the implementation default-off.** The
pinned default-off control at revision
`d306fa7e35c033943990a8241673470e115e9477` completed its fak-native P=512/T=8
response, but the final three memory samples crossed the predeclared +12 GiB swap
gate before completion. The candidate was not admitted and never executed, so this
packet makes no speed, GC, footprint, or safety comparison.

The control used the exact Qwen3.8-27B Q4_K_M artifact SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
`FAK_Q4K_GATEUP_SLAB=0`, 512-token prefill chunks, 12 load workers, a fresh
process, and exclusive machine ownership. Its response receipt records
`inkernel` / `metal` / `metal/qwen35-hybrid-session-v1`, Q4_K true, no fallback,
64.81099575 seconds prefill, 77.313835416 seconds TTFT, and 56.27728775 seconds
decode for eight accepted tokens.

## Safety result

The first recorded swap value was 2,947,872,195 bytes, making the declared +12
GiB boundary 15,832,774,083 bytes. Samples 129–131 recorded 16,086,141,501,
16,535,194,173, and 16,535,194,173 bytes at 15:00:24Z–15:00:26Z. The response
completed at 15:00:30Z. These are three consecutive crossings before completion;
the candidate arm therefore remained unexecuted.

`control-memory-samples.raw.tsv` and `control-response.json` are byte-for-byte
copies of the path-free raw campaign artifacts. The raw sampler wrote literal
`\t` separators; the witness test deliberately parses that exact format instead
of normalizing it. `campaign-state.log` records that the run produced one control
arm directory and zero candidate arm directories. `control-server-summary.log` and `isolation-summary.log` are
scrubbed summaries: they retain execution identity, selector, hashes, teardown,
and restoration while removing local paths and PIDs. The watcher and restoration
files are copied unchanged because they contain no credentials or private paths.

## Readback

```console
go test ./docs/_witnesses/issue-9102-qwen38-gateup-slab -run '^TestGateUpSlabWitness$' -count=1 -v
```

The test pins every copied/scrubbed artifact by SHA-256, recomputes the three
consecutive swap crossings from the raw samples, validates that they precede the
completed native response, checks the exact control/candidate/config identities,
proves the candidate never executed, and verifies clean watcher/restoration
evidence. This is an immutable REJECT receipt, not a completed A/B.
