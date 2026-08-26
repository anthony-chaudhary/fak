# Issue #8385 — partial Mac decode-window campaign

Verdict: **HOLD / DEMOTE**. This is a truthful partial native-arm witness, not a decode-decay
or llama.cpp-parity result.

The fresh fak-native request asked for 2,048 tokens but stopped normally after 251. Its buffered
response contains 251 consecutive `fak.native-decode-trace/1` events and 251 ordered
`fak.native-decode-token-ids/1` IDs. It contains no heavyweight native-inference receipt. The
campaign adapter then rejected the realistic nested `usage.prompt_tokens_details` object before
it could archive repetition one. No second native repetition and no comparator process started.
Therefore neither the three-repetition decay fold nor the matched 3×2 parity fold is eligible.

The nonperturbative memory watcher recorded no safety crossing: maximum swap growth was
7,582,850,744 bytes against a 12 GiB ceiling, and both free-memory floors stayed at or above 16%
against the 10% floor. Because the arm aborted before three responses froze, there is no post-arm
`phys_footprint_peak`; this directory makes no peak-footprint claim. Cleanup exited zero and
verified restoration of the exact `Submitted` service by argv hash, binary hash, port owner,
health, and exact model.

`native-response.json` is a curated wire response: its response ID and assistant text are scrubbed,
while usage, finish reason, trace timestamps, and token IDs remain intact. Process IDs, machine
identity, and private paths are omitted from the other artifacts. `provenance.json` records the
source, binary, model, and comparator hashes without claiming the comparator ran.

Deterministic readback:

```text
go test ./docs/_witnesses/issue-8385-qwen38-mac-decode-windows -count=1
```
