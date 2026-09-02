# Captured run — `examples/local-hardware-benchmark`

Real run of the legacy entry point (`go run . …`, identical to `fak bench local …`)
on the capture host: darwin/arm64, Apple M3 Pro. Offline except that `run` executes
exactly the child command the operator supplies — here a trivial `/bin/echo hello`,
chosen so the sealed-receipt mechanics can be demonstrated without a model or GPU.

## 1. `inventory` — normalized hardware + benchmark catalog (excerpt)

```console
$ go run . inventory
{
  "hardware": {
    "os": "darwin",
    "arch": "arm64",
    "cpu": "Apple M3 Pro",
    "memory_bytes": 38654705664,
    "accelerators": [
      {
        "vendor": "Apple",
        "kind": "gpu",
        "model": "Apple M3 Pro",
        "backend": "Metal"
      }
    ],
    "toolchains": {
      "metal": "xcrun: error: unable to find utility \"metal\", not a developer tool or in PATH"
    }
  },
  "benchmarks": [
    {
      "name": "ablate",
      "kind": "verb",
      "need": "offline",
      "summary": "Self-ablation feature sweep: replays ONE frozen tool-call trace under N runtime-feature configs and reads each arm's cost/benefit (vDSO hits, denies, p50 latency, tokens) straight off the kernel counters  -  the N-arm generalization of `fak bench`, apples-to-apples on one workload hash.",
      "run": "fak ablate --sweep vdso"
    },
    ... 40 more catalog rows in the same shape, through "webbench-token-measure" ...
  ]
}
```

The catalog row this capture runs against:

```json
{
  "name": "modelbench",
  "kind": "cmd",
  "need": "weights",
  "summary": "In-kernel pure-Go forward-pass latency / tok-per-sec, so the kernel's model numbers are self-measured not borrowed.",
  "run": "go run ./cmd/modelbench -dir internal/model/.cache/smollm2-135m"
}
```

## 2. `run` — execute the operator command, seal a receipt

```console
$ go run . run --benchmark modelbench --engine fak-native --out /tmp/receipt.json -- /bin/echo hello
hello

receipt: /tmp/receipt.json
verify: fak bench local verify /tmp/receipt.json
```

Exit 0. The child ran exactly as supplied (no fallback to any external engine) and
its scrubbed argv, duration, exit status, and output digest were sealed into the
receipt at `/tmp/receipt.json`.

## 3. `verify` — the seal checks out

```console
$ go run . verify /tmp/receipt.json
VERIFIED fak.local-hardware-benchmark.receipt/v1 6ca136ddeff03b5762cfc53513af4c4290f64f6274ffa21f94bf55df93648dc4 benchmark=modelbench engine=fak-native exit=0
```

Exit 0. Verification fails closed on unknown schema versions, unknown JSON fields,
trailing JSON data, and integrity mismatches.

## 4. `submit` — no-upload packet

```console
$ go run . submit /tmp/receipt.json
## Local hardware benchmark receipt

- Schema: `fak.local-hardware-benchmark.receipt/v1`
- Receipt SHA-256: `6ca136ddeff03b5762cfc53513af4c4290f64f6274ffa21f94bf55df93648dc4`
- Benchmark: `modelbench`
- Engine: `fak-native`
- Command: `["/bin/echo","hello"]`
- Exit status: `0`
- Output SHA-256: `5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03`
- Hardware: `darwin/arm64`; CPU `Apple M3 Pro`; memory `38654705664` bytes
- Accelerators: Apple Apple M3 Pro (Metal)
- fak: `0.45.0` (`cef36f01524a`)
- repo revision: `a947c40907ffa88944e9b5024debcd8d4f457d08`

The receipt was generated locally by `fak bench local` (promoted from `examples/local-hardware-benchmark`). The tool did not upload it. I reviewed this packet for privacy before submission.

Related: #10421, #10444

---
Open this URL to review and explicitly submit (this tool never uploads):
https://github.com/anthony-chaudhary/fak/issues/new?body=...&labels=benchmark&title=bench%3A+local+hardware+receipt+darwin%2Farm64+6ca136ddeff0
```

Exit 0. `submit` only prints the Markdown issue body and the prefilled URL; it never
opens the URL or uploads anything.

## What the capture proves

- **The receipt workflow is end-to-end runnable offline.** inventory → run → verify →
  submit each completed on a plain laptop with no model, weights, or key.
- **The seal is real.** The verify line re-computes and matches the receipt's SHA-256
  integrity seal; the submit packet echoes the same digest.
- **Explicit-engine discipline.** The receipt records `engine=fak-native` because the
  operator labeled it — nothing was auto-selected, and no fallback fired.
- **No-upload submission.** The last step's entire effect is stdout; the operator
  reviews and explicitly submits.
