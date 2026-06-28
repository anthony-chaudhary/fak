# Example output

A captured run of `./run.sh` — `fak serve` in front of its **offline mock planner**
(no model, no GPU), driving `N = 1, 2, 5` workers that share one prompt prefix through
the one kernel. The bytes/turns columns are an exact accounting of the request bodies;
the `✓ live kernel` line is the real kernel serving every worker behind one `fak serve`.
Reproduce: `./examples/fleet-reuse-demo/run.sh` (or `--offline` for the accounting only).

```
fak — fleet shared-prompt reuse demo  workers=1,2,5  kernel=http://127.0.0.1:8099
  the first worker pays for the shared setup; everyone after reads it for free

reuse curve — N workers sharing one prompt prefix behind fak serve
  shared setup prefix = 837 B (system prompt + tool catalog + house rules); per-worker δ ≈ 80 B

  metric                               naive re-send        fak shared-prompt
                                            N × full       1 × full + (N-1)·δ
  --------------------------------------------------   ----------------------
  N = 1
    total prompt bytes sent                    917                    917  (1.00× less)
    model turns re-processing setup              1                      1  (1.00× less)
    injection in shared context     per-worker (1×)    walled at 1st admit
  N = 2
    total prompt bytes sent                  1,831                    994  (1.84× less)
    model turns re-processing setup              2                      1  (2.00× less)
    injection in shared context     per-worker (2×)    walled at 1st admit
  N = 5
    total prompt bytes sent                  4,569                  1,221  (3.74× less)
    model turns re-processing setup              5                      1  (5.00× less)
    injection in shared context     per-worker (5×)    walled at 1st admit

✓ live kernel served 5/5 workers behind one `fak serve` at http://127.0.0.1:8099 (model=mock); every worker shared the same 837 B system prefix — the reuse substrate.

summary: reuse curve consistent  ·  at N=5: 3.74× fewer prompt bytes, setup re-processed 1× not 5×  ·  live: 5/5 workers served behind one fak serve
  honest scope: the ~60× is only vs the naive re-send loop; the ~4× is vs a tuned warm-cache
  stack; and the reuse win is self-host only (an app that calls a frontier API gets the safety
  floor, not the savings). This is the measured curve for THIS small N — not projected to scale.
```

## Reading the curve

- **N = 1** has nothing to reuse yet, so naive and fak are byte-identical (`1.00×`). The
  demo does not pretend there is a win here — that honesty is what makes the rest credible.
- **N = 2 → 5** is the reuse curve: the shared 837 B setup is sent once behind fak
  instead of once per worker, so the prompt-byte ratio climbs `1.84× → 3.74×` and the
  setup is re-processed `1×` rather than `N×`. This lands in the README's **~1.5–4× vs a
  tuned warm-cache stack** region — not the ~60×-vs-naive headline, and not the
  frontier-scale "agent city" projection.
- The `injection in shared context` row tracks the same prefix: a poisoned shared-context
  entry would reach every worker in a naive loop (`N×`) but is walled at the first admit
  behind fak. The live quarantine mechanism is proven in `../adjudication-demo/` and
  `../session-reload/`; here it is the structural count.

The `--offline` run prints the identical table without the `✓ live kernel` line (it skips
the kernel and shows the accounting only) — useful on a box with no Go build or where you
just want the curve.
