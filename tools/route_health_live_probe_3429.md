# Live route-health probe witness — real NIM provider route (#3429)

Parent: #3035 (`fak dispatch route-health` selected-route smoke gate).
Follow-up: #3429 — "Run one live `fak dispatch route-health` probe against a real
remote provider route." Handoff key: `route-health-live-provider-probe`.

This file is the durable, in-repo witness for the one live validation the #3429
handoff asked for: a single real probe of a real OpenAI-compatible provider route,
with the `fak-route-health/1` ledger row, the `status`/`gate` folds, and the
`tools/dispatch_status.py --fast` `routes:` line all rendered from that one live row.

## What was run (2026-07-19T09:52:51Z, provider NIM)

Route: `nim/netra/meta/llama-3.1-8b-instruct` against
`https://integrate.api.nvidia.com/v1`, credential in `NVIDIA_API_KEY`.

```
$ fak dispatch route-health probe \
    --base-url https://integrate.api.nvidia.com/v1 \
    --model meta/llama-3.1-8b-instruct \
    --provider nim --account netra \
    --api-key-env NVIDIA_API_KEY --workspace .
route-health probe nim/netra/meta/llama-3.1-8b-instruct: healthy (HTTP 200)
  recheck: fak dispatch route-health probe --base-url https://integrate.api.nvidia.com/v1 --model meta/llama-3.1-8b-instruct --provider nim --account netra --api-key-env NVIDIA_API_KEY
# exit 0
```

### `fak-route-health/1` ledger row (`.dispatch-runs/route-health.jsonl`)

```json
{"schema":"fak-route-health/1","route":"nim/netra/meta/llama-3.1-8b-instruct","provider":"nim","account":"netra","model":"meta/llama-3.1-8b-instruct","base_url":"https://integrate.api.nvidia.com/v1","class":"healthy","status":200,"probed_at_unix":1784454771,"recheck":"fak dispatch route-health probe --base-url https://integrate.api.nvidia.com/v1 --model meta/llama-3.1-8b-instruct --provider nim --account netra --api-key-env NVIDIA_API_KEY"}
```

### `status` fold

```
$ fak dispatch route-health status --workspace .
route health — 1 route(s) probed, 0 suppressed
ledger: .dispatch-runs/route-health.jsonl
  nim/netra/meta/llama-3.1-8b-instruct  class=healthy  probe_age=1s  http=200
    recheck: fak dispatch route-health probe --base-url https://integrate.api.nvidia.com/v1 --model meta/llama-3.1-8b-instruct --provider nim --account netra --api-key-env NVIDIA_API_KEY
# exit 0
```

### `gate` fold (pre-spawn verdict for this route key)

```
$ fak dispatch route-health gate --provider nim --account netra --model meta/llama-3.1-8b-instruct --workspace .
route-health gate nim/netra/meta/llama-3.1-8b-instruct: pass — last probe healthy (1s ago)
# exit 0
```

### `tools/dispatch_status.py --fast` `routes:` line (rendered from the live row)

```
$ python tools/dispatch_status.py --fast
...
║ routes    : 1 probed, 0 suppressed (#3035)
```

## Acceptance gate (#3429)

Met: the transcript shows a `fak-route-health/1` ledger row and the `--fast`
`routes:` line rendered from it (`1 probed, 0 suppressed`).

## Promotion evidence

The seven typed failure classes were already hermetically witnessed under #3035;
this run promotes the **healthy** path from assumed to observed against a real
remote provider — a live HTTP 200 from the NIM `integrate.api.nvidia.com` route,
classified `healthy`, gate `pass`, folded into both the Go `status` snapshot and
the Python `--fast` card.

## Demotion / retirement evidence

The prior impl-header witness note in `cmd/fak/dispatch_route_health.go` (commit
`31c6538bd4`, dated 2026-07-12, model `deepseek-ai/deepseek-v4-pro`, "transcript on
#3035") had **no ledger witness** — no `.dispatch-runs/route-health.jsonl` row
existed for it, so that claim is un-witnessed and is superseded by this real,
reproducible run. (Correcting that cmd-lane header is left to a cmd-lane worker;
this witness lives in the tools lane that owns the `--fast` fold.)

## Invalidating assumption

A `healthy`-at-probe-time row cannot promise `healthy`-at-spawn-time: the route can
rate-limit or 5xx between this probe and the next live-worker spawn. The gate/`--fast`
fold also trusts the ledger's `probed_at_unix`/`cooldown_until_unix`, so a
clock-skewed provider `Retry-After` landing in the past would silently un-suppress a
still-limited route. This probe validates the plumbing, not a standing guarantee.

## Caveat (host / build)

The full `fak` binary does not build on the current trunk (unrelated peer WIP in
`internal/agent`, `internal/policy`, and `cmd/fak` leaves undefined symbols /
redeclarations). The probe was therefore run from a byte-identical standalone build
of the real `cmd/fak/dispatch_route_health.go` route-health code plus the real
`llmd_smoke.go` auth helpers — same schema, same classification, same ledger writer.
The HTTP call, the 200, and the ledger row are genuinely live; only the binary
packaging differed.
