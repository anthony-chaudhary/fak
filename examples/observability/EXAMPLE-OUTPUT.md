# Captured run — `./run.sh`

A real run of [`run.sh`](run.sh) on an offline `fak serve` (mock engine — no model, API
key, or GPU). The script drives one allow, one deny, one quarantined result, and a
repeated read-only call the vDSO serves from cache, then scrapes `/metrics`, fetches
`/debug/vars`, and demonstrates the auth path. Every PromQL query in
[`README.md`](README.md) names a metric that appears below.

```console
$ ./run.sh
[obs] starting kernel: fak serve 127.0.0.1:8080  (offline mock engine, floor = customer-support-readonly-policy.json)
[obs] kernel healthy: {"engine":"inkernel","model":"mock","ok":true,"planner":"mock"}

[obs] driving calls through the kernel: allow, deny, quarantine, + a repeated read-only call (vDSO)

[obs] GET /metrics — the lines an operator queries:
# Q1  Is the floor firing?  (verdict counters + kernel deny/quarantine totals)
fak_gateway_operations_total{operation="admit",verdict="QUARANTINE",reason="SECRET_EXFIL",disposition="",by="normgate"} 1
fak_gateway_operations_total{operation="syscall",verdict="ALLOW",reason="",disposition="",by="monitor"} 2
fak_gateway_operations_total{operation="syscall",verdict="ALLOW",reason="",disposition="",by="vdso"} 1
fak_gateway_operations_total{operation="syscall",verdict="DENY",reason="POLICY_BLOCK",disposition="TERMINAL",by="monitor"} 1
fak_kernel_denies_total 1
fak_kernel_quarantines_total 1

# Q2  Is the cache hitting?  (vDSO hit ratio + backing counters)
fak_kernel_submits_total 4
fak_kernel_vdso_hits_total 1
fak_gateway_vdso_hit_ratio 0.25

# Q3  Is anything stuck inflight?  (aggregate gauge + oldest-request age)
fak_gateway_inflight_requests 1
fak_gateway_inflight_max_age_seconds 0

# build labels (version / engine / model / vdso)
fak_gateway_build_info{version="0.32.0",engine="inkernel",model="mock",vdso="true"} 1

[obs] GET /debug/vars — break-glass live process view:
# kernel counters block
{
  "submits": 4,
  "vdso_hits": 1,
  "engine_calls": 2,
  "denies": 1,
  "transforms": 0,
  "quarantines": 1,
  "admitted": 2,
  "vdso_hit_ratio": 0.25
}
# gateway: inflight + uptime + auth_required
{
  "inflight": 1,
  "uptime_s": 1.646,
  "auth_required": false
}
# runtime: goroutines + heap (is it leaking / wedged?)
{
  "goroutines": 37,
  "alloc_bytes": 2953952,
  "num_gc": 3
}

[obs] restarting WITH --require-key-env to demonstrate the auth path
  GET /metrics  (no token)                       -> HTTP 401
  GET /metrics  (Bearer token)                   -> HTTP 200
  GET /healthz  (always open)                    -> HTTP 200

[obs] done — see README.md for the PromQL queries and EXAMPLE-OUTPUT.md for a captured run.
```

## Reading the capture

- **Q1 — the floor fired.** Three verdict shapes are visible in
  `fak_gateway_operations_total`: an `ALLOW` (the allow-listed `search_kb`), a
  `DENY`/`POLICY_BLOCK` (the off-policy `refund_payment` refused by structure), and a
  `QUARANTINE`/`SECRET_EXFIL` on the `admit` path (the poisoned `fetch_url` result the
  result-side floor paged out — note `by="normgate"`, the adjudicator that caught it). The
  kernel aggregates agree: `fak_kernel_denies_total 1`, `fak_kernel_quarantines_total 1`.
- **Q2 — the cache hit.** The second identical read-only call was served from the vDSO
  pool, not re-run: `fak_kernel_vdso_hits_total 1` of `fak_kernel_submits_total 4`, so
  `fak_gateway_vdso_hit_ratio 0.25`. A cache-served call also surfaces as a verdict row
  with `by="vdso"`.
- **Q3 — nothing stuck.** `fak_gateway_inflight_max_age_seconds 0` — the only in-flight
  request at scrape time is the scrape itself (`fak_gateway_inflight_requests 1`). Under
  load you would also see one `fak_gateway_inflight_requests_by_route{route="…"}` series
  per backed-up route.
- **Break-glass.** `/debug/vars` returns the same kernel counters as JSON plus the live
  runtime view (`goroutines`, heap `alloc_bytes`, `num_gc`) — the wedged-process probe you
  reach for when you want the state *now* without a Prometheus round trip.
- **Auth.** Restarted under `--require-key-env`, `/metrics` returns `401` without a token
  and `200` with the `Bearer`; `/healthz` stays open. `/debug/vars` follows the same gate.

> Exact numbers (`uptime_s`, `alloc_bytes`, `num_gc`, `goroutines`, the `version` label)
> vary run to run and host to host; the metric **names** and verdict **shapes** are what
> this example pins.
