# Bounded serving QPS search

Opt-in `fak webbench serving` capacity search adapts Microsoft Vidur
`CapacitySearch.search`, at
`microsoft/vidur@abae7f63aa857300f5cdc6f5e0d27860cd24721b`,
`vidur/config_optimizer/config_explorer/capacity_search.py:125-182`
(blob `74c7fd517f8d233bc1fa0a86dd6ae42a0be0fbaa`). Ref #10854.

This is the integrated successor to the pure-function-only capacity-search leaf;
that earlier scope does not authorize caller wiring. This adaptation uses the
existing RunServingSweep -> measureServingSweepTrack -> runSamples SSE consumer.
It is not Vidur simulator parity: no Ray, CSV cache, simulation process, adaptive
upper-bound expansion, or scheduling-delay estimator is copied. Both bracket
endpoints are probed before bounded bisection. All attempts count against the
probe budget; a failed low endpoint establishes no passing bound. A passing high
endpoint establishes only that the supplied upper bound passed, not a maximum.

## Explicit operating envelope

Supply one track, exact model/engine receipt and batch-capacity provenance,
`--concurrency`, at least one of `--ttft-p99-budget-ms` and
`--itl-p99-budget-ms`, and all six bounds:

```
--capacity-min-qps 10 --capacity-max-qps 20 --capacity-tolerance-qps 1
--capacity-max-probes 8 --capacity-max-requests 80 --capacity-max-duration 30s
```

The workload (`--agents`) must contain at least two requests and its size times
max probes must fit the explicit request budget. Invalid/nonfinite rates,
unrepresentable timer intervals, missing budgets and incompatible contracts
refuse before measurement. Search does not combine with `--concurrencies`.
Fixed concurrency sweeps remain unchanged when capacity bounds are absent.
Capacity probes do not fetch auxiliary metrics URLs: the request budget covers
only the existing inference requests, without hidden per-probe metrics traffic.

Dispatch waits at least one rate interval between enqueue operations, without
catch-up bursts. Context cancellation interrupts dispatch and HTTP requests.
Observed enqueue rate below 95% of requested rate is **unknown**, never a pass;
this tolerance admits timer overhead, not a claim of exact open-loop load.
Queue delay is not silently included in measured server TTFT. The result records
both requested and enqueue rates, independent TTFT/ITL SLO checks, immutable
workload identity and contract. Failed requests, missing/invalid quantiles,
cancellation and under-delivery cannot tighten a passing bracket. Previous
passing observations may remain in a partial result but unknown/inconclusive
status must not be interpreted as a completed search.

`capacity_search` is separate from fixed-sweep points, peak and SLA-knee
certification. Qualification is `bounded_client_probe_not_capacity_certification`.
A finite client replay does not establish sustainable physical throughput,
stationarity, statistical confidence, production capacity or hardware parity.
The supplied engine identity is a declaration, not proof that an endpoint ran
that engine. Caller-supplied HTTP transports must honor request cancellation.

Focused witnesses: `TestCapacitySearchBisectsMaxSustainableQPS` (synthetic
threshold/error and loopback consumer) and
`TestWebbenchCapacitySearchCLIIntegration` (actual production CLI and loopback
SSE, including HTTP failure). These are CPU software evidence only, no external
endpoint/model/GPU load. No throughput gain or full-system parity is claimed.

## Upstream license

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
