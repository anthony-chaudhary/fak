# OTLP trace export dependency decision — 2026-08-19

## Verdict

Use a small dependency-free OTLP/HTTP JSON exporter for fak's bounded server/provider span vocabulary. Do not add the OpenTelemetry Go SDK yet.

The repository intentionally has zero external Go dependencies. The shipped need is narrow: asynchronous export of W3C-linked gateway and provider spans with fixed, low-cardinality attributes. The implementation uses the standard library, a 256-span nonblocking queue, two-second HTTP timeout, bounded shutdown drain, and explicit accepted/exported/dropped/failed/depth metrics. Export is off by default, so unset configuration constructs no span queue or worker.

Adopt the OTel SDK when fak needs dynamic processors, multiple exporters, resource detectors, tail/head sampling, baggage propagation, or semantic-convention breadth that would otherwise reproduce a substantial part of the SDK. That is the measured complexity threshold; package popularity alone is not a reason to add a runtime dependency.

## Privacy and cardinality

Exported attributes are fixed to service name, route template, HTTP method, status, and timing. Prompts, responses, tool arguments, raw URLs, users, session IDs, and arbitrary labels are excluded. Trace/span IDs are correlation fields, not metric labels.

## Waterfall witness

[`docs/_witnesses/otlp-trace-join-2026-08-19.txt`](../_witnesses/otlp-trace-join-2026-08-19.txt) captures a collector-backed server span. Provider-child linkage is pinned by `TestProviderSpanContextCreatesChild`: the child retains the 128-bit trace ID, names the server span ID as `parentSpanId`, and receives a fresh span ID. This separates total gateway duration from provider-attempt duration without recording payload content.
