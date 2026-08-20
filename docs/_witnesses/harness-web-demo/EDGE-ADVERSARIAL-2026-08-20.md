---
title: "Harness operator home edge and adversarial witness — 2026-08-20"
description: "Captured regression evidence for bounded, fail-closed gateway status and safe operator-home rendering."
---

# Harness operator home edge and adversarial witness — 2026-08-20

Issue: #8244. Spine: `internal/harnessweb@r2+gd34d48dda`.

## Result

The operator home keeps its eight dashboard destinations disabled until the configured gateway returns a successful, bounded, valid `/debug/vars` document. A 502 response, malformed JSON, or a response larger than 2 MiB no longer leaves live links enabled. Hostile session text remains JSON data and is HTML-escaped at the API boundary; the page inserts live labels, descriptions, and paths with `textContent`.

The same table locks the existing operating envelope: eight dashboard destinations, a 5-second refresh interval, and the narrow layout at `max-width:600px`.

## Captured test run

```text
go test ./internal/harnessweb -run 'Edge|Adversarial' -v -count=1
--- PASS: TestAdversarialPublicGatewayURL
--- PASS: TestEdgeGatewayOverviewFailsClosed
--- PASS: TestEdgeCapturedHomeKeepsDashboardAndRefreshEnvelope
PASS
ok github.com/anthony-chaudhary/fak/internal/harnessweb 0.060s

go test ./internal/harnessweb -count=1
ok github.com/anthony-chaudhary/fak/internal/harnessweb 0.186s
```

## Boundary

This proves the loopback operator home against invalid gateway URLs, empty state, bounded upstream failures, malformed and oversized status documents, hostile display text, and the committed desktop/narrow render contract. It does not claim authenticated remote deployment or browser-engine fuzzing.
