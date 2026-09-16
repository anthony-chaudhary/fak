---
title: "Cache-value roll-up front door"
description: "docs/cache-value-rollup.md is the reader-facing cache-effectiveness P&L map; it keeps WITNESSED kernel reuse and OBSERVED provider-dollar savings unblended."
---

# Cache-value roll-up front door

[← Claims index](../../CLAIMS.md)


- [SHIPPED] Cache-value roll-up front door (#1308): `docs/cache-value-rollup.md` is the reader-facing cache-effectiveness P&L map. It explains why the signal was scattered, keeps Track 1 WITNESSED kernel reuse separate from Track 2 OBSERVED provider-dollar savings, states the #1066 marginal-over-warm-KV / WITNESSED-vs-OBSERVED / net-not-gross fences, names how to read the Slack-card fields, and gives the shipped Track-1 reproduce command (`fak nightrun score --json`) without claiming the future `fak cachevalue report --since` spelling on builds that do not expose it yet. Linked from `README.md` and `llms.txt`; `llms-full.txt` is regenerated from the same map. Witness: `python tools/gen_llms_full.py --check` (the front-door link plus regeneration); `make claims-lint`; `go test ./internal/managedocs -run TestClaimsIndexHasOnePagePerFormerClaim` (this page exists and carries its index backlink). [exposure: default-on]