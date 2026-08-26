---
title: "Harness stack compatibility resolution"
description: "fak harness stack resolve answers whether a requested harness/model/tool/policy/runtime/hardware stack can compose."
---
# Harness stack compatibility resolution

`fak harness stack resolve` answers whether a requested harness/model/tool/policy/runtime/hardware stack can compose. It keeps hard compatibility separate from recommendations and returns a provenance-bearing receipt rather than a bare yes/no.

```bash
fak harness stack resolve --manifest stack.json
fak harness stack resolve --manifest stack.json --json
```

The input must use `fak-stack-manifest/1`; JSON output uses `fak-stack-receipt/1`. The resolver follows required, recommended, optional, conflict, and substitute relations. Required failures return exit `3` with the transitive dependency chain and remediation. Missing recommendations remain warnings and do not block launch.

This surface establishes **technical composition**, not workload fitness or live hardware support. Use the integrated stack-preflight workflow when workload authority/evidence floors and exact hardware/quantization support must also gate launch. A synthetically supported tuple is not a live GPU witness.

See the [captured first-class CLI witness](../notes/HARNESS-STACK-RESOLVE-CLI-2026-08-15.md) and the [resolver spine](../notes/NATIVE-HARNESS-STACK-RESOLUTION-SPINE-2026-08-15.md).

For the production-facing model behind these manifests—especially ordinary local machines, stable host inventory, optional organization/domain profiles, and set-valued tools/skills/routes—see [Harness stacks are typed assemblies, not uniform profile piles](../notes/HARNESS-STACK-LOCAL-FIRST-MODEL-2026-08-15.md).
