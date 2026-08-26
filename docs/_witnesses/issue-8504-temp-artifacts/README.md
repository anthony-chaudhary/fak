---
title: "Issue #8504: live stale-temp-artifact dogfood"
description: "Reference documentation for Issue #8504: live stale-temp-artifact dogfood, preserving the page's implementation details, evidence, and operating context."
---

# Issue #8504: live stale-temp-artifact dogfood

On 2026-08-22, the `temp-artifacts` spine from commit
`278ca63914655af02bae654970d861b0384129d0` was built from committed trunk
`49afd3788614c6fb56424c7dce56b191cbf830e0` and run against this Windows
control point's real OS temporary directory.

The failing-before preview used a 24-hour threshold. It found 22 matching
direct artifacts: 8 stale and eligible (854,104,623 bytes) and 14 fresh and
preserved (1,766,423,257 bytes). The apply run reaped exactly those 8 eligible
artifacts through the quarantine/recheck/delete path. A second preview found
the same 14 fresh artifacts, zero eligible artifacts, and no warnings.

The run surfaced no implementation or operating defect, so there are no
follow-up issue numbers. `receipt.json` records the empty defect ledger and
the marker-key prefix that would have been used for any finding. Absolute temp
paths and process-inspection details are scrubbed; artifact basenames, byte
counts, typed reasons, source commits, commands, and exit codes remain.

Run the committed witness check with:

```text
go test ./docs/_witnesses/issue-8504-temp-artifacts -count=1
```
