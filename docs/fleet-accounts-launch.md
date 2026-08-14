---
title: "Account-bound fleet worker launch"
description: "Resolve and execute Claude, Codex, and OpenCode workers through one typed account boundary."
---

# Account-bound fleet worker launch

Fleet and super-loop workers must launch through the account roster. Do not export an
ambient config home and invoke a worker directly. Resolve and preview a launch with:

```bash
fak fleet-accounts launch --product codex --task-tier 1 --task "hard engineering" --prompt "resolve #6534"
```

Use `exec` instead of `launch` to run the returned command. Every supported worker product
is bound to its own configuration root: Claude uses `CLAUDE_CONFIG_DIR`, Codex uses
`CODEX_HOME`, and OpenCode uses `XDG_CONFIG_HOME`. OpenCode fleet dispatch additionally
refuses to build `opencode run` unless both a resolved account record and a task tier reach
the typed launch-decision seam; an environment-only fallback is not accepted.

The default `.fak/fleet-launches.jsonl` ledger records the resolved account, product,
configured model, invoked model, endpoint class, task tier, verdict, and override marker.
It deliberately excludes tokens, config paths, and all environment values.

## Narrow tier-3 override

Tier-3 OpenCode/API seats are restricted to narrow tier-3 work. The operator must both
classify the task as tier 3 and pass the explicit override:

```bash
fak fleet-accounts exec --product opencode --account nemo-tier3 \
  --task-tier 3 --allow-tier3-narrow --prompt "update this bounded document"
```

The override never permits a tier-3 seat to run tier-1 or tier-2 work. For hard engineering,
the resolver must select a ready tier-1 product such as Codex or refuse the launch.
