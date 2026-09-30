---
title: "Grant one exact command to the native agent"
description: "Use a narrow, repeatable command grant without widening the native Bash allowlist."
---

# Grant one exact native-agent command

`fak agent` keeps its focused Bash allowlist by default. When a task needs one
additional, operator-reviewed command, grant that command byte for byte:

```text
fak agent --native \
  --allow-bash-command "my-tool inspect --json" \
  --task "Inspect the workspace with the approved command and summarize the result."
```

Repeat `--allow-bash-command` to grant more than one command. Each value is an
exact match: extra arguments, shell operators, prefixes, suffixes, or different
spacing produce a different command and remain refused. Empty grants are invalid.

The flag only extends the bounded native `Bash` tool for this process. It does
not disable focused-command filtering, capability adjudication, workspace
confinement, timeouts, or dangerous-argument checks. Code tools remain enabled
by default; combining a grant with `--code-tools=false` is an error.
