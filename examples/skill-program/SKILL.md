---
name: repo_search
description: Search repository text through a deterministic first-class fak command.
---

# Repository search

Use this tool only when the current provider request advertises it. The prose
explains intent; it does not register or execute the command.

```fak-program
{"version":"fak.skill-program/v1","name":"repo_search","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},"executor":{"argv":["fak","code","search","--json"],"adapter":{"version":"fak.command-adapter/v1","argv":[{"field":"query","flag":"--query"}],"result":"json"}},"aliases":{"codex":"functions.shell_command","openai":"repo_search"}}
```
