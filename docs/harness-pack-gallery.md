# Harness pack gallery

`fak harness gallery` turns representative user needs into bounded, user-owned starter
packs. It answers “what kind of harness should I create?” before the builder chooses an
adapter or edits runtime code.

```text
fak harness gallery list
fak harness gallery show --id readonly-support
fak harness gallery init --id coding-workspace --dir ./my-pack
fak harness gallery selfcheck
```

The four built-ins deliberately select different seams:

| ID | Need | Public seam |
|---|---|---|
| `readonly-support` | grounded customer answers without writes | generated config plus policy |
| `coding-workspace` | local browser coding loop with bounded tools | harnesskit UI plus workspace/tool adapters |
| `cited-research` | reproducible primary-source synthesis | skill pack plus trace/artifact sinks |
| `incident-operations` | diagnosis and approved remediation | policy, approval, telemetry, and secret adapters |

`init` creates `harness.pack.json` and `README.md` only when absent. Both are user-owned;
a rerun preserves them. The manifest records required and excluded capabilities, the
selected seam, witness, and exact regeneration command. These are decision-grade examples,
not fabricated implementation claims: each README separates the ten-minute spine from its
weekend extension.
