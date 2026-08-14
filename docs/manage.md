# `fak manage`: the managed-agent front door

`fak manage` is the primary command for running and operating an agent through fak.
Its short spelling is `fak m`:

```text
fak manage codex
fak m codex
fak manage claude
```

`manage` provides the complete command surface historically exposed by the
legacy compatibility spelling: all launch flags, wrapped-agent behavior, and operator subcommands such
as `allow`, `deny`, `policy`, `compile`, `sessions`, `resume`, and `restart-audit` use
the same implementation. The optional `--` separator remains supported when flags
or child arguments make the boundary useful:

```text
fak manage --provider openai -- codex
fak m --policy examples/customer-support-readonly-policy.json -- claude
```

## Guard sunset

The legacy `guard` front-door spelling remains a behavior-compatible alias. Its help points to
`manage`/`m`, and it is not removed in this migration. New documentation, scripts,
and operator instructions should use `fak manage` or `fak m`; existing automation
can migrate without an immediate flag or behavior change.

The broader name is intentional: fak manages performance, context, provider routing,
policy, evidence, and agent lifecycle. Policy guarding is one managed capability,
not the name of the whole value surface.
