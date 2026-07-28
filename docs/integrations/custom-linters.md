---
title: "Custom linter subprocess ABI"
description: "The fak-custom-lint/1 wire schema for user- and agent-authored linters: a bounded, verdict-only process boundary with a minimal environment, not an OS sandbox."
---

# Custom linter subprocess ABI

The `agent-hook` seam in [Extension seams](../extension-seams.md) is the default for
user- and agent-authored linters. Its host implementation is
`internal/hooks.RunCustomLint`; the versioned wire schema is `fak-custom-lint/1`.

This is a process boundary, not a claim that the child is an OS sandbox. fak gives the
child a minimal environment (PATH, OS/temp/home essentials, and the schema marker), does
not inherit arbitrary tokens or secrets, bounds input/stdout/stderr/findings and runtime,
and treats the child as verdict-only. Operators still use normal OS containment when the
program itself is hostile.

## Request

One JSON value is written to stdin:

```json
{
  "schema": "fak-custom-lint/1",
  "hook": "pre-tool",
  "subject": {"tool": "shell", "args": {"command": "git status"}}
}
```

`subject` is intentionally host-neutral JSON. Claude, Codex, and native fak adapters map
their vendor payload into it; vendor hook JSON is not the ABI.

## Response

The linter writes exactly one JSON value to stdout:

```json
{
  "schema": "fak-custom-lint/1",
  "disposition": "deny",
  "findings": [{
    "id": "example.destructive-command",
    "severity": "error",
    "message": "recursive deletion is not admitted",
    "location": {"path": "script.ps1", "line": 4, "column": 1},
    "evidence": "Remove-Item -Recurse"
  }]
}
```

Disposition is `allow`, `deny`, or `advisory`. Findings require stable `id`, `severity`,
and `message`; location/evidence are optional. A linter never rewrites the intercepted
operation. The host alone performs or refuses the effect.

## Failure contract

Each registration declares `open` or `closed`:

- Security/integrity gates use `closed`: timeout, crash, malformed/extra JSON, overflow,
  invalid findings, or schema mismatch becomes a synthetic deny finding.
- Advisory/style checks may use `open`: the same failure becomes a synthetic advisory
  finding and execution may continue. It is still observable; it is never a silent pass.

Defaults are a 5 second deadline, 1 MiB input/stdout, 64 KiB stderr, and 256 findings.
Registrations may lower or explicitly raise these bounds. Extra environment variables
must be allowlisted in `CustomLintSpec.Env`; ambient environment is not inherited.

## Conformance witness

`cmd/customlintfixture` is deliberately capable of allow, deny, timeout, crash,
malformed output, and overflow. The host-side captured tests run all cases:

```bash
go test ./internal/hooks -run CustomLint -count=1
```

The test proves both failure policies and verifies that an ambient test secret is absent
from the child environment. Future marketplace/discovery work may describe and verify
this command, but discovery metadata must not widen its capabilities or count as a
runtime witness.
