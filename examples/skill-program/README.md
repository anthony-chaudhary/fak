# Explicit skill-program example

This example demonstrates the four separate gates for a custom tool:

1. the `fak` executable is installed;
2. policy allows its canonical operation;
3. `fak skill compile` selects it into this request's tool snapshot;
4. the provider/model receives that snapshot.

A skill file alone satisfies none of those gates. Its prose is usage guidance;
only the versioned `fak-program` block is compiled.

## Run

Prerequisite: an installed `fak` executable on `PATH`. This is a local compile
check; it needs no provider, model, API key, network, or GPU. From the repository
root, run:

```powershell
./examples/skill-program/run.ps1
```

The runner compiles both the hidden default snapshot and the explicitly exposed
Codex snapshot. It exits `0` only if registration remains hidden by default, the
Codex alias is `functions.shell_command`, canonical identity remains
`repo_search`, and executor argv is absent from the provider-visible tool.

The two-snapshot runner took 0.41 seconds in the observed warm Windows run on
2026-08-20; expect it to finish in under one second. The
compiler is deterministic for the same `SKILL.md` bytes and dialect/exposure
flags, so digests and summarized verdicts are stable across re-runs. The runner
creates no state and is safe to re-run. See [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md)
and the deeper [custom-tool registration design note](../../docs/notes/CUSTOM-TOOL-REGISTRATION-SKILL-PROGRAMS-2026-08-14.md).

## What you see

The first `PASS` proves registration did not imply exposure. The second proves
the explicit Codex alias still points back to canonical `repo_search`; the third
proves the provider-visible snapshot did not receive host executor arguments.

## What this does not claim

This demo does not prove that policy permits `repo_search`, that the executor is
installed and runnable, or that a provider will invoke the advertised tool. It
only proves compilation, per-request exposure, aliasing, canonical identity, and
separation of host-only executor data from the model view.

## Inspect the raw snapshots

From the repository root:

```powershell
fak skill compile --json examples/skill-program/SKILL.md
# model_view.tools is empty; model_view.omitted[0].reason == NOT_SELECTED

fak skill compile --json --dialect codex --expose repo_search examples/skill-program/SKILL.md
# model_view.tools[0].name == functions.shell_command
# model_view.tools[0].canonical_name == repo_search
```

The `codex` alias is deliberately illustrative. A familiar name is valid only
when the argument and result semantics truly match the harness-native tool. Do
not use a popular name merely to exploit a model prior: that raises selection
probability while silently changing the contract. Keep the canonical identity
and registration digest authoritative at dispatch.

Self-check (PowerShell):

```powershell
$hidden = fak skill compile --json examples/skill-program/SKILL.md | ConvertFrom-Json
$shown  = fak skill compile --json --dialect codex --expose repo_search examples/skill-program/SKILL.md | ConvertFrom-Json
if ($hidden.model_view.tools.Count -ne 0) { throw 'registration leaked into exposure' }
if ($hidden.model_view.omitted[0].reason -ne 'NOT_SELECTED') { throw 'missing omission witness' }
if ($shown.model_view.tools[0].name -ne 'functions.shell_command') { throw 'dialect alias absent' }
if ($shown.model_view.tools[0].canonical_name -ne 'repo_search') { throw 'canonical identity lost' }
```
