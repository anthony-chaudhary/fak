# Explicit skill-program example

This example demonstrates the four separate gates for a custom tool:

1. the `fak` executable is installed;
2. policy allows its canonical operation;
3. `fak skill compile` selects it into this request's tool snapshot;
4. the provider/model receives that snapshot.

A skill file alone satisfies none of those gates. Its prose is usage guidance;
only the versioned `fak-program` block is compiled.

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
