# Captured skill-program self-check

Command, run from the repository root on 2026-08-20:

```powershell
./examples/skill-program/run.ps1
```

Output:

```text
PASS hidden snapshot: tools=0 omitted=repo_search reason=NOT_SELECTED
PASS codex snapshot: name=functions.shell_command canonical=repo_search
PASS executor isolation: provider-visible tool contains no executor argv
```

Exit code `0` means all four snapshot invariants were observed. A compile error,
implicit exposure, alias/canonical-name drift, or executor-data leak makes the
runner exit nonzero.
