# Codex post-tool error recurrence — 2026-08-17

## Verdict

The red rows observed on this workstation are not recurring `PostToolUse` hook failures.
The active profile (`CODEX_HOME=C:\Users\USER\.codex-anthonydefault1`) recorded 53
`ERROR` rows in the preceding 24 hours; all 53 came from
`codex_core::tools::router` after `shell_command` returned non-success (51 exit 1, one
exit 124, one interrupted process). The same window contained zero actual post-tool
hook failures. Codex CLI 0.147.0 records these normal command outcomes at ERROR level,
so every negative probe, red test, refused preview, grep miss, and timeout looks like a
new infrastructure failure after the call.

Captured command:

```text
fak-dev codex-tool-errors --codex-home %CODEX_HOME% --since 24h
logged ERROR rows : 53
tool outcomes     : 53
contract defects  : 0
post-tool hooks   : 0
verdict: OUTCOMES_NOT_HOOKS
```

## Shift-left handling

`fak-dev codex-tool-errors` now classifies the store at the source boundary instead of
asking an operator to infer cause from red UI rows. It separates:

- tool outcomes: dispatched calls that returned non-success;
- contract defects: malformed tool arguments that should be rejected before dispatch;
- actual post-tool hook failures; and
- unrelated Codex errors.

Agents should avoid using a nonzero exit as ordinary branching when the question can be
answered directly (for example, PowerShell `Test-Path` instead of a failing shell probe),
and should label intentionally negative witnesses in the command output. Real failing
tests and guard refusals remain nonzero and visible; the diagnosis changes attribution,
not semantics.

## Running dev update

The installed `C:\Users\USER\bin\fak.exe` was stale (`b6fe10685473` versus
`origin/main fda04d2a5a16`). `fak self-update --root C:\work\fak --target
C:\Users\USER\bin\fak.exe` gated and installed `fda04d2a5a16`. Existing long-lived
processes keep their already-loaded image; new invocations resolve the updated binary.
The in-repo `fak.exe` is a dirty audit-only hot copy and was intentionally not replaced.
