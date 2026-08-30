# Issue #10378 — first-crash RSI session witness

Date: 2026-08-30

## Delivered implementation

The implementation landed intact in `780522d6e66374296308c042c4dfdde3b21b5e98`, touching only:

- `cmd/fak/guard_child_supervision.go`
- `cmd/fak/guard_crash_rsi.go`
- `cmd/fak/guard_crash_rsi_test.go`

The landing surface produced `MESSAGE_RACE`: its generic merge subject did not retain the intended witness-grade message. Per `dos man wedge MESSAGE_RACE --explain`, this follow-up witness preserves the original commit and records the intended claim without rewriting shared history.

## Failing-before proof

At base `658def25ccfffc303ed7d85105002155c6d97ad2`, the committed test file was copied into a sanctioned detached worker worktree and run with:

```text
go test ./cmd/fak -run 'TestGuardCrashRSI' -count=1
```

Result: exit 1. Compilation failed because `guardCrashRSIMarkerEnv`, `guardCrashRSILaunch`, `guardCrashRSIRequest`, and `guardMaybeLaunchCrashRSI` did not exist before the fix.

## Passing-after proof

At implementation tip `780522d6e66374296308c042c4dfdde3b21b5e98`:

```text
fak validate --ref HEAD \
  --mine cmd/fak/guard_child_supervision.go \
  --mine cmd/fak/guard_crash_rsi.go \
  --mine cmd/fak/guard_crash_rsi_test.go \
  --test-run 'TestGuardCrashRSI|TestGuardCrashRestart' \
  --timeout 12m --json
```

Result: `ok=true`, `partial=false`, `timed_out=false`; isolated WSL build, vet, and focused tests all passed in 29.760 seconds.

## Independent read-back

An independent read-only reviewer found no functional defect and confirmed:

- only the first eligible generic crash launches RSI analysis;
- typed recovery and normal/non-crash paths bypass it;
- the tag is a stable SHA-256-derived `guard-crash-rsi/<16 hex>` value;
- the prompt explicitly targets the original `fak guard` child crash;
- raw trace, original argv, ambient environment, and provider keys are excluded;
- `FAK_GUARD_CRASH_RSI` prevents recursive spawning;
- unsupported/unsafe inputs skip and launch failure remains fail-open;
- Claude and Codex launches are asynchronous and use their supported noninteractive/read-only CLI shapes;
- tests use an injected launcher and do not invoke a provider.

The reviewer noted two non-blocking test-depth opportunities: pin the exact tag hash and directly assert the final Claude/Codex argv arrays. These do not refute the shipped behavior or current deterministic acceptance tests.
