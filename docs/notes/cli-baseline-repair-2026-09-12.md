# CLI baseline repair acceptance receipt

Date: 2026-09-12. Parent: [#12832](https://github.com/anthony-chaudhary/fak/issues/12832).

The following six test-only fixes are ancestors of `origin/main`. Each closes one independent baseline failure cohort. This receipt records scoped software verification; it does not establish full CLI-suite or physical desktop qualification.

| Issue | Resolving commit | Acceptance evidence |
| --- | --- | --- |
| [#12842](https://github.com/anthony-chaudhary/fak/issues/12842) | `97b9874e500bcee4622be66dd0997fa70ba7f37d` | Function and command identity scope the Windows spawn audit. Background and detached configurations pass; ordinary and shadowed unconfigured helpers fail. Positive inheritance assertions retain the intentional selfcheck descendant. Production process configuration is unchanged. |
| [#12846](https://github.com/anthony-chaudhary/fak/issues/12846) | `f4e148a69d9bc0e664c3f65a6da00a575511be6e` | Both front doors use structural gateway configuration matching. Formatting variants pass; wrong variables, false literals, comments and unrelated literals fail. Default-on and opt-out assertions remain. |
| [#12844](https://github.com/anthony-chaudhary/fak/issues/12844) | `278aeb2d45ff0348dbed31a3a557e7d5db18d908` | Independently allocated paired temporary workspace fixtures contain Go units. Explicit roots replace developer checkout discovery. Both repository tags and nonempty companion results remain asserted. |
| [#12849](https://github.com/anthony-chaudhary/fak/issues/12849) | `d4bac5714f279cb8bc7b7cfc799fa8cceafc7307` | The active fixture invokes the Go test binary as a harmless child that emits `ses_active_123`. Active execution succeeds, expired execution remains prevented, and success receipt coverage passes. Production subprocess behavior is unchanged. |
| [#12845](https://github.com/anthony-chaudhary/fak/issues/12845) | `61224ed684f8f0b27deb0ce842bdb41e9a6d716a` | Coverage receipt assertions compare exact sorted canonical dimension names, reject duplicates, check schema and depth, and explicitly require the ungated performance benchmark dimension. Production detectors remain unchanged. |
| [#12850](https://github.com/anthony-chaudhary/fak/issues/12850) | `8020d200dadad58453daa36db6f8fd577b16308e` | Three original spawn-policy fixtures reach fake spawners with managed preparation arguments, default isolation, cwd and environment asserted. Preparation failure remains terminal and preparation globals restore through cleanup. No real worker is launched. |

## Reproducible scoped witnesses

Run on Windows from the repository root, with `FAK_PRIVATE_ROOT` unset for the fixture cohort:

```text
go test ./cmd/fak -run '^(TestWindowsBackgroundSpawnsSuppressConsoleWindows|TestTerminalReliefSpawnModesSuppressConsoleWindows|TestWindowgateSelfcheckReportJSON|TestBackgroundConfigured.*)$' -count=1 -timeout=90s
go test ./internal/windowgate -run '^TestConfigure(BackgroundCommandSetsWindowsNoWindow|DetachedCommandClearsNoWindowAndDetaches)$' -count=1 -timeout=90s
go test ./cmd/fak -run '^TestDeferColdTools' -count=1 -timeout=90s
go test ./cmd/fak -run '^(TestDebtOrchestratorCLIDualRepo|TestDebtLanesCLITargetRepoPrivate)$' -count=1 -timeout=90s
go test ./cmd/fak -run '^TestCronOpenCode(Success|UntilActive|UntilExpiration)$' -count=1 -timeout=90s
```

Independent verification executed every listed test on Windows with exit zero: the combined CLI cohort passed in 0.103s, windowgate flags in 0.023s, and the cron cohort in 0.208s. The companion configuration environment variable was unset for the combined CLI run; its fixture roots are independent temporary directories.

The cron command intentionally uses the existing test name `UntilExpiration`; the issue's original `UntilExpired` selector omitted that test. The corrected three-test cohort passed independently with exit zero. Other parent cohorts retain their own acceptance gates.

## Additional landed cohorts

The coverage receipt witness passed in 0.091s. The dispatch witness (including owner handoff) passed in 4.713s; related guard fixtures passed in 7.783s. Retained execution receipts and landed source were independently read back.

```text
go test ./cmd/fak -run '^TestDebtLanesCLICoverageReceipt$' -count=1 -timeout=90s
go test ./cmd/fak -run '^(TestDispatchCommandExecutedLiveCodexAllowsGuardedSubscriptionChildFromUnguardedParent|TestDispatchTickCodexLoopGateDefaultOffSkipsAudit|TestDispatchTickLiveFailsNonzeroEarlyExitAndPinsClaudeAccountEnv|TestDispatchTickWorktreePrepareFailureIsTerminal|TestWorkerWorktreeEnabledGrammar|TestDispatchWorktreeOwnerHandoffProtectsSpawnedWorker)$' -count=1 -timeout=120s
```