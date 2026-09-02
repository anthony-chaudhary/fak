---
title: "Launchguard: bounded supervisor launches"
description: "How fak launchguard applies a host-local circuit breaker to detached agent and service supervisors with persistent attempt and lease state."
---

# Launchguard: bounded supervisor launches

`internal/launchguard` is the shared host-local circuit breaker for detached
agent and service supervisors. Callers choose a stable identity, call `Admit`
before starting a process, and call the returned lease's `Finish` exactly once.
The store persists only a SHA-256 identity digest, attempt timestamps, typed
state, and the active owner PID/token.

An identity receives a bounded rolling attempt budget, exponential backoff with
bounded caller-supplied jitter, and a terminal quarantine. Quarantine never
expires automatically: inspect it with `fak launchguard status --identity ID`
and recover deliberately with `fak launchguard reset --identity ID`. Reset
refuses while an owner is active. An owner is recovered only when its PID is
dead **and** its owner record is older than the configured stale threshold.

## Supervisor migration pattern

1. Derive a stable, scrubbed identity from the service role and configuration
   generation; do not include credentials, raw argv, logs, or volatile PIDs.
2. Share one launchguard directory across every supervisor for that OS user.
3. On `admitted` or `stale-recovered`, start the process and finish the lease
   with success or failure. Treat `duplicate-active`, `backoff`, and
   `quarantined` as typed refusals and expose them to the operator.
4. Keep the outer scheduler's retry count and the script's retry loop bounded;
   launchguard is the common safety envelope, not permission to stack infinite
   supervisors.
5. Require an explicit reset after the underlying startup fault is corrected.

Crash-RSI is the first adopter. The disabled `Qwen36-NVIDIA-Node` scheduled task
remains disabled unless an operator explicitly selects it for reference or
compatibility work. Native-inference and performance paths never use
launchguard refusal as a reason to fall back automatically to llama.cpp.

## Witness

```bash
go test ./internal/launchguard -count=1
go vet ./internal/launchguard
go test ./cmd/fak -run 'TestLaunchguard|TestGuardCrashRSI|TestGuardFailureRSI' -count=1
```

The package tests cover concurrent admission, scrubbed persistence, attempt
windows, exponential backoff and jitter, terminal quarantine, stale-owner
recovery, status, explicit reset, and successful budget clearing.
