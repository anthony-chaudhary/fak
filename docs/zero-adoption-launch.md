---
title: "Zero-adoption provider launch with fak launch"
description: "Put the fak manage in front of an existing Claude Code or Codex install using PATH shims, without changing the command your users already type."
---

# Zero-adoption provider launch

`fak launch` can put the fak manage in front of an existing Claude Code or Codex
installation without changing the command users type.

```sh
fak launch install --provider all --default claude
# Ensure the printed shim directory precedes the provider's own directory on PATH.
claude                 # now: fak manage -- <original claude> ...
fak                    # launches the configured default provider
```

The installer records the already-resolved provider executable, creates a small shim,
and idempotently adds the shim directory to supported PowerShell/bash/zsh/fish startup
files inside a clearly delimited fak-owned block. It prints the one-line command for the
current shell because startup-file edits cannot mutate an already-running process. Use
`--no-path` for managed environments that own PATH themselves. Uninstall removes only the
fak block and converges as a no-op when it is already absent; user bytes around it remain
unchanged.

It never renames or overwrites the provider binary. `fak launch status` shows
the exact recorded paths. `fak launch uninstall --provider all` removes the shims
and configuration bindings.

## Which Codex command?

These entry points are intentionally different; they are not aliases:

| Command | Role | Pipeline | Use it when |
|---|---|---|---|
| `codex` | **Canonical** zero-adoption front door | managed shim -> `fak launch codex` -> `fak guard` -> recorded provider | Normal interactive Codex use after `fak launch install`. |
| `fak m codex` | Noncanonical general manage surface | `fak manage` -> `fak guard` -> `codex` from `PATH` | Explicit guard/manage experimentation. On Windows, managed-wrapper resolution remains tracked by #8866. |
| `fak codex` | Specialized Codex loop surface | freshness admission -> loop gate -> `fak guard` -> `codex` from `PATH` | You specifically need checkout freshness, loop-gate, split-pane, or resume translation behavior. |

Run `fak launch doctor` before changing launch wiring. Its versioned `--json` output
includes an `entry_points` matrix with each command's role, pipeline, readiness, reason,
and recovery action. Tests keep this matrix unique and deterministic; readiness and
operator scorecards can consume it instead of inferring parity from similar command names.
A provider-level `READY` result proves the managed bare command, not that every wrapper
which resolves the provider again through `PATH` is safe.

Bypass is deliberately available at three scopes:

```sh
claude --fak-direct       # this invocation only
FAK_DIRECT=1 claude       # this environment/session
fak launch disable       # persisted; every shim/TUI-launched provider passes through
fak launch enable        # restore interception
```

The same one-shot escape works without a shim as `fak launch --direct claude ...`.
This provides a recovery path even when the guard or its TUI is unwanted. The
special `--fak-direct` token is consumed by the shim and is not passed to the
underlying provider.

Choose or change bare-`fak` behavior with `fak launch default claude` or
`fak launch default codex`. Add a third provider without waiting for a fak release:

```sh
fak launch add qwen-local --command /opt/qwen/bin/agent --arg --profile --arg coding --default --shim
fak launch list --json       # redacted: names and argument counts, never local paths/arguments
qwen-local "fix the test"    # template argv first, then user argv; no shell evaluation
fak launch remove qwen-local # also removes its owned shim and clears it as default
```

Alias names must match `[a-z][a-z0-9-]*`, cannot be paths, and cannot shadow reserved
`fak` verbs. Repeatable `--arg` values are persisted as an argv array; spaces, Unicode,
quotes, and leading dashes retain exact argument boundaries. Custom aliases inherit
`--fak-direct`, `FAK_DIRECT`, `launch disable|enable`, status, doctor, and uninstall
behavior from the built-ins.

Configuration is stored in the platform user config
directory under `fak/launch.json`; `FAK_LAUNCH_CONFIG` and `FAK_LAUNCH_BIN` are
available for managed installs and tests.


## Diagnose launch posture

```bash
fak launch doctor
fak launch doctor --json
fak launch help
```

Generated shims target the managed `fak-launch` copy in the shim directory rather than
the transient package-manager or `go install` source path. After replacing or moving fak,
run `fak launch doctor --repair` once to refresh that stable copy and every owned shim;
provider bindings and the direct escape remain untouched. Launch config is schema-versioned
and transparently migrates the original unversioned shape to `fak.launch.v2` on the next write.

Doctor checks each provider without launching it and reports one of `READY`,
`NOT_ON_PATH`, `SHADOWED`, `UNDERLYING_MISSING`, `RECURSIVE`, `DISABLED`, or
`CONFIG_INVALID`, plus one recovery command for every non-ready row. Its versioned JSON
redacts local paths to basenames and never includes prompts or forwarded arguments.
