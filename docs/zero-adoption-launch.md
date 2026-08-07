# Zero-adoption provider launch

`fak launch` can put the fak guard in front of an existing Claude Code or Codex
installation without changing the command users type.

```sh
fak launch install --provider all --default claude
# Ensure the printed shim directory precedes the provider's own directory on PATH.
claude                 # now: fak guard -- <original claude> ...
fak                    # launches the configured default provider
```

The installer records the already-resolved provider executable and creates a small
shim; it never renames or overwrites the provider binary. `fak launch status` shows
the exact recorded paths. `fak launch uninstall --provider all` removes the shims
and configuration bindings.

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

Doctor checks each provider without launching it and reports one of `READY`,
`NOT_ON_PATH`, `SHADOWED`, `UNDERLYING_MISSING`, `RECURSIVE`, `DISABLED`, or
`CONFIG_INVALID`, plus one recovery command for every non-ready row. Its versioned JSON
redacts local paths to basenames and never includes prompts or forwarded arguments.
