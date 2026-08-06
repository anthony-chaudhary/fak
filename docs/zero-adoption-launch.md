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
`fak launch default codex`. Configuration is stored in the platform user config
directory under `fak/launch.json`; `FAK_LAUNCH_CONFIG` and `FAK_LAUNCH_BIN` are
available for managed installs and tests.
