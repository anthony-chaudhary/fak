# Captured selfcheck

Captured from the repository root on Windows with Go 1.26.7. The command is
offline and exits 0 only after the scoped lookup returns a canonical term with a
contrast. Its fixed query and local public index make this output reproducible.

```console
$ go run ./cmd/disambiguationdemo -selfcheck
QUERY runtime: ambiguous choices=[agent-application gateway-serving guard-enforcement model-serving worker-execution]
SCOPE runtime=gateway-serving -> runtime
CONTRAST The gateway runtime is a long-lived transport server; the fak CLI kernel is the command surface used to configure and launch it.
SELFCHECK PASS: public local index only; no model, key, GPU, network, or private data
```

Observed warm-run wall time: 1.9 seconds. A cold Go toolchain download or first
compile is outside that timing.
