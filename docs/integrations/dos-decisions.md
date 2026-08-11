# fak-DOS writable decisions adapter

DOS 0.29.0's native `dos decisions --all --json` surface is read-only. The public
`fak-dos` adapter adds the missing host-write boundary without modifying the cached
plugin or a global Python installation.

```text
fak-dos decisions add --workspace . --key bench-6349-blank --action OPEN_ISSUE --severity P1 --payload '{"case":"blank-reason"}' --json
fak-dos decisions list --workspace . --json
fak-dos decisions remove --workspace . --key bench-6349-blank --json
```

`add` is idempotent by key: the same structured action, severity, and payload returns
`created:false`; conflicting reuse is rejected. `list` delegates to
`dos decisions --all --json` and appends host rows, so consumers retain every native
DOS decision and gain exact `key`, `action`, and `severity` fields. `remove` is an
idempotent cleanup operation. Host events live under the workspace's ignored DOS state
at `.dos/decisions/host.jsonl`; they never edit plans, plugin caches, or global installs.

Install with `go install github.com/anthony-chaudhary/fak/cmd/fak-dos@latest`, or run
from a checkout with `go run ./cmd/fak-dos ...`.
