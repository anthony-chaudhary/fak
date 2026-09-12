---
title: "Serve help category baseline gap"
description: "Track the independently reproduced 16-flag serve help inventory failure."
---

# Serve help category baseline gap

Ref #12851. Status: open. Coordinates with #12581 and #12832.

The existing `TestEveryServeFlagIsShiftedIntoAHelpCategory` fails on clean
public baseline `4a3a6aa6c6c36261eb27143f6ceab1c73612588e` under Linux/amd64,
Go 1.26.6. The relevant `cmd/fak/serve.go`, `serve_help.go`, and
`serve_help_test.go` files are identical to those at
`71c9e905d7014eb8e77cee1f9341b29ed5379cdc`.

Witness:

```text
go test -race -count=1 -p=1 -timeout=10m ./cmd/fak -run '^TestEveryServeFlagIsShiftedIntoAHelpCategory$'
FAIL: TestEveryServeFlagIsShiftedIntoAHelpCategory
FAIL github.com/anthony-chaudhary/fak/cmd/fak 0.239s
exit 1
```

All failures point to `serve_help_test.go:58`, with these uncategorized flags:

- `appliance-observability`
- `claude`, `claude-config`, `write-claude-config`
- `codex`, `codex-config`, `codex-config-path`, `write-codex-config`
- `pi`, `pi-config`, `pi-config-path`, `write-pi-config`
- `ctx`
- `max-batch-prefill-tokens`, `max-total-tokens`
- `memory-governor`

The #12581 candidate reproduced the same list. Its sole help-category change
adds `print-features`, which is not in the failure list. This is separate
from the eleven Windows CLI baseline failures in #12832. Targeted feature
test success does not establish a green full CLI suite.

The next bounded change belongs in `cmd/fak/serve_help.go`: place each flag
in its appropriate existing category, preserving the exhaustive assertion
and concise category output. Acceptance:

```text
go test -race -count=1 ./cmd/fak -run 'TestEveryServeFlagIsShiftedIntoAHelpCategory|TestServeHelp'
```

The baseline test remains unchanged in #12581; repairing these categories is
the follow-up tracked by #12851.
