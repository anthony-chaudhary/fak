---
title: "fak config bails: the reason token is the next step"
description: "Every fak refusal whose cause is configuration prints a stable reason token, the knobs it observed, a read-only check, and a `fak recover` command that prints or runs the fix."
---

# Config bails

A **config bail** is fak exiting early because of your *configuration* rather than
your run: a capability floor that will not parse, an env var you believe holds a
key and does not, a `--restart-on-budget` with no budget behind it.

fak refusing there is correct. A malformed floor must fail before the listener
binds, not after. What was wrong is that the refusal used to be where your
information *ended*:

```
fak: parse policy.json: unexpected end of JSON input
```

That names the failure and nothing else — not which knob carried it, not what to
read, not what clears it. Your next move is a guess, and an agent's next move is
to retry the same command.

Every config bail now renders the same block instead:

```
fak policy: --check rejected the manifest: policy floor.json: invalid manifest: unexpected EOF
  reason: POLICY_LOAD_FAILED
  config: file floor.json = did not validate  (want: a manifest whose every deny cites a closed-vocabulary reason)
  check:  fak policy --dump   # the default manifest, to diff yours against
  next:   fak recover POLICY_LOAD_FAILED --set path=floor.json
```

## Reading the block

| Line | What it is |
|---|---|
| headline | The verb that refused and what it refused to do. |
| `reason:` | A stable token from a closed vocabulary. Greppable across a fleet log, so refusals can be counted **by cause**. |
| `config:` | Every knob that participated in the decision: its **origin** (`flag` / `env` / `file` / `cwd`), the value fak **observed**, and — on the knob to change — `(want: …)`. |
| `check:` | A read-only command that shows you what fak saw. Never mutates. |
| `next:` | Not advice, a **command**. |

The `config:` block is the part a bare error cannot have. Origin matters as much
as value: `--policy` and `FAK_POLICY` are fixed in different places, and if you
set the env var you will not find it by re-reading your command line. The
observed value is what fak *read*, not what you meant — which is the whole point
when you mis-set a flag two shells ago.

## The `next:` line

`fak recover <REASON>` prints the concrete checks for that token:

```console
$ fak recover POLICY_LOAD_FAILED --set path=floor.json
recover POLICY_LOAD_FAILED (dry-run)
reason: the capability floor at --policy did not load; fak refuses to serve on a floor it could not read
commands:
  fak policy --check floor.json
    # validate the manifest and print the precise rejection (read-only)
  fak policy --dump
    # print the default manifest to diff your file against
note: a floor that fails to load is never downgraded to a permissive default — that is the refusal working, not a bug
note: every deny in the manifest must cite a closed-vocabulary reason; a free-text reason is the most common rejection
```

Dry-run is the default. `--execute` runs the steps marked safe; `--json` emits
the plan as `fak.recover.v1`; `fak recover --list` shows every token.

### `--set` binds the placeholders

A catalog is static, so config plans carry placeholders (`<path>`, `<env>`,
`<addr>`). The **bail site** knows the real values, so it prints the recovery
pre-bound — that is what the `--set path=floor.json` in the `next:` line is. You
only type `--set` yourself when invoking `fak recover` cold.

`--execute` refuses while a placeholder is still unbound rather than shelling out
a literal `<path>`, and says which name to bind:

```console
$ fak recover POLICY_LOAD_FAILED --execute
fak recover: POLICY_LOAD_FAILED needs path bound before --execute can run its commands
  next: fak recover POLICY_LOAD_FAILED --execute --set path=<value>
```

### Why so little auto-corrects

`--execute` only runs steps marked **safe**: read-only, and correct without
knowing what you *meant*. `fak policy --check` qualifies — it prints the precise
validation error and changes nothing.

Most config bails have no such step, deliberately. fak cannot invent a missing
credential, cannot pick a context budget, and cannot know whether the fix for a
bad `--restart-on-budget` is to add a budget or drop the flag. Guessing would
convert a loud refusal into a silent misconfiguration — the exact failure this
whole path exists to prevent. Those plans stay manual and spend their words on
the checks instead.

## The vocabulary

Tokens are additive-only and stable. **Emitted** means a bail site prints it
today; **catalog** means the recovery resolves but the site still prints a bare
sentence. The catalog rows are tracked in `pendingBailWiring`
(`cmd/fak/bail_test.go`), which fails if an entry goes stale — the site got wired
but the entry stayed — and fails if a new reason arrives with neither a site nor
an entry. A reason cannot quietly ship as a recovery for a refusal fak never
makes.

| Reason | Cause | Status |
|---|---|---|
| `POLICY_LOAD_FAILED` | The capability floor at `--policy` did not load. fak refuses to serve on a floor it could not read, and never downgrades to a permissive default. | emitted |
| `KEY_ENV_UNSET` | A key-bearing env var was named but is unset or empty. Naming the var is what arms the requirement. | emitted |
| `BUDGET_FLAG_INCOHERENT` | The session-budget flags contradict each other — e.g. `--restart-on-budget` with no positive `--context-budget-tokens`. | emitted |
| `WEIGHTS_REQUIRED` | The requested artifact is derived from the model's own header and no weights were given. | emitted |
| `UNAUTHENTICATED_OFF_HOST_BIND` | The requested `--addr` is reachable from off this host and no inbound token door is configured. | emitted |
| `NOT_A_WORKSPACE` | The working directory is not inside a fak workspace (no `dos.toml` upward), so the corpus, devindex, and session-state planes bind the wrong tree. A warning, not a refusal. | catalog |
| `POLICY_CHECK_NO_FILE` | `--policy-check` validates a manifest and none was given. | catalog |
| `KEY_PRINCIPAL_UNRESOLVED` | A `--key-principal` binding did not resolve, so the tenant keyset is only half armed. | catalog |
| `ADDR_REQUIRED` | `fak serve` was given no transport: `--addr` was cleared and `--stdio` was not passed. | catalog |
| `BACKEND_UNAVAILABLE` | The named `--backend` is not registered in this binary. | catalog |
| `ROUTE_MANIFEST_INVALID` | The model-routing manifest or account roster did not load. | catalog |

`fak recover --list` prints this live, merged with the tree-class tokens
(`OFF_TRUNK`, `COLLISION_RISK`, `MERGE_IN_PROGRESS`, …) that cover working-tree
refusals rather than configuration. One namespace, so `fak recover <TOKEN>`
resolves any reason fak emits.

## A few worth knowing before you hit them

**`KEY_ENV_UNSET` is almost always an environment-inheritance problem, not a
typo.** A var exported in your shell does not reach a service, a launchd/systemd
unit, or an MCP server started by your editor. Set it where fak is *launched*.
fak never prints the value, and a check that echoes the secret to confirm it is
set defeats the point.

**`NOT_A_WORKSPACE` is a warning, which is exactly why it is worth checking.**
fak still serves — against the wrong tree. When fak runs as an MCP server the cwd
is your *editor's*, not yours; set the server entry's `"cwd"` to your fak
workspace root.

**`UNAUTHENTICATED_OFF_HOST_BIND` has a sanctioned escape.** Bind loopback
(`--addr 127.0.0.1:8080`, the default), require a bearer (`--require-key-env`),
or use `--stdio`, which binds no socket at all. If the host really is meant to
serve an unauthenticated interface,
`--unsafe-allow-unauthenticated-bind` proceeds and says so loudly on every boot.
That is the intended escape, not a workaround to hide.

## Not to be confused with

`CompactReason*` in `internal/agent` is a different vocabulary: it records why
the *compactor* declined to compact a request (`too_few_msgs`, `decode_failed`,
…) and feeds gateway metrics. Those are telemetry about a request that still
succeeded. Config bails are a CLI process refusing to start.

## See also

- [Troubleshooting route](troubleshooting.md) — start here when you have a
  symptom rather than a token.
- [Serve configuration](serve-config.md) — the flags themselves.
