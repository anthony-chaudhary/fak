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

Tokens are additive-only and stable, and every one is emitted by a real bail
site. `TestConfigBailReasonsAreEmitted` (`cmd/fak/bail_test.go`) holds that
property: a reason with a plan but no site fails the build unless it is
explicitly declared in `pendingBailWiring` with the site it is waiting on. So a
reason cannot quietly ship as a recovery for a refusal fak never makes, and a
deliberate gap has to be recorded rather than discovered later.

| Reason | Cause | Where it fires |
|---|---|---|
| `POLICY_LOAD_FAILED` | The capability floor at `--policy` did not load. fak refuses to serve on a floor it could not read, and never downgrades to a permissive default. | any verb that installs a floor (`serve`, `chat`, `attest`, the agent verbs), and `fak policy --check` |
| `KEY_ENV_UNSET` | A key-bearing env var was named but is unset or empty. Naming the var is what arms the requirement. | `serve --require-key-env`, `serve --engine-cache-admin-key-env`, `console agent` against a credential-gated target |
| `KEY_PRINCIPAL_UNRESOLVED` | A `--key-principal` binding did not resolve, so the tenant keyset is only half armed. | `serve --key-principal` |
| `BUDGET_FLAG_INCOHERENT` | The session-budget flags contradict each other — e.g. a reset/restart-on-budget with no positive `--context-budget-tokens`. | `serve`, `console agent` |
| `ADDR_REQUIRED` | No transport: `--addr` is empty and `--stdio` was not passed. | `serve` |
| `UNAUTHENTICATED_OFF_HOST_BIND` | The requested `--addr` is reachable from off this host and no inbound token door is configured. | `serve` bind guard |
| `BACKEND_UNAVAILABLE` | The named `--backend` is not registered in this binary. | `serve`, `serve --plan-json` |
| `ROUTE_MANIFEST_INVALID` | The model-routing manifest or account roster did not load. | `serve --route-manifest`, `serve --route-accounts` |
| `WEIGHTS_REQUIRED` | The requested artifact is derived from the model's own header and no weights were given. | `serve --plan-json` |
| `POLICY_CHECK_NO_FILE` | `--policy-check` validates a manifest and none was given. | `serve --policy-check` |
| `NOT_A_WORKSPACE` | The working directory is not inside a fak workspace (no `dos.toml` upward), so the corpus, devindex, and session-state planes bind the wrong tree. A warning, not a refusal. | `serve` startup |
| `UPSTREAM_TRUST_UNVERIFIED` | A corporate CA bundle was declared in `FAK_CA_BUNDLE` and fak is not validating with it — the file did not read, held no `CERTIFICATE` block, or the platform trust store it must widen was unavailable. | `guard` launch, `doctor trust` |
| `UPSTREAM_UNSUPPORTED` | The wrapped agent is routed to a request-signed cloud gateway (Bedrock SigV4, Vertex ADC), so it ignores `ANTHROPIC_BASE_URL` and fak's gateway would adjudicate nothing. | `guard` launch |

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

**`UPSTREAM_TRUST_UNVERIFIED` is a certificate problem on the host, not a fak
misconfiguration.** Behind a TLS-intercepting proxy nothing is blocked — a private
root re-signs every connection and the chain simply does not validate. Every tool
reports it differently (`x509: certificate signed by unknown authority`,
`SELF_SIGNED_CERT_IN_CHAIN`, `CRYPT_E_NO_REVOCATION_CHECK`, `self-signed
certificate in certificate chain`) and all four read as "the network is down".
Point `FAK_CA_BUNDLE` at a PEM file holding your corporate root; fak validates
against the platform store **plus** that root, never the root alone, and derives
the per-runtime variables children need from the same file. There is deliberately
no skip-verify escape. See [Managed hosts and corporate TLS](managed-hosts.md).

**`UPSTREAM_UNSUPPORTED` is not a credential failure.** Your cloud credential is
fine. A child running with `CLAUDE_CODE_USE_BEDROCK=1` (or `CLAUDE_CODE_USE_VERTEX=1`)
signs each request itself and never reads `ANTHROPIC_BASE_URL`, so fak's gateway
repoint is inert and there is no traffic for it to adjudicate. The supported route
on that posture is `fak serve --stdio --policy FILE` — fak as an MCP server the
agent calls, which is provider-agnostic. To run the guard anyway for its other
properties with the model traffic unadjudicated,
`FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1` proceeds and says so on every launch.

## Not to be confused with

`CompactReason*` in `internal/agent` is a different vocabulary: it records why
the *compactor* declined to compact a request (`too_few_msgs`, `decode_failed`,
…) and feeds gateway metrics. Those are telemetry about a request that still
succeeded. Config bails are a CLI process refusing to start.

## See also

- [Troubleshooting route](troubleshooting.md) — start here when you have a
  symptom rather than a token.
- [Serve configuration](serve-config.md) — the flags themselves.
