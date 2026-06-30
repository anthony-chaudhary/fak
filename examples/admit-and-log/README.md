# fak kernel — the `admit_and_log` posture

**The default floor fails closed: a tool not on the allow-list is `DEFAULT_DENY`.**
For unattended batch runs that can't afford a false-deny on a harmless read, the
`fak-policy/v1` schema has one opt-in escape hatch — `"posture": "admit_and_log"`. It
admits a *read-shaped* default-deny **and stamps the verdict with what would have
happened under the strict floor**, so every admit stays auditable:

```
verdict: ALLOW   by: monitor
posture: admit_and_log (would_deny: DEFAULT_DENY)
```

The `would_deny=DEFAULT_DENY` field is the contract: a downstream log or journal can
reconstruct the strict-floor verdict from the admit record. Write-shaped calls and
explicit denials are **untouched** by the posture — they still fail closed.

```
          read-shaped, not allow-listed              write-shaped / explicit deny
                        │                                        │
   fail_closed ─────────┴──▶ DENY  DEFAULT_DENY        ──────────┴──▶ DENY  (always)
   admit_and_log ───────┴──▶ ALLOW posture=admit_and_log
                                   would_deny=DEFAULT_DENY        ──┴──▶ DENY  (still closed)
```

This demo runs the **same** read-shaped call under both postures, then shows the two
calls the posture *won't* relax. No model, no network — the kernel adjudicates a named
call from a policy file, so the verdict is deterministic.
Expected runtime: the four `preflight` witnesses complete in seconds.

## Run it

```bash
./examples/admit-and-log/run.sh
```

It needs only the `fak` binary on your `PATH` (or set `FAK=/path/to/fak`); it shells out
to `fak preflight`, which adjudicates a `(policy, tool, args)` triple and prints the
verdict. Full captured run: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The three witnesses

### 1. `fail_closed` — a read-shaped tool off the allow-list is denied

[`research-batch-fail-closed.json`](research-batch-fail-closed.json) is the strict floor
(`"posture": "fail_closed"`). `read_internal_wiki` is not on its allow-list, so it hits
the default-deny floor:

```bash
fak preflight --policy examples/admit-and-log/research-batch-fail-closed.json \
  --tool read_internal_wiki --args '{}'
# verdict=DENY reason=DEFAULT_DENY by=monitor
```

### 2. `admit_and_log` — the same call is admitted, with the would-deny stamp

[`research-batch-policy.json`](research-batch-policy.json) is identical except for
`"posture": "admit_and_log"`. The same `read_internal_wiki` call is now **admitted**, and
the verdict carries the forensic metadata. Use `--explain` (or `--json`) to see it:

```bash
fak preflight --policy examples/admit-and-log/research-batch-policy.json \
  --tool read_internal_wiki --args '{}' --explain
# verdict: ALLOW   by: monitor
# posture: admit_and_log (would_deny: DEFAULT_DENY)
# explanation: read_internal_wiki allowed under the admit-and-log posture
#              (would otherwise be DEFAULT_DENY); forensic metadata recorded.
```

The plain (non-`--explain`) line prints `verdict=ALLOW reason=NONE by=monitor`; the
`posture` / `would_deny` fields live in the structured verdict, surfaced by `--explain`
and `--json` — the form a journal actually folds.

### 3. `admit_and_log` does NOT relax write-shaped or explicitly-denied calls

The posture only downgrades a *read-shaped* default-deny. Two contrasting calls under the
**same** `admit_and_log` policy still fail closed:

```bash
# An explicit deny entry — irreversible/exfil-shaped — is untouched by the posture.
fak preflight --policy examples/admit-and-log/research-batch-policy.json \
  --tool upload_file --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor

# A write-shaped name (not read-shaped, not allow-listed) stays at the floor.
fak preflight --policy examples/admit-and-log/research-batch-policy.json \
  --tool write_report --args '{}'
# verdict=DENY reason=DEFAULT_DENY by=monitor
```

## What read-shaped means (and why the research template can't show this)

The posture admits a default-deny call **only** if its tool name starts with one of
`read_`, `get_`, `search_`, `list_`, `lookup_`, `find_`, `calc` (or is exactly
`calculate`) and is not write-shaped. That set is name-based and conservative on purpose:
the caller's metadata is model-controlled and must not be able to widen authority.

A subtlety worth calling out: the shipped [`research-agent-policy.json`](../research-agent-policy.json)
is the only template that sets `admit_and_log`, but it **also** lists those same prefixes
in `allow_prefix`. So under that template a `read_*` call is *affirmatively allowed* by a
policy rung and never reaches the posture path — you'd see `ALLOW reason=NONE`, not the
`would_deny=DEFAULT_DENY` admit. To actually witness the posture downgrade, the read-shaped
name must fall through to the default-deny floor. That's why this demo ships its own
policy: same allow-list, but **no** `allow_prefix` shadowing the read-shaped family, so the
posture is the rung that decides.

## When to use it (honest scope)

- **Intended use case.** Unattended/batch runs where a false-deny on a read is more
  expensive than a false-admit, but irreversible/exfil actions must still hard-fail.
  Think a research crawler that may call read tools you didn't enumerate, running where no
  human is watching to lift a spurious deny.
- **The metadata contract.** Every admit records `would_deny=DEFAULT_DENY`, so a
  downstream log/journal can reconstruct exactly what the strict floor would have done.
  The relaxation is auditable, not silent.
- **This is not "permit-all."** It admits *read-shaped* names only. Write-shaped calls,
  self-modify attempts, redaction hits, arg-rule violations, and every explicit `deny`
  entry still fail closed — those checks run *before* the posture path and return first.
  Keep irreversible/exfil-shaped tools off the allow-list and `DEFAULT_DENY` (now logged,
  not silently admitted) still holds them.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: runs the three `fak preflight` witnesses |
| `research-batch-policy.json` | the `admit_and_log` floor (no `allow_prefix` shadowing) |
| `research-batch-fail-closed.json` | the same floor at `fail_closed`, for the contrast |
| `EXAMPLE-OUTPUT.md` | a captured run |

Related: [`POLICY.md`](../../POLICY.md) (the `posture` row + the read-shaped set),
[`research-agent-policy.json`](../research-agent-policy.json) (the only shipped template
that sets the posture), and [`adjudication-demo/`](../adjudication-demo/) (the call-side
capability gate against a real model).
