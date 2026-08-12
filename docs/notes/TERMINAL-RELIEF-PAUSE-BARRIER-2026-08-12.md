# Terminal relief runs inside a managed pause barrier

#6417 shipped a reversible terminal actuator, but it could only *abstain* when the
pressured host owned live agents — it had no way to relieve pressure gracefully.
#6436 makes host replacement impossible outside a lifecycle transaction.

`fak terminal-relief --apply` now:

1. creates a lifecycle transaction (`terminal-relief-<host pid>-<ns>`),
2. discovers the managed forest under the pressured host,
3. publishes durable `prepare` then `pause` requests on the fleet bus,
4. waits for a **completed** acknowledgement — plus checkpoint evidence from every
   adapter that declares `application_checkpoint` — and reads those acknowledgements
   back off the bus rather than trusting its own tally,
5. re-discovers membership, and only then calls the actuator.

A missing, refused or late acknowledgement, a dynamically added child, an unmanaged
member, or a failed checkpoint all leave the verdict at `ABSTAIN` with **zero** stop
calls, and the relief cooldown unspent. Below the pressure threshold the barrier
emits no lifecycle traffic at all.

## Captured end-to-end witness

Mixed forest (fak harness + Codex + Claude), pressured host:

```json
{
  "transaction_id": "tx-pressure-1",
  "forest_id": "forest-terminal",
  "verdict": "READY",
  "reason": "all active members quiesced, host replaced, restored, resumed, and ready",
  "stop_calls": 1,
  "restore_calls": 1,
  "acks": [
    { "member_id": "claude", "state": "completed", "readback_ref": "claude:readback" },
    { "member_id": "codex",  "state": "completed", "readback_ref": "codex:readback" },
    { "member_id": "root",   "state": "completed", "checkpoint_ref": "fak-harness:checkpoint", "readback_ref": "root:readback" }
  ],
  "readback": [
    "claude:pause", "codex:pause", "fak-harness:checkpoint",
    "claude:readiness", "codex:readiness", "fak-harness:readiness"
  ]
}
```

Durable bus tree for the same transaction:

```text
lifecycle/requests/tx-pressure-1/{claude,codex,root}.json   # prepare, then pause
lifecycle/acks/tx-pressure-1/{claude,codex,root}.json       # completed + checkpoint
```

## Operator read-back

The stall monitor logs one barrier line per sample and appends it to
`%LOCALAPPDATA%\Fleet\terminal-relief-barrier.log`:

```text
terminal-relief barrier READY transaction=tx-pressure-1 forest=forest-terminal stops=1 restores=1 readback=claude:pause,codex:pause,... reason=all active members quiesced, host replaced, restored, resumed, and ready
```

An abstention is logged the same way, so "nothing happened" is as auditable as a
replacement:

```powershell
fak terminal-relief --json
Get-Content "$env:LOCALAPPDATA\Fleet\terminal-relief-barrier.log" -Tail 20
Get-ChildItem "$env:LOCALAPPDATA\Fleet\lifecycle\acks" -Recurse
```

`--barrier-deadline` (default 30s) bounds every acknowledgement; expiry abstains.
