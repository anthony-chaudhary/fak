# Move a managed context between homes

`fak profile continuity` is the task-oriented personal-continuity front door. It discovers the managed **skills**, **workflows**, and **policies** already present in one fak home, previews what is safe, exports a versioned portable package, restores it into another home, and switches or rolls back the active context. It uses `fak.portability/v1` from #6598; it does not invent another interchange schema or require a service.

```text
fak profile continuity preview --home HOME-A
fak profile continuity export  --home HOME-A --out context.fakpkg.json --commit
fak profile continuity apply   --home HOME-B --package context.fakpkg.json --commit
fak profile continuity switch  --home HOME-B --package pkg-... --commit
fak profile continuity status  --home HOME-B
fak profile continuity rollback --home HOME-B --receipt rcpt-... --commit
```

Mutation is a dry-run unless `--commit` is explicit. `--json` provides stable structured output and repeatable `--select kind[:name]` narrows discovery/export. Export, apply, switch, and rollback write immutable JSON receipts under `HOME/receipts/`; the switch receipt is the rollback handle.

## Safety and recovery

Export and apply fail closed on credential/token fields or values, private hostnames, absolute host paths, and undeclared history/transcript data. A malformed, incompatible, or digest-tampered package cannot alter the active context. Apply stages all objects before rename, so interruption leaves the prior context active and a retry is safe. Reapplying an identical package reports `already applied`. Unknown object kinds remain in the restored context for inspection but are inactive and absent from behavior read-back.

Run the captured clean-room witness without a service:

```text
fak profile continuity selfcheck
```

It builds three real managed object files in the source home, drives the real export/apply/switch/read-back/rollback objects in a second isolated home, and prints the four durable receipt IDs.
