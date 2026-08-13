# Explainable continuity

`fak profile continuity` is one task-oriented front door over the shipped portability foundations. It does not replace the Object contract, package format, reconciliation, egress, organization, registry, or adapter seams.

## Six nouns, five starting tasks

| Noun | Meaning |
|---|---|
| **Object** | one managed skill, workflow, policy, or adapter |
| **Collection** | named Objects governed together |
| **Context** | the active resolved Collection |
| **Package** | portable, inspectable Context snapshot |
| **Channel** | private, organization, or public route |
| **Transaction** | preview, receipt, and rollback boundary |

Start with `backup`, `restore`, `switch`, `share`, or `publish`. Every mutation previews unless `--commit` is present. `status`, `preview`, `explain`, receipts, `recover`, and `rollback` use the same reason-code model. Ordinary failures print at most two `Next:` actions; log reading is not a recovery step.

```text
$ fak profile continuity explain --reason EGRESS_DENIED
EGRESS_DENIED — Channel policy prevents sensitive Object data leaving its scope
  Next: preview redactions
  Next: choose a narrower Channel
```

```json
$ fak profile continuity explain --reason INTERRUPTED --json
{
  "code": "INTERRUPTED",
  "meaning": "a Transaction stopped before activation; the prior Context remains active",
  "next_actions": ["run recover", "inspect the Transaction receipt"]
}
```

Progressive disclosure preserves expert control: repeatable `--select kind[:name]`, `--channel`, explicit package and receipt IDs, sync policy/precedence controls, `--commit`, and `--json` remain auditable. `backup` aliases export; `restore` aliases apply; `share` and `publish` select organization and public Channels; `recover` aliases receipt-based rollback.

## Captured witness

`fak profile continuity ux-selfcheck` and `--json` replay the checked-in #6597 deterministic proxy budget. The candidate requires six visible nouns rather than ten recalled subsystem concepts and three decisions rather than eight, while all eight expert-control categories remain represented (8→8). Fixtures capture 40/80/120-column output, no-color meaning, reason codes, bounded next actions, and human/JSON parity. The machine capture is [`docs/_witnesses/portability-continuity-selfcheck-2026-08-13.json`](_witnesses/portability-continuity-selfcheck-2026-08-13.json).
