# Fleet repository-targeting incident — 2026-08-13

## Verdict

A fleet intended only for `anthony-chaudhary/fak` widened into sibling repositories because repository identity was implicit. Owner-global issue discovery and workspace paths were treated as authority. They are not. The legacy `anthony-chaudhary/fleet-public` repository was tombstoned on 2026-08-13, its remaining issues were administratively closed, and its GitHub repository was archived with Issues and Projects disabled.

## What happened

The operator's objective was repository-scoped: work only in `anthony-chaudhary/fak`. Selection nevertheless considered repositories under the same owner, including `fleet-public`, because the target was reconstructed from ambient cwd/path and broad issue queries rather than carried as an explicit run invariant. A sibling directory looked actionable even though it was obsolete. Empty or exhausted work in the intended repository could therefore widen discovery instead of draining.

The closures in `fleet-public` are tombstone administration, not implementation claims. Maintained work remains in `anthony-chaudhary/fak`; anything genuinely absent must be deduplicated there against current evidence.

## Deterministic contract

Every repository-scoped fleet run now follows these invariants:

1. **One run, one declared target.** Planning receives an immutable GitHub `owner/repo` identity.
2. **Origin corroborates authority.** Before `fak superloop drive --execute` walks candidates, takes a lease, writes its ledger, or runs a front door, the selected workspace's `remote.origin.url` must normalize to the declared `--repo`.
3. **Queries are repository-scoped.** Candidate discovery uses an explicit `-R owner/repo`; an owner-wide search is evidence gathering, never an execution queue.
4. **Prompts carry the target.** Every worker objective names the same `owner/repo`; cwd and sibling paths do not amend it.
5. **Effects are folded by target.** Issue closure and ship witnesses are accepted only when GitHub `repository.nameWithOwner` equals the declared target.
6. **Tombstones are excluded.** Archived repositories and repositories with a tombstone notice are never candidate sources.
7. **Drain does not widen.** An empty target backlog stops/drains. Changing repositories requires a new explicit operator command, never fallback discovery.

## Executable guard

`fak superloop drive --execute` requires `--repo owner/name`. It resolves origin with `git -C <workspace> config --get remote.origin.url`, normalizes supported GitHub HTTPS/SSH forms, and refuses missing, unknown, invalid, or mismatched targets before any fleet effect. Dry-run drives remain compatible so operators can inspect an unbound plan; execution is fail-closed.

This guard covers the durable drive front door. Callers must still preserve the same declared target through candidate query, worker prompt, and closure fold; passing the initial handshake is necessary, not permission to infer a different target later.

## Witness

- Remote tombstone: `anthony-chaudhary/fleet-public` README and `SUNSET.md` name `anthony-chaudhary/fak` as canonical.
- GitHub state: zero open issues, archived, Issues disabled, Projects disabled, homepage points to `anthony-chaudhary/fak`.
- Local behavior: execute without `--repo` returns `REPO_TARGET_REQUIRED`; mismatched origin returns `REPO_TARGET_MISMATCH`; matching HTTPS and SSH origins pass the target gate; refusal leaves no ledger.
