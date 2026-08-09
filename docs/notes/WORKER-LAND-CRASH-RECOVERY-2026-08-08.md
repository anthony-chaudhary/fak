# Worker-land crash recovery — 2026-08-08

`fak worktree worker land` uses an isolated index and `git commit-tree` before it
compare-and-swap updates trunk. Every candidate commit is anchored first under:

```
refs/fak/worker-land/<worktree-name>/<candidate-sha>
```

The local named ref protects process/session crashes and Git GC. For host/disk
loss, opt into remote publish and independent read-back before trunk CAS:

```bash
fak worktree worker land --worktree D --recovery-remote origin ...
# fail closed instead of proceeding LOCAL_ONLY when the remote cannot witness it:
fak worktree worker land --worktree D --recovery-remote origin --require-remote-recovery ...
```

A successful remote claim means `git ls-remote` returned the exact candidate SHA.
The read-only local mirror lives under `refs/fak/remoteworkerland/<remote>/...` and
its last successful refresh has a reflogged stamp under
`refs/fak/remoteworkerland-stamp/<remote>`. Remote/network failure never removes
the local ref; best-effort mode lands and reports the failed receipt, while
required mode refuses trunk CAS.

## Resume / inspect

```bash
fak worktree worker list
fak worktree worker recover
fak worktree worker recover --remote origin --fetch
```

Recovery JSON classifies:

- `LOCAL_ONLY`: process-crash safe, not yet host-loss witnessed;
- `REPLICATED`: exact SHA exists locally and in the read-back mirror;
- `REMOTE_ONLY`: fresh clone/host-loss case; restore the printed local ref then land;
- `LANDED`: Git proves the candidate is already in current `HEAD`.

Inspect any candidate with `git show <ref>`. A `RECOVERABLE` local candidate can
be re-landed from its worker or operator-reviewed with `git cherry-pick <ref>`.
A remote-only candidate's printed action uses `git update-ref` to restore the
local recovery root before inspection/landing.

## Guarded cleanup

Local cleanup refuses an unlanded candidate unless explicitly forced:

```bash
fak worktree worker recover --cleanup refs/fak/worker-land/<worktree>/<sha>
fak worktree worker recover --cleanup refs/fak/worker-land/<worktree>/<sha> --force
```

Remote cleanup is report-only by default. It fetches the remote default branch
and requires the candidate to be its ancestor. Peer-named refs require an extra
`--allow-peer`; age/lease expiry is never deletion proof because the ref is the
work itself:

```bash
fak worktree worker recover --remote origin \
  --cleanup-remote refs/fak/worker-land/<worktree>/<sha> \
  --worktree-name <worktree>
# only after reviewing eligible=true:
fak worktree worker recover --remote origin \
  --cleanup-remote refs/fak/worker-land/<worktree>/<sha> \
  --worktree-name <worktree> --apply
```

The original recovery ref is intentionally retained after successful trunk CAS;
ancestry read-back then classifies it `LANDED`, preventing duplicate application
while preserving an audit/recovery root until explicit cleanup.

