# On-demand releases

Use the existing release cadence workflow through the native front door:

```bash
fak release dispatch --execute --json
```

The command sends an exact, visible input set to `.github/workflows/release-cadence.yml` on
`main`: `dry_run=false`, `require_ci_green=true`, and `force=false`. The workflow owns the
release lock, CI-readiness gate, version decision, cut, tag, publish, and artifact verification.
The local peer-dirty checkout is not copied into the release: GitHub checks out the selected
ref, while the local `fak release ship` path independently cuts in a transient detached
worktree.

Preview the dispatch without touching GitHub:

```bash
fak release dispatch --json
```

Useful overrides remain explicit:

```bash
fak release dispatch --execute --plan-only  # decide and plan; do not cut
fak release dispatch --execute --ref main   # select the workflow ref explicitly
fak release dispatch --execute --force      # bypass only the substantive-commit floor
```

`--require-ci-green` defaults to true. Disable it only for a deliberate recovery release with
`--require-ci-green=false`; drift, parse, locking, and publication gates still apply. A successful
dispatch returns the exact queued run URL (and the workflow Actions URL in JSON) to check. It proves enqueueing, not publication; the
release is complete only after the workflow's publication-verification step is green.

For a local operator-driven cut, use `fak release ship --execute --json`. That command keeps the
shared checkout untouched by working from a detached clean worktree and uses the same release
substrate and single-writer lock.
