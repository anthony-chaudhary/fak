# Release Branch-Regime Status

`fak release status` reports two separate release facts:

- `rolling` / `latest_tag`: whether the public release tag is current.
- `branch_regime`: whether the development branch has drifted from the release branch, and whether a promotion is currently blocked.

The branch-regime fields are additive JSON:

- `development_branch`, `development_head`: the hot integration role and resolved commit.
- `release_branch`, `release_head`: the public/front-door role and resolved commit.
- `development_ahead`, `release_ahead`, `drift`: git-derived branch distance.
- `promotion_blocked`, `promotion_blockers`: why a dev -> main promotion should hold.
- `release_lock_held`: whether another release writer owns the single-writer lock.

Operator actions:

| Status | Action |
|---|---|
| `drift=no_drift` | Hold; development and release heads match. |
| `drift=development_ahead` and no blockers | Treat `promotion_candidate` as the source SHA for a release-promotion dry run. |
| `RELEASE_AHEAD` | Stop and inspect; the release/front-door branch has commits not in development. |
| `DEVELOPMENT_CI_RED` | Fix or confirm CI on the development head before promotion. |
| `RELEASE_LOCK_HELD` | Wait for the active release writer or inspect `python tools/release_lock.py status`. |
| `*_HEAD_UNKNOWN` or `BRANCH_ROLE_CONFIG` | Refresh refs or fix `dos.toml [branch_roles]` before trusting the status. |

Do not read a quiet `main` as "nothing shipped" once `development_branch` is `dev`; read `branch_regime.development_ahead` and `promotion_blockers` instead.
