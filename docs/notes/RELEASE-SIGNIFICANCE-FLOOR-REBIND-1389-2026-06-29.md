# release significance-floor ship rebind — #1389

On 2026-06-29 the auto-cut significance floor (issue #1389) landed on
`origin/main`, but a shared-trunk `index.lock` race swept its two files into a
peer's commit:

- commit `6c874ba5` is subjected `feat(cadence): add a scheduled milestone tick
  with durable trend accrual (fak ci)` (issue #1440) but its diff ALSO carries
  the significance-floor work — `tools/release_decide.py` (+109) and
  `tools/release_decide_test.py` (+89).

The commit is immutable on `origin/main` (already pushed — we never force-push
the shared trunk). The CODE is correct, live, and tested (16/16 green, ruff
clean), but it rides under a `(fak ci)` / #1440 subject, so the
`dos verify (fak release)` referee cannot bind #1389.

This note is the corrective witness: it binds the `(fak release)` leaf to the
real significance-floor work under a subject that names #1389.

**Witnessed deliverable:** a `BELOW_FLOOR` gate in `tools/release_decide.py` —
a window whose commits are ALL provably-trivial (docs/chore/test/style/ci/build)
is held below the floor, so the 2h auto-cut cannot mint a release from low-value
commits. Fail-safe by design via `is_significant()`: breaking changes,
unrecognized types, and bare subjects all count as significant, so the floor
only ever suppresses a provably-trivial window, never a real release.
Configurable via `FAK_RELEASE_SIGNIFICANCE_FLOOR` / `FAK_RELEASE_MIN_SUBSTANTIVE`
+ `--no-significance-floor`; `--force` overrides. Verify live:
`git show origin/main:tools/release_decide.py | grep BELOW_FLOOR`.

## Second deliverable (2026-06-30) — the min-interval debounce (AC2)

The significance floor stops *trivial* windows; it does not stop a *substantive*
window from minting a tag every 2h. Issue #1389's second acceptance criterion —
"a documented min-interval debounce between autonomous tags" — is that second
floor, and it now lands as a `TOO_SOON` blocker:

- `tools/release_context.py` surfaces `last_tag_age_seconds` (the age of the last
  reachable tag, via `%(creatordate:unix)` with a committer-date fallback).
- `tools/release_decide.py` adds a `TOO_SOON` hold: when `min_interval_hours > 0`
  and a would-release window's last tag is younger than that interval, the
  auto-cut waits. It is checked **last** and only when nothing else already holds,
  so a real blocker (CI red, drift) always wins the reason. Fail-safe: `--force`
  skips it, and an unknown/missing tag age fails **open** (does not block).
- The knob is **separate from the manual path**: off by default
  (`min_interval_hours = 0`); `.github/workflows/release-cadence.yml` arms it for
  **scheduled** ticks only via the repo variable `FAK_RELEASE_MIN_INTERVAL_HOURS`,
  so a manual `workflow_dispatch` is never debounced. Also exposed as
  `--min-interval-hours` (env `FAK_RELEASE_MIN_INTERVAL_HOURS`).

Witnessed by 8 added hermetic tests in `tools/release_decide_test.py` (recent-tag
hold, old-tag clear, off-by-default, `--force` bypass, unknown-age fail-open,
real-blocker-wins, nothing-to-ship, and the env-knob default). Verify live:
`git show <sha>:tools/release_decide.py | grep TOO_SOON`.

Refs #1389
