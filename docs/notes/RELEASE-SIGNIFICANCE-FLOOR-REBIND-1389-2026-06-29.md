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

Refs #1389
