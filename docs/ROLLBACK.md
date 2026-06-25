---
title: "fak rollback runbook: revert, downgrade, pin"
description: "How fak operators recover from a bad trunk change: revert commits, downgrade to stable tags, pin fleet versions, and repair measured state."
---

# Rollback runbook — getting back to a known-good state

When a change on the fast-moving trunk goes wrong — a regression a gate caught, a
"tail wags the dog" design that turned out worse, a confusing half-state — this is how
an operator gets fak back to a stable version. The mechanisms below already exist in the
tree; this page is the one place that tells you which to reach for and in what order.

Reach for the cheapest layer that fixes your problem. Most incidents stop at layer 1.

| Layer | Use when | Command |
|---|---|---|
| 1. Revert the commit | a specific bad commit is on the trunk | `git revert <sha>` |
| 2. Downgrade to a stable tag | the trunk is unstable; you need a known-good build | `git checkout v0.31.0` |
| 3. Pin a stable version | a fleet should hold a version and not auto-upgrade | `FAK_APP_VERSION=0.31.0` |
| 4. Revert measured state | a KPI / scorecard regressed | the keep/revert ladder + a re-pin |

## 1. Revert the commit (the trunk default)

The trunk (`main`) is a shared, multi-session tree and the trunk guard refuses off-trunk
work, so **never rewrite history to undo a change** — no force-push, no `reset --hard` on a
pushed commit. Roll the change back the same way you ship one: with a new, signed commit.

```bash
git revert <sha>            # creates an inverse commit; resolve any conflict in place
git commit -s               # if the revert paused for a manual resolution
```

`git revert` is safe on a shared trunk because it moves forward, never rewriting a commit a
peer may have already built on. If several commits are bad, revert the range
(`git revert <oldest>^..<newest>`). This is the first thing to try for almost every incident.

## 2. Downgrade to a stable version (git tags + the VERSION marker)

Stable versions are git tags of the form `vX.Y.Z`, cut only by the CI-gated release cadence
(it waits for a green build before it tags), so a tag is a version that passed every gate.
The current tags are `v0.30.0`, `v0.31.0`, `v0.32.0`. To run a previous tag:

```bash
git fetch --tags
git checkout v0.31.0        # check out the tag to run the last known-good build
go build ./cmd/fak          # rebuild from that tree
```

The single source of truth for "what version is this" is the repo-root `VERSION` file
(currently `0.32.0`); `fak version` prints it via `internal/appversion`. After a downgrade,
`VERSION` reflects the tag you checked out, so benchmark artifacts and `fak version` agree.
To return to the tip, `git checkout main`.

## 3. Pin a stable version (no auto-upgrade)

To **pin a stable version** for a fleet — hold it on a known-good build and not move with the
trunk — set the version explicitly. `internal/appversion.Current()` resolves in this order, so
any of these pins the answer:

1. `FAK_APP_VERSION=0.31.0` in the environment (highest precedence; the simplest **pin to v0.31.0**).
2. The `VERSION` file on the checked-out tree (pin by staying on a tag).
3. A release build's `-ldflags "-X …/internal/appversion.BuildVersion=0.31.0"`.

The environment pin wins over the `VERSION` file, so a fleet host can run a pinned version
without checking out a different tree. Clearing `FAK_APP_VERSION` lets the `VERSION` file win
again.

## 4. Revert measured state (baselines + the keep/revert ladder)

Two ratchets defend the trunk against a silent quality regression, and each has a defined way
back:

- **The RSI keep/revert ladder.** `internal/shipgate` keeps a change only on a strict, measured
  gain and otherwise renders `REVERT`; `cmd/rsiloop -mode track` re-measures the deterministic
  main KPI against `internal/rsiloop/testdata/main-kpi-baseline.jsonl` and exits non-zero on a
  regression. A red track gate means the KPI genuinely dropped — fix the change or revert it
  (layer 1); do not re-pin the RSI baseline to paper over a real regression.

- **The scorecard portfolio ratchet.** `tools/scorecard_control_pane.py --check` is GREEN while
  portfolio debt holds at-or-below `tools/scorecard_baseline.json` and RED only when it
  regresses. If a debt regression is the bad change, fix it (retire the debt with the owning
  scorecard's skill) and **re-pin** the floor so the ratchet locks the recovery in:

  ```bash
  python tools/scorecard_control_pane.py --pin   # re-pin scorecard_baseline.json after a debt drop
  ```

  Re-pin only **down**, after a real recovery — never up to hide a regression. The `main-kpi-baseline`
  and `scorecard_baseline` files are the committed floors a `git revert` of a bad re-pin restores.

- **The garden bundle + its brake.** `tools/garden_bundle.py --check` (`make garden-check`) folds
  the read-only gardening passes — the scorecard ratchet above plus the fresh-status rollup — into
  one verdict, and a scheduled host tick (`tools/register_control_pane_tick.sh --mode garden`) runs
  it on a daily cadence through the loop ledger. It is read-only (it never commits), so the rollback
  is just to **stop it**: export `FAK_GARDEN=off` on the host (the env-side brake), or set the loop
  paused in the live governor policy:

  ```bash
  # one-off: skip the bundle on this host
  FAK_GARDEN=off make garden
  # durable: pause the garden tick in the governor (live policy is .fak/loop-policy.json,
  # seeded from the tracked template tools/loop-policy.default.json)
  fak loop control --loop garden/default --pause     # or edit "paused": true in the policy
  ```

  The `min_interval_seconds` floor (12h, from the template) is the cadence brake that keeps a
  flapping scheduler from piling up ticks. Remove the tick entirely with
  `tools/register_control_pane_tick.sh remove --mode garden`.

## 5. Restore session / fleet state (`fak snapshot`)

For live state that the code-level layers above don't reach — a running session, a whole fleet
of drive states — fak ships a uniform, tamper-evident capture-and-restore seam
(`internal/snapshot` + the portable `internal/sessionimage` bundle), fronted by the `fak snapshot`
command. Every envelope carries a sha256 body digest, so a truncated or tampered snapshot
**fails closed** on load rather than restoring a corrupt state.

```bash
fak snapshot info  --file fleet.snap            # load + integrity-verify an envelope before trusting it
fak snapshot dump-fleet    --addr <gateway> --out fleet.snap   # capture a LIVE fleet's drive state
fak snapshot restore-fleet --addr <gateway> --file fleet.snap  # re-establish that fleet on another gateway
```

A fleet dumped this way restores **verbatim**, including terminal states (a stopped session
restores stopped, not revived). A session image survives a model change — it drops the KV cache
but preserves the logical session and its content-addressed pages — so you can restore a session
onto a different build. Run `fak snapshot kinds` to list the primitives this build can dump, and
`fak snapshot demo` for the offline, no-key witness that the dump → restore round-trip holds.

---

See also: [`AGENTS.md`](../AGENTS.md) for the trunk + commit rules, the
[stability scorecard](STABILITY-SCORECARD.md) for what "stable" is measured against, and
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full ship contract.
