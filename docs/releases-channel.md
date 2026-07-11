---
title: "The #releases channel"
description: "How fak uses the #releases channel as the scoreboard-workspace status feed for cut releases - the dual of the #blockers progress channel."
---

# The #releases channel

`#releases` is the release surface for cut releases — the release event ("something
shipped") that had no Slack feed before #1004. The feeder code has shipped, and — like
every sibling feeder — release reporting now **folds onto the CI/CD reporting sink
`C0BGQ411TCJ`** by default (see
[cicd-reporting-slack-sink.md](decisions/cicd-reporting-slack-sink.md)), so a provisioned
repo publishes every release report there with no operator setup. Set the
`FAK_RELEASES_CHANNEL` repo variable to split releases back out to a dedicated room (e.g.
the old `#releases`, `C0BGHS7HFV1`), or `FAK_CICD_REPORT_CHANNEL` to repoint the whole
reporting family at once.

## What already ships

`release-artifacts.yml` runs an `announce` job after the release's checksum assets land
(`needs: checksums`). It renders a release card (`kpi=release`, `value=<tag>`,
`detail=<release-notes URL>`) and always writes it to `GITHUB_STEP_SUMMARY` first, so the
run is auditable even when nothing posts:

```bash
go run ./cmd/fak scoreboard post \
  --channel "$FAK_RELEASES_CHANNEL" \
  --kpi release \
  --value "$TAG" \
  --grade A \
  --verdict OK \
  --detail "$NOTES_URL" \
  --source ci
```

It posts for real whenever `FAK_SCOREBOARD_TOKEN` (secret) is present; the channel is no
longer a gate because it defaults to the CI/CD reporting sink `C0BGQ411TCJ`. Without the token the step is skipped
and the summary says so, so a fork or a secret-less run never hard-fails. The contract is
pinned by `tools/release_artifacts_workflow_test.py`
(`test_announces_release_after_assets_land`, `test_release_announce_dry_runs_without_secret`).

`release-notify.yml` separately watches the release-cadence run itself and posts EVERY
release-observability card — `fired`, `failed-mid-chain`, `cadence-stalled`, and the
downstream `artifacts-failed` — to the CI/CD reporting sink via the same defaulted
`FAK_RELEASES_CHANNEL` (#1390). Severity rides the card's grade/verdict (A·OK for a fired
release, F/D/C·ACTION for a failure or stall), not a separate channel — so all release
reporting, success and failure alike, lands in one place.

## Operator step (token only)

The channel default (the CI/CD reporting sink `C0BGQ411TCJ`) is baked in, so the only
remaining requirement is the posting token:

1. Confirm `FAK_SCOREBOARD_TOKEN` (the scoreboard-workspace bot token) is set as a repo
   secret — every other feeder in this workspace depends on the same token — and that the
   bot has been invited to the reporting sink (channel `C0BGQ411TCJ`, team `T0BDEJF1HGB`, the same
   workspace `#scoreboard` / `#steering-guard` / `#blockers` live in).
2. *(optional)* To retarget a fork or a test run, set the `FAK_RELEASES_CHANNEL` repo
   variable under *Settings -> Secrets and variables -> Actions -> Variables*; it overrides
   the built-in default.

Until the token is set, every release keeps landing correctly; the announce step just
renders the card into the run summary instead of posting it.
