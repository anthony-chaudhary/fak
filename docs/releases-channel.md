---
title: "The #releases channel"
description: "How fak uses the #releases channel as the scoreboard-workspace status feed for cut releases - the dual of the #blockers progress channel."
---

# The #releases channel

`#releases` is the scoreboard-workspace status feed for cut releases. It is the dual of
`#steering-guard` / `#scoreboard` — the release event ("something shipped") that had no
Slack feed before #1004. The feeder code has shipped, and — like every sibling feeder
(#blockers, #bench, #dojo, …) — it now carries a **built-in default channel id,
`C0BGHS7HFV1`**, so a provisioned repo publishes every release report there with no
operator setup. The `FAK_RELEASES_CHANNEL` repo variable is now only an override.

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
longer a gate because it defaults to `C0BGHS7HFV1`. Without the token the step is skipped
and the summary says so, so a fork or a secret-less run never hard-fails. The contract is
pinned by `tools/release_artifacts_workflow_test.py`
(`test_announces_release_after_assets_land`, `test_release_announce_dry_runs_without_secret`).

`release-notify.yml` separately watches the release-cadence run itself (fired /
failed-mid-chain / stalled) and also posts its `fired` ping to `#releases` via the same
defaulted `FAK_RELEASES_CHANNEL` — see that workflow's header comment for the autonomous-
release-observability distinction (#1390). Its failure/stall alerts are, by design, blocker
signals (a dangling cut, a rotting `@latest`) and route to the louder `#blockers` channel,
not to `#releases`.

## Operator step (token only)

The channel default (`C0BGHS7HFV1`) is baked in, so the only remaining requirement is the
posting token:

1. Confirm `FAK_SCOREBOARD_TOKEN` (the scoreboard-workspace bot token) is set as a repo
   secret — every other feeder in this workspace depends on the same token — and that the
   bot has been invited to `#releases` (channel `C0BGHS7HFV1`, team `T0BDEJF1HGB`, the same
   workspace `#scoreboard` / `#steering-guard` / `#blockers` live in).
2. *(optional)* To retarget a fork or a test run, set the `FAK_RELEASES_CHANNEL` repo
   variable under *Settings -> Secrets and variables -> Actions -> Variables*; it overrides
   the built-in default.

Until the token is set, every release keeps landing correctly; the announce step just
renders the card into the run summary instead of posting it.
