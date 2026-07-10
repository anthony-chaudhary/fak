---
title: "The #releases channel"
description: "How fak uses the #releases channel as the scoreboard-workspace status feed for cut releases - the dual of the #blockers progress channel."
---

# The #releases channel

`#releases` is the scoreboard-workspace status feed for cut releases. It is the dual of
`#steering-guard` / `#scoreboard` — the release event ("something shipped") that had no
Slack feed before #1004. The feeder code has shipped; the channel itself is the one
remaining operator step.

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

It posts for real only when both `FAK_SCOREBOARD_TOKEN` (secret) and `FAK_RELEASES_CHANNEL`
(repo variable) are present; otherwise the step is skipped and the summary names which one
is missing. A fork or a secret-less run never hard-fails. The contract is pinned by
`tools/release_artifacts_workflow_test.py` (`test_announces_release_after_assets_land`,
`test_release_announce_dry_runs_without_secret_or_channel`).

`release-notify.yml` separately watches the release-cadence run itself (fired /
failed-mid-chain / stalled) and also posts its `fired` ping to `#releases` via the same
`FAK_RELEASES_CHANNEL` variable — see that workflow's header comment for the autonomous-
release-observability distinction (#1390).

## Operator one-time step (the actual blocker)

The feeder cannot post until the channel exists:

1. Create `#releases` in the scoreboard workspace (team `T0BDEJF1HGB`, the same workspace
   `#scoreboard` / `#steering-guard` / `#blockers` live in) and invite the scoreboard bot.
2. Record its channel id as the `FAK_RELEASES_CHANNEL` repo variable under
   *Settings -> Secrets and variables -> Actions -> Variables*.
3. Confirm `FAK_SCOREBOARD_TOKEN` (the scoreboard-workspace bot token) is already set as a
   repo secret — every other feeder in this workspace depends on the same token.

Until both are set, every release keeps landing correctly; the announce step just renders
the card into the run summary instead of posting it.
