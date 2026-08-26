---
title: "Guarded Codex default profiles in a non-FAK repository — 2026-08-18"
description: "Verdict: guarded Codex now receives Caveman medium and Ponytail medium by default through Codex's -c developerinstructions=... configuration seam."
---
# Guarded Codex default profiles in a non-FAK repository — 2026-08-18

**Verdict:** guarded Codex now receives Caveman medium and Ponytail medium by default through Codex's -c developer_instructions=... configuration seam.

- The witness workspace was a temporary non-FAK Git repository containing enchmark.txt.
- TestInjectGuardProfilesDefaultsIntoCodexDeveloperInstructions captures the child argv and proves both governed fragments are TOML-quoted into developer_instructions before the original Codex arguments.
- TestLaunchPostureGuardCodexNamesActiveProfilesAndInertWire proves ak doctor launch-posture reports both profiles active rather than unsupported.
- The local executable was $((& codex --version) -join ' ').
- Existing independent opt-outs remain: --output-profile full and --work-profile standard; unknown harnesses remain byte-identical by default and fail on explicit unsupported profile requests.

Structured artifact: [guard-codex-default-profiles-non-fak-2026-08-18.json](guard-codex-default-profiles-non-fak-2026-08-18.json).

The doctor still exits non-zero for guarded Codex/OpenAI because Anthropic request-body compaction/elision/deferral/anchoring are inert and provider calibration may be missing. This witness claims only the two profile adapters.
