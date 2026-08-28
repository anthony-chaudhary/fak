#!/usr/bin/env bash
set -euo pipefail

budget="${FAK_TRAJECTORY_BUDGET:-configs/trajectory-attribution-nightly.v1.json}"
corpus="${FAK_TRAJECTORY_CORPUS:-local}"
claude_root="${FAK_TRAJECTORY_CLAUDE_ROOT:?FAK_TRAJECTORY_CLAUDE_ROOT is required}"
codex_root="${FAK_TRAJECTORY_CODEX_ROOT:?FAK_TRAJECTORY_CODEX_ROOT is required}"
receipt="${FAK_TRAJECTORY_RECEIPT:-trajectory-attribution-receipt.json}"

args=(
  trajectory nightly
  --budget "$budget"
  --corpus "$corpus"
  --claude-root "$claude_root"
  --codex-root "$codex_root"
  --receipt "$receipt"
)
if [[ -n "${FAK_TRAJECTORY_HISTORY:-}" ]]; then
  args+=(--history "$FAK_TRAJECTORY_HISTORY")
fi
if [[ -n "${FAK_TRAJECTORY_AT:-}" ]]; then
  args+=(--at "$FAK_TRAJECTORY_AT")
fi

go run ./cmd/fak "${args[@]}"
