# Codex DOS Host-Opacity Debt

## Summary

- audit_status: `WARN`
- actionability_status: `PASS`
- residual: `HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE, HOST_SHELL_OPACITY, UNKNOWN_TREE_WARNINGS`
- dos_kernel_version: `0.28.0`
- dos_kernel_using_latest: `None`
- codex_hook_fast_path: `PASS` `{"native_launcher": 4}`
- git_gate_status: `PASS` proved_at=`2026-06-25T14:51:04.727080Z`
- sessions_audited: `10`

## Evidence

- recent_window_unknown_tree_rate: `0.311072`
- recent_window_delegates: `664`
- post_repair_observations: `5406`
- post_repair_delegates: `0`
- post_repair_unknown_tree_warnings: `1999`
- post_repair_shell_shapes: `{"non_shell_tool": 537, "shell_in_tree_or_safe_write_target": 6, "shell_no_write_target_detected": 3042}`
- post_repair_shell_families: `{"build_test": 135, "git_read": 675, "git_write": 62, "inline_script": 37, "other_shell": 471, "powershell_inspect": 79, "powershell_read": 681, "python_script": 356, "python_test": 266, "search_rg": 270, "shell_redirect": 16}`
- post_repair_mutating_shell_families: `{"git_write": 62}`
- post_repair_mutating_sessions: `[{"codex_session_file": "rollout-2026-06-25T07-25-17-019eff2b-c03e-78a1-8716-cb82d70e28ac.jsonl", "mutating_shell_family_counts": {"git_write": 2}, "shell_family_counts": {"build_test": 7, "git_read": 71, "git_write": 2, "other_shell": 29, "powershell_inspect": 5, "powershell_read": 37, "python_script": 37, "python_test": 13, "search_rg": 20, "shell_redirect": 2}, "shell_shape_counts": {"non_shell_tool": 9, "shell_in_tree_or_safe_write_target": 1, "shell_no_write_target_detected": 222}, "thread_id": "019eff2b-c03e-78a1-8716-cb82d70e28ac", "tool_call_rows": 232, "write_op_counts": {"output-flag": 2}}, {"codex_session_file": "rollout-2026-06-25T02-38-18-019efe24-ff94-73a0-8b81-0958a40f72e6.jsonl", "mutating_shell_family_counts": {"git_write": 13}, "shell_family_counts": {"build_test": 30, "git_read": 82, "git_write": 13, "other_shell": 27, "powershell_inspect": 7, "powershell_read": 92, "search_rg": 47}, "shell_shape_counts": {"non_shell_tool": 103, "shell_no_write_target_detected": 298}, "thread_id": "019efe24-ff94-73a0-8b81-0958a40f72e6", "tool_call_rows": 401, "write_op_counts": {}}, {"codex_session_file": "rollout-2026-06-25T01-48-58-019efdf7-d4a8-7c30-b2e4-438c92fafec2.jsonl", "mutating_shell_family_counts": {"git_write": 41}, "shell_family_counts": {"build_test": 10, "git_read": 281, "git_write": 41, "other_shell": 155, "powershell_inspect": 8, "powershell_read": 85, "python_script": 59, "python_test": 9, "search_rg": 56, "shell_redirect": 1}, "shell_shape_counts": {"non_shell_tool": 22, "shell_no_write_target_detected": 705}, "thread_id": "019efdf7-d4a8-7c30-b2e4-438c92fafec2", "tool_call_rows": 727, "write_op_counts": {}}, {"codex_session_file": "rollout-2026-06-25T01-27-31-019efde4-32f6-7b22-9e5b-1ccddb6c3987.jsonl", "mutating_shell_family_counts": {"git_write": 6}, "shell_family_counts": {"git_read": 82, "git_write": 6, "inline_script": 8, "other_shell": 43, "powershell_inspect": 16, "powershell_read": 64, "python_script": 46, "python_test": 10, "search_rg": 21}, "shell_shape_counts": {"non_shell_tool": 32, "shell_no_write_target_detected": 296}, "thread_id": "019efde4-32f6-7b22-9e5b-1ccddb6c3987", "tool_call_rows": 328, "write_op_counts": {}}]`
- post_git_gate_shell_families: `{"powershell_read": 1, "python_script": 1}`
- post_git_gate_mutating_shell_families: `{}`
- post_repair_write_ops: `{"output-flag": 3, "redirect": 4}`

## Interpretation

The DOS native fast path is repaired and post-repair delegate count is zero. The current actionable WARN is opaque mutating shell usage: Codex ran shell families such as `git_write` without a structured tool boundary, so fak/DOS could not apply the named operation gate before the mutation-shaped call.

## Requested Upstream Improvement

- Route mutating Git operations through structured fak-gated tools such as `git_add`, `git_commit`, and `git_push`; do not run them as opaque shell commands.
- Include path/footprint metadata in Codex tool-call hook payloads, or expose host-visible read/search/list tools with path arguments.
- Preserve the current privacy boundary: the audit needs tool names, path metadata, timestamps, and counts, not prompts, command bodies, tool output, diffs, or model text.
- Keep shell command bodies out of durable reports; classify locally and emit only categories such as `shell_no_write_target_detected` or `shell_out_of_tree_write_target`.

## Privacy Boundary

- copied: session filenames, thread ids, timestamps, tool names, counts, latency summaries, shell shape categories, shell family categories, write operation kinds
- dropped: prompts, command bodies, tool arguments, tool results, diffs, model text, hook command bodies
