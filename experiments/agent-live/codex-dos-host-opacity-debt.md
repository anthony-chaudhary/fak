# Codex DOS Host-Opacity Debt

## Summary

- audit_status: `WARN`
- actionability_status: `WARN`
- residual: `none`
- dos_kernel_version: `0.28.0`
- dos_kernel_using_latest: `None`
- codex_hook_fast_path: `PASS` `{"native_launcher": 4}`
- git_gate_status: `PASS` proved_at=`2026-06-25T10:15:36.616692Z`
- sessions_audited: `10`

## Evidence

- recent_window_unknown_tree_rate: `0.367964`
- recent_window_delegates: `751`
- post_repair_observations: `3832`
- post_repair_delegates: `0`
- post_repair_unknown_tree_warnings: `1990`
- post_repair_shell_shapes: `{"non_shell_tool": 585, "shell_in_tree_or_safe_write_target": 4, "shell_no_write_target_detected": 2370}`
- post_repair_shell_families: `{"build_test": 126, "git_read": 426, "git_write": 14, "inline_script": 33, "other_shell": 452, "powershell_inspect": 55, "powershell_read": 548, "python_script": 231, "python_test": 260, "search_rg": 220, "shell_redirect": 9}`
- post_repair_mutating_shell_families: `{"git_write": 14}`
- post_repair_mutating_sessions: `[{"codex_session_file": "rollout-2026-06-25T01-27-31-019efde4-32f6-7b22-9e5b-1ccddb6c3987.jsonl", "mutating_shell_family_counts": {"git_write": 6}, "shell_family_counts": {"git_read": 80, "git_write": 6, "inline_script": 8, "other_shell": 40, "powershell_inspect": 16, "powershell_read": 63, "python_script": 38, "python_test": 10, "search_rg": 21}, "shell_shape_counts": {"non_shell_tool": 30, "shell_no_write_target_detected": 282}, "thread_id": "019efde4-32f6-7b22-9e5b-1ccddb6c3987", "tool_call_rows": 312, "write_op_counts": {}}, {"codex_session_file": "rollout-2026-06-25T01-48-58-019efdf7-d4a8-7c30-b2e4-438c92fafec2.jsonl", "mutating_shell_family_counts": {"git_write": 1}, "shell_family_counts": {"build_test": 8, "git_read": 14, "git_write": 1, "other_shell": 41, "powershell_inspect": 3, "powershell_read": 28, "python_script": 4, "search_rg": 16}, "shell_shape_counts": {"non_shell_tool": 3, "shell_no_write_target_detected": 115}, "thread_id": "019efdf7-d4a8-7c30-b2e4-438c92fafec2", "tool_call_rows": 118, "write_op_counts": {}}, {"codex_session_file": "rollout-2026-06-25T02-38-18-019efe24-ff94-73a0-8b81-0958a40f72e6.jsonl", "mutating_shell_family_counts": {"git_write": 5}, "shell_family_counts": {"build_test": 14, "git_read": 41, "git_write": 5, "other_shell": 15, "powershell_inspect": 7, "powershell_read": 59, "search_rg": 30}, "shell_shape_counts": {"non_shell_tool": 69, "shell_no_write_target_detected": 171}, "thread_id": "019efe24-ff94-73a0-8b81-0958a40f72e6", "tool_call_rows": 240, "write_op_counts": {}}, {"codex_session_file": "rollout-2026-06-25T01-33-50-019efde9-fab7-7c13-bf3a-a994010253a3.jsonl", "mutating_shell_family_counts": {"git_write": 2}, "shell_family_counts": {"build_test": 19, "git_read": 113, "git_write": 2, "other_shell": 53, "powershell_inspect": 6, "powershell_read": 32, "python_script": 11, "search_rg": 20}, "shell_shape_counts": {"non_shell_tool": 50, "shell_no_write_target_detected": 256}, "thread_id": "019efde9-fab7-7c13-bf3a-a994010253a3", "tool_call_rows": 306, "write_op_counts": {}}]`
- post_git_gate_shell_families: `{"build_test": 1, "git_read": 1, "git_write": 1, "other_shell": 1, "powershell_read": 4, "python_script": 1, "search_rg": 1}`
- post_git_gate_mutating_shell_families: `{"git_write": 1}`
- post_repair_write_ops: `{"redirect": 4}`

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
