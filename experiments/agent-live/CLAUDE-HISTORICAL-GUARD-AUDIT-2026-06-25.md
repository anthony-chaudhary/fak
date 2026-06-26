# Claude Code historical guard replay

- generated: `2026-06-26T03:44:50.815290Z`
- status: **`PASS`**
- sessions_discovered: `20`
- sessions_audited: `16`
- tool_calls_seen: `231`
- unique_tool_calls_replayed: `228`
- truncated: `False`

## Verdict Counts

- ALLOW: `177`
- DENY: `51`

## Reason Counts

- DEFAULT_DENY: `42`
- POLICY_BLOCK: `9`

## Non-Allow Samples

- `c38bdf84780b2f80` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `797271dc71a4e4cc` TaskCreate -> `DENY` / `DEFAULT_DENY`
- `cc976c3fc44043ef` SendMessage -> `DENY` / `DEFAULT_DENY`
- `67f4ee5881b85c94` TaskCreate -> `DENY` / `DEFAULT_DENY`
- `45853eecae65bf8a` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `1ff7ab011e8dba53` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `1d5df4dd79b33f9a` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `79a8e7ad26a04cf3` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `b72886ffb73ac7aa` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `c5cee59184a30614` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `212ed19fd0804ba5` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `1efc96a44b137e31` TaskOutput -> `DENY` / `DEFAULT_DENY`

## Transcript Friction Signals

- summarized_sessions: `20`
- evidence_tag_counts: `{"DENY_OR_BLOCKED_FEEDBACK": 20, "HOOK_OR_API_WALL_FEEDBACK": 20, "HOST_PERMISSION_INTERRUPT": 20, "LARGE_RESULT": 7, "SHELL_HEAVY_SESSION": 11, "TOOL_ERROR_RECOVERY": 20}`
- remediation_session_counts: `{"align_policy_with_real_tool_shapes": 20, "cap_or_summarize_large_outputs": 7, "clear_hook_or_api_wall_feedback": 20, "fix_tool_contract_or_error_recovery_loop": 20, "reduce_permission_interruptions_or_scope_policy": 20, "replace_shell_with_path_visible_tools": 11}`
- marker_line_counts: `{"deny_or_blocked": 448, "error_recovery": 1463, "hook_or_api_wall": 509, "permission": 440}`
- result_shape_counts: `{"error_text": 220, "list": 17, "other_text": 1356, "permission_text": 6, "result_record_dict": 1548, "result_record_str": 51}`
- max_result_chars: `55614`

## Top Friction Sessions

- `a585f940822d0d52` root=`.claude/C--work-fak` tool_calls=`140` marker_lines=`1659` max_result_chars=`26764` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, replace_shell_with_path_visible_tools, cap_or_summarize_large_outputs`
- `5bc67363724625d9` root=`.claude/C--work-fak` tool_calls=`24` marker_lines=`222` max_result_chars=`20399` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, replace_shell_with_path_visible_tools, cap_or_summarize_large_outputs`
- `3ce3bdae26a0fe0e` root=`.claude-4e449ab2/C--work-fak` tool_calls=`2` marker_lines=`77` max_result_chars=`21302` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, replace_shell_with_path_visible_tools, cap_or_summarize_large_outputs`
- `9b2bbc8a51e22227` root=`.claude/C--work-fak` tool_calls=`16` marker_lines=`138` max_result_chars=`55614` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, LARGE_RESULT` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, cap_or_summarize_large_outputs`
- `4b2e843b571dedd8` root=`.claude/C--work-fak` tool_calls=`22` marker_lines=`118` max_result_chars=`55614` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, LARGE_RESULT` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, cap_or_summarize_large_outputs`
- `2d68492c6fb39131` root=`.claude/C--work-fak` tool_calls=`2` marker_lines=`51` max_result_chars=`3271` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, replace_shell_with_path_visible_tools`
- `f08be8fbd74c6b9d` root=`.claude/C--work-fak` tool_calls=`2` marker_lines=`42` max_result_chars=`32162` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, LARGE_RESULT` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, cap_or_summarize_large_outputs`
- `0d8992bbaeb632fa` root=`.claude-7cc12304/C--work-fak` tool_calls=`3` marker_lines=`37` max_result_chars=`7096` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION` remediation=`clear_hook_or_api_wall_feedback, reduce_permission_interruptions_or_scope_policy, align_policy_with_real_tool_shapes, fix_tool_contract_or_error_recovery_loop, replace_shell_with_path_visible_tools`

## Privacy

This replay records only tool names, verdict metadata, aggregate counts, and hash digests. It never writes prompts, tool arguments, tool results, or raw transcript text.
