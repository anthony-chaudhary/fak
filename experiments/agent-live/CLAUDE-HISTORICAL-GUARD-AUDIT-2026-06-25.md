# Claude Code historical guard replay

- generated: `2026-06-26T02:24:20.264335Z`
- status: **`PASS`**
- sessions_discovered: `20`
- sessions_audited: `13`
- tool_calls_seen: `230`
- unique_tool_calls_replayed: `227`
- truncated: `False`

## Verdict Counts

- ALLOW: `181`
- DENY: `46`

## Reason Counts

- DEFAULT_DENY: `39`
- POLICY_BLOCK: `7`

## Non-Allow Samples

- `79a8e7ad26a04cf3` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `b72886ffb73ac7aa` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `c5cee59184a30614` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `212ed19fd0804ba5` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `1efc96a44b137e31` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `674924f5562ca309` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `1e51e1031b86b69a` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `82713e2d6f1c3e62` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `6d0d16ddbc53699d` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `8aac1bf9e47e908d` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `60f58c4809dfdbb2` TaskOutput -> `DENY` / `DEFAULT_DENY`
- `b89b00f4a0fac60a` TaskOutput -> `DENY` / `DEFAULT_DENY`

## Transcript Friction Signals

- summarized_sessions: `20`
- evidence_tag_counts: `{"DENY_OR_BLOCKED_FEEDBACK": 17, "HOOK_OR_API_WALL_FEEDBACK": 20, "HOST_PERMISSION_INTERRUPT": 20, "LARGE_RESULT": 9, "SHELL_HEAVY_SESSION": 11, "TOOL_ERROR_RECOVERY": 19}`
- marker_line_counts: `{"deny_or_blocked": 429, "error_recovery": 1527, "hook_or_api_wall": 508, "permission": 421}`
- result_shape_counts: `{"error_text": 230, "list": 25, "other_text": 1463, "permission_text": 13, "tool_use_result_dict": 1665, "tool_use_result_str": 66}`
- max_result_chars: `64944`

## Top Friction Sessions

- `a585f940822d0d52` root=`.claude/C--work-fak` tool_calls=`129` marker_lines=`1485` max_result_chars=`26764` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT`
- `5bc67363724625d9` root=`.claude/C--work-fak` tool_calls=`24` marker_lines=`222` max_result_chars=`20399` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT`
- `f5770132f058da8b` root=`.claude-4e449ab2/C--work-fak` tool_calls=`0` marker_lines=`59` max_result_chars=`38156` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT`
- `4c03751e6c6c01b0` root=`.claude-7cc12304/C--work-fak` tool_calls=`1` marker_lines=`39` max_result_chars=`58302` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION, LARGE_RESULT`
- `7cb3d3b0088c9072` root=`.claude/C--work-fak` tool_calls=`30` marker_lines=`401` max_result_chars=`18936` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, SHELL_HEAVY_SESSION`
- `9b2bbc8a51e22227` root=`.claude/C--work-fak` tool_calls=`12` marker_lines=`110` max_result_chars=`55614` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, LARGE_RESULT`
- `4b2e843b571dedd8` root=`.claude/C--work-fak` tool_calls=`7` marker_lines=`65` max_result_chars=`55614` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, LARGE_RESULT`
- `f08be8fbd74c6b9d` root=`.claude/C--work-fak` tool_calls=`2` marker_lines=`42` max_result_chars=`32162` tags=`HOOK_OR_API_WALL_FEEDBACK, HOST_PERMISSION_INTERRUPT, DENY_OR_BLOCKED_FEEDBACK, TOOL_ERROR_RECOVERY, LARGE_RESULT`

## Privacy

This replay records only tool names, verdict metadata, aggregate counts, and hash digests. It never writes prompts, tool arguments, tool results, or raw transcript text.
