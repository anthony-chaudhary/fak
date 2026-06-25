# Claude Code historical guard replay

- generated: `2026-06-25T14:58:24.018378Z`
- status: **`PASS`**
- sessions_discovered: `10`
- sessions_audited: `6`
- tool_calls_seen: `39`
- unique_tool_calls_replayed: `38`
- truncated: `False`

## Verdict Counts

- ALLOW: `35`
- DENY: `3`

## Reason Counts

- DEFAULT_DENY: `2`
- POLICY_BLOCK: `1`

## Non-Allow Samples

- `c38bdf84780b2f80` TaskUpdate -> `DENY` / `DEFAULT_DENY`
- `228e30224e260877` Workflow -> `DENY` / `DEFAULT_DENY`
- `f5a056b4348603d3` Bash -> `DENY` / `POLICY_BLOCK`

## Privacy

This replay records only tool names, verdict metadata, aggregate counts, and hash digests. It never writes prompts, tool arguments, tool results, or raw transcript text.
