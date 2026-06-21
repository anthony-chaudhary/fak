# API-Host Compatibility Contract

A compatible API host is one that can present an OpenAI-compatible upstream, a covered native provider transcript wire, or direct HTTP/MCP access to the FAK/DOS syscall boundary.

## Summary

- Host classes proven: 5/5
- Contract gate: yes

## Host Classes

| host class | status | claim |
|---|---|---|
| `openai_compatible_upstream` | PROVEN | An OpenAI-compatible upstream can be fronted while FAK/DOS owns tool-call filtering, repair, and downstream stream=true chunk synthesis after adjudication. |
| `native_provider_transcript_adapters` | PROVEN | Native provider adapters preserve pre-send quarantine and provider-specific tool shapes. |
| `direct_kernel_http_syscall` | PROVEN | Any-language clients can bypass provider quirks and call the kernel over native HTTP. |
| `direct_kernel_mcp_syscall` | PROVEN | Any-language clients can call the same kernel over MCP stdio or MCP-over-HTTP. |
| `live_scoped_host_evidence` | PROVEN | Committed live artifacts prove the bridge on at least one real frontier OpenAI-compatible host plus local OpenAI-compatible shims. |

## Non-Claims

| item | status | reason |
|---|---|---|
| `arbitrary_api_host_without_compatible_wire` | OUT_OF_CONTRACT | A host must expose an OpenAI-compatible wire, one of the covered native provider wires, or use direct HTTP/MCP syscall integration. |
| `streaming_chat_completions_delta_passthrough` | OUT_OF_CONTRACT | Downstream stream=true is supported by synthesized chunks after full adjudication; raw upstream streaming delta passthrough remains out of contract. |
| `paid_or_keyed_live_execution_without_credentials` | EXTERNAL_STATE | Billing, API-key, or edge-access failures are typed readiness states, not bridge failures. |
| `provider_semantics_beyond_tool_wire` | OUT_OF_CONTRACT | The bridge owns tool-call admission and result quarantine; it does not claim vendor model quality or universal task success. |
