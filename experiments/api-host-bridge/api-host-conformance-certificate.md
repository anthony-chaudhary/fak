# API-Host Conformance Certificate

Any API host is covered when it satisfies one of the proven compatible host classes; hosts outside those wire contracts are explicit non-claims.

## Summary

- Capabilities proven: 10/10
- Failed capabilities: 0
- Missing required non-claims: 0
- Artifact errors: 0
- Certificate gate: yes

## Capabilities

| capability | status | claim |
|---|---|---|
| `openai_compatible_host_conformance` | PROVEN | Any host presenting the OpenAI-compatible chat-completions wire can sit behind FAK under compatible aliases, arbitrary base paths, opaque model ids, optional auth, ignored vendor extension fields, and stream=true client responses synthesized after adjudication. |
| `openai_compatible_host_profile_corpus` | PROVEN | The OpenAI-compatible bridge is checked against host-profile drift: null arguments, legacy function_call, typed content parts, extra fields, omitted tool_choice without advertised tools, rogue proposed tool calls, multichoice responses, and content-only responses. |
| `openai_client_proxy_boundary` | PROVEN | OpenAI-compatible clients see only FAK-admitted proposed tool calls after deny filtering and transform repair. |
| `native_provider_transcript_wires` | PROVEN | Covered native provider transcript wires preserve pre-send quarantine and parse provider-specific tool-call shapes. |
| `direct_http_syscall_boundary` | PROVEN | Any-language clients can bypass provider quirks and call the FAK/DOS kernel boundary over native HTTP. |
| `direct_mcp_syscall_boundary` | PROVEN | Any-language clients can call the same FAK/DOS kernel boundary over MCP stdio or MCP-over-HTTP. |
| `expanded_candidate_host_roster` | PROVEN | A broad no-spend roster of API-host target templates is syntactically valid, maps to supported bridge wire classes, and carries exact readiness/acceptance rerun commands. |
| `candidate_host_acceptance` | PROVEN | Candidate API hosts are classified into ready, live-confirmed, typed external blocker, supported-unprobed, or unsupported-wire states without unclassified drift. |
| `live_scoped_host_evidence` | PROVEN | Committed live artifacts prove at least one real frontier OpenAI-compatible path and local OpenAI-compatible shims, while external auth/billing failures are typed. |
| `external_state_residual_audit` | PROVEN | Credential, billing, readiness, and retry state for roster targets is machine-audited without treating external state as solved. |

## Qualification Rules

| rule | status | detail |
|---|---|---|
| `openai_compatible_wire` | SUPPORTED | Host exposes an OpenAI-compatible chat-completions wire; model id, base path, auth, and vendor extension fields may vary within that wire contract. Downstream stream=true is supported by emitting synthesized chunks after full adjudication, not by passing through raw upstream deltas. |
| `covered_native_provider_wire` | SUPPORTED | Host uses one of the covered native provider transcript wires: anthropic, gemini, openai-compatible, or xai. |
| `direct_kernel_wire` | SUPPORTED | Client calls the FAK/DOS boundary directly over HTTP or MCP. |
| `unknown_wire` | OUT_OF_CONTRACT | A host with no compatible wire is not covered until a transcript adapter or direct syscall integration is added. |
| `paid_or_keyed_remote_state` | EXTERNAL_STATE | Billing, credentials, rate limits, and edge-access restrictions must be resolved before live remote smoke runs can prove that host instance. |
