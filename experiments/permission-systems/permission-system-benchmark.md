# Permission-System Benchmark

| system | deterministic risk coverage | soft/review | unguarded risk allows | known max FNR | result admission | API-host bridge | bridge controls | bridge result quarantine |
|---|---:|---:|---:|---:|---|---|---:|---|
| FAK/DOS gateway | 6/6 (100.0%) | 0 | 0 |  | QUARANTINE | yes | 6/6 (100.0%) | QUARANTINE |
| Claude Code auto mode | 0/6 (0.0%) | 6 | 0 | 17.0% | WARNING | no | 0/6 (0.0%) | WARNING |
| Codex workspace sandbox | 3/6 (50.0%) | 3 | 0 |  | REVIEW_AFTER | no | 0/6 (0.0%) | REVIEW_AFTER |
| GitHub Copilot cloud agent | 2/6 (33.3%) | 4 | 0 |  | REVIEW_AFTER | no | 0/6 (0.0%) | REVIEW_AFTER |
| Manual permission prompts | 0/6 (0.0%) | 6 | 0 |  | PROMPT | no | 0/6 (0.0%) | PROMPT |
| Bypass / dangerous skip | 0/6 (0.0%) | 0 | 6 |  | UNBOUNDED_ALLOW | no | 0/6 (0.0%) | UNBOUNDED_ALLOW |

## API-Host Bridge Dimensions

- `host_agnostic_openai_compatible_proxy`: A compatible API host can sit behind the gateway with arbitrary base paths, opaque model IDs, optional auth, ignored vendor extension fields, object-or-string tool arguments, and stream=true client responses synthesized after adjudication.
- `synthetic_host_profile_conformance`: A synthetic compatible-host profile corpus covers null arguments, legacy function_call, typed content parts, extra fields, omitted tool_choice without advertised tools, rogue proposed tool calls, multichoice responses, and content-only replies.
- `pre_execution_tool_call_admission`: Proposed tool calls are deterministically filtered or repaired before a client sees them.
- `pre_send_tool_result_quarantine`: Hostile tool-result bytes sent by a client are quarantined before the upstream model sees them.
- `roster_driven_host_qualification`: Candidate hosts are rostered, probed, and qualified into live, ready, credential-needed, external-blocked, or out-of-contract states.
- `dos_style_executable_bridge_proof`: The bridge claim is backed by source witnesses and commands run fresh by the proof gate.

## API-Host Bridge Witnesses

- tools/api_host_bridge_proof.py rolls every bridge artifact into one requirement-by-requirement proof gate
- tools/api_host_compat_contract.py defines the scoped compatible-host contract and non-claims
- tools/api_host_acceptance_probe.py classifies arbitrary candidate hosts into ready, typed blocker, or unsupported-wire states
- tools/api_host_bridge_matrix.py resolves source witnesses
- tools/api_host_bridge_gate.py runs required witnesses with -count=1
- fak/internal/gateway TestChatProxyProviderAdaptersEndToEnd proves a client-facing bridge across covered upstream provider wires
- fak/internal/gateway TestChatProxyOpenAICompatibleObjectArgumentsAreHostAgnostic proves object-or-string function arguments survive host quirks
- fak/internal/gateway TestChatProxyOpenAICompatibleHostProfileConformance proves compatible-host profile drift preserves the FAK tool boundary
- fak/internal/gateway TestChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend proves client tool-result bytes are quarantined before upstream send
- tools/api_host_live_inventory.py classifies live API-host success and auth/billing blockers
- tools/api_host_readiness_probe.py refreshes current /models readiness without spending chat tokens
- tools/api_host_qualification.py applies the conformance certificate to every roster target
- tools/permission_source_audit.py verifies external permission-system source claims before the benchmark is trusted

## Sources

- Anthropic: How we built Claude Code auto mode: https://www.anthropic.com/engineering/claude-code-auto-mode
- Claude Code docs: permission modes: https://code.claude.com/docs/en/permission-modes
- Claude Code docs: permissions: https://code.claude.com/docs/en/permissions
- OpenAI Codex docs: sandboxing: https://developers.openai.com/codex/concepts/sandboxing
- OpenAI Codex docs: permission profiles: https://developers.openai.com/codex/permissions
- GitHub docs: Copilot coding agent firewall: https://docs.github.com/en/enterprise-cloud@latest/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-firewall
- GitHub docs: Copilot Agents responsible use: https://docs.github.com/en/copilot/responsible-use/agents
- Local FAK bridge witnesses: tools/api_host_bridge_proof.py; tools/api_host_compat_contract.py; tools/api_host_acceptance_probe.py; tools/api_host_bridge_matrix.py; tools/api_host_bridge_gate.py; tools/api_host_live_inventory.py; tools/api_host_readiness_probe.py; tools/permission_source_audit.py
