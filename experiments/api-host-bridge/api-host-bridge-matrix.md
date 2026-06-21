# API-Host Bridge Matrix

- Required witnesses resolved: 9/9
- Bridge covered: yes
- Provider shapes covered: anthropic, gemini, openai-compatible, xai

| witness | required | status | command |
|---|:---:|---|---|
| `policy_manifest_floor` | yes | resolved | `go test ./internal/policy ./cmd/fak -run 'Test(ArgRulesAreLoadBearing|LoadedPolicyIsLoadBearing|ApplyRuntimeInstallsIFCManifestPolicy)$'` |
| `openai_compatible_gateway` | yes | resolved | `go test ./internal/gateway -run 'Test(ChatProxyFiltersAndRepairs|ChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend|HTTPModelsAndHealth)$'` |
| `native_provider_adapters` | yes | resolved | `go test ./internal/agent -run 'Test(PreSendQuarantineRedactsToolResultsAcrossAdapters|ProviderAdaptersMarshalNativeToolShapes|ProviderAdaptersOmitToolChoiceWithoutTools|ProviderAdaptersParseToolCalls|HTTPPlannerUsesProviderAdapterAndPreSendQuarantine)$'` |
| `provider_proxy_end_to_end` | yes | resolved | `go test ./internal/gateway -run 'TestChatProxyProviderAdaptersEndToEnd$'` |
| `host_agnostic_openai_compatible` | yes | resolved | `go test ./internal/gateway -run 'TestChatProxyOpenAICompatible(AliasIsHostAgnostic|ObjectArgumentsAreHostAgnostic|StreamModeStreamsAdjudicatedCalls)$'` |
| `openai_compatible_host_profiles` | yes | resolved | `go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleHostProfileConformance$'` |
| `direct_http_syscall` | yes | resolved | `go test ./internal/gateway -run 'Test(HTTPSyscallAllow|HTTPAdjudicateTransformRepairsArgs|HTTPQuarantineSurfaced|HTTPSyscallWitnessFailsClosed)$'` |
| `direct_mcp_syscall` | yes | resolved | `go test ./internal/gateway -run 'Test(MCPStdioRoundtrip|MCPOverHTTP)$'` |
| `dos_style_turnbench` | yes | resolved | `go test ./internal/turnbench -run 'TestRun_(AirlineClassesAreLiveKernelEvents|VDSOAblationIsARealPathSwap|HappyPathSavesNothing)$'` |
| `live_api_host_inventory` | no | resolved | `python tools/api_host_live_inventory.py --out fak/experiments/api-host-bridge/api-host-live-inventory.json --markdown fak/experiments/api-host-bridge/api-host-live-inventory.md` |
| `current_api_host_readiness` | no | resolved | `python tools/api_host_readiness_probe.py --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md` |
