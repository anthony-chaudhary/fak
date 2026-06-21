# API-Host Bridge Gate

- Required witness commands passed: 9/9
- Executable bridge covered: yes

| witness | required | status | elapsed ms | command |
|---|:---:|---|---:|---|
| `policy_manifest_floor` | yes | passed | 1462 | `go test ./internal/policy ./cmd/fak -run 'Test(ArgRulesAreLoadBearing|LoadedPolicyIsLoadBearing|ApplyRuntimeInstallsIFCManifestPolicy)$'` |
| `openai_compatible_gateway` | yes | passed | 1294 | `go test ./internal/gateway -run 'Test(ChatProxyFiltersAndRepairs|ChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend|HTTPModelsAndHealth)$'` |
| `native_provider_adapters` | yes | passed | 1343 | `go test ./internal/agent -run 'Test(PreSendQuarantineRedactsToolResultsAcrossAdapters|ProviderAdaptersMarshalNativeToolShapes|ProviderAdaptersOmitToolChoiceWithoutTools|ProviderAdaptersParseToolCalls|HTTPPlannerUsesProviderAdapterAndPreSendQuarantine)$'` |
| `provider_proxy_end_to_end` | yes | passed | 1344 | `go test ./internal/gateway -run 'TestChatProxyProviderAdaptersEndToEnd$'` |
| `host_agnostic_openai_compatible` | yes | passed | 1327 | `go test ./internal/gateway -run 'TestChatProxyOpenAICompatible(AliasIsHostAgnostic|ObjectArgumentsAreHostAgnostic|StreamModeStreamsAdjudicatedCalls)$'` |
| `openai_compatible_host_profiles` | yes | passed | 1304 | `go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleHostProfileConformance$'` |
| `direct_http_syscall` | yes | passed | 1251 | `go test ./internal/gateway -run 'Test(HTTPSyscallAllow|HTTPAdjudicateTransformRepairsArgs|HTTPQuarantineSurfaced|HTTPSyscallWitnessFailsClosed)$'` |
| `direct_mcp_syscall` | yes | passed | 1348 | `go test ./internal/gateway -run 'Test(MCPStdioRoundtrip|MCPOverHTTP)$'` |
| `dos_style_turnbench` | yes | passed | 1562 | `go test ./internal/turnbench -run 'TestRun_(AirlineClassesAreLiveKernelEvents|VDSOAblationIsARealPathSwap|HappyPathSavesNothing)$'` |
| `live_api_host_inventory` | no | skipped |  | `python tools/api_host_live_inventory.py --out fak/experiments/api-host-bridge/api-host-live-inventory.json --markdown fak/experiments/api-host-bridge/api-host-live-inventory.md` |
| `current_api_host_readiness` | no | skipped |  | `python tools/api_host_readiness_probe.py --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md` |
