---
loop: goal
witness: "go test -v ./internal/agent/... ./internal/harnessprofile/... ./cmd/fak -run 'TestNativeHarness|TestHarnessProfile|TestPosture|TestMCP'"
budget: { max_iters: 20 }
lane: agent
---
# Objective
Enable the native agent harness (`fak agent`, `fak agent --native`, `fak chat`, and `internal/agent`) to use `fak guard` and `fak mcp` features by default, while strictly ensuring the default posture is PERMISSIVE ALLOW (`PostureDefaultOpen`): unlisted benign tools and dynamic MCP tools are allowed by default, while critical fail-closed security rails (dangerous gotchas blocklist, self-modification, explicit denies, and dangerous argument rules) remain fully enforced.

# Non-Goals
- Do not edit frozen ABI (`internal/abi`).
- Do not disable fail-closed dangerous gotchas (`rm -rf /`, raw disk writes, fork bombs, etc.).
- Do not perform blanket git operations (`git add -A`).
- Do not introduce random un-grandfathered scripts.
- Do not break backward compatibility with external harnesses (Claude, Codex, OpenCode).

# Plan
- [x] 1. Set default posture to `PostureDefaultOpen` (permissive allow) in `internal/agent/tools.go` and `cmd/fak/agent.go` with explicit opt-in strict profile/posture support.
- [x] 2. Register native harness (`fak` / `fak-agent`) as a first-class profile in `internal/harnessprofile/harnessprofile.go` and enable MCP registration in `cmd/fak/guard_mcp.go`.
- [x] 3. Implement default MCP tools arming (`agent.ArmMCPTools` / `gateway.MCPFloorToolDefs`) and in-process execution (`fak_read`, `fak_tools_search`, `fak_adjudicate`, `fak_syscall`, `fak_capabilities`) in `internal/agent` and `cmd/fak/agent.go`.
- [x] 4. Register `fak-native` in `internal/policy/harness-profiles.json`, `cmd/fak/guard-default-policy.json`, and `cmd/fak/guard_harness_profiles_test.go`.
- [x] 5. Write comprehensive unit and integration tests verifying default permissive allow, MCP tool execution, guard wrapping of native harness, and fail-closed security invariants.
- [x] 6. Run test witnesses and verification gates.

# Results and Verification Evidence
- **Default Permissive Allow (`PostureDefaultOpen`)**:
  - `internal/agent/tools.go`: Set `activePosture = adjudicator.PostureDefaultOpen` as the default posture for `Configure()`. Unlisted benign tools are allowed with `VerdictAllow` and `meta["posture"]="default_open"`.
  - Added `SetConfiguredPosture` and `ConfiguredPosture` for runtime adjustment.
  - Fail-closed security invariants strictly verified: `delete_account` (explicit deny) denied with `POLICY_BLOCK`; self-modify writes (e.g. into `internal/abi/` or `dos.toml`) denied with `SELF_MODIFY`; dangerous gotchas (`rm -rf /`, `kill -9 1`) denied with `POLICY_BLOCK`.
  - `cmd/fak/agent.go` and `cmd/fak/chat.go`: Added `--posture` flag defaulting to `default_open` with support for `fail_closed`/`strict` and `admit_and_log`.
  - Pick up `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` when `--base-url` is omitted under `fak guard`.
- **Harness Profile & Guard Integration**:
  - `internal/harnessprofile/harnessprofile.go`: Registered `"fak"` / `"fak-agent"` in `builtins` with `WireOpenAI`, `RepointEnv`, and `RepointSettingsFile`.
  - `cmd/fak/guard_mcp.go`: Added `guardIsFakCommand` and updated `installGuardMCPRegistrationAt` to generate `fak-mcp-config.json` and append `--mcp-config` to `fak agent`.
  - `cmd/fak/agent.go`: Added `--mcp-config` flag and `FAK_MCP_CONFIG` env support.
- **Native MCP Features Arming and Execution**:
  - `internal/agent/mcptools.go`: Implemented `ArmMCPTools`, `DisarmMCPTools`, `MCPToolCatalog`, `mcpToolGate` (rank 22), and `inProcessMCPEngine` ("inprocess_mcp").
  - Exposed 5 native MCP tools: `fak_read` (routed to `readEngine`), `fak_tools_search` (progressive disclosure), `fak_adjudicate` (in-syscall evaluation), `fak_syscall` (adjudicate + execute), `fak_capabilities`.
  - Default armed in `cmd/fak/agent.go` and `cmd/fak/chat.go` via `--mcp-tools` (default `true`).
- **Policy Profiles & Anti-Drift Floor Verification**:
  - `internal/policy/harness-profiles.json`: Registered `"fak-native"` with required tools.
  - `cmd/fak/guard-default-policy.json`: Added `"fetch_web"` to `allow`.
  - `cmd/fak/guard_harness_profiles_test.go`: Added `fak-native` profile and verified against guard default policy.
- **Witness Tests**:
  - `go test -v ./internal/agent/... -run 'TestNativeHarness|TestMCP|TestConfigured'`: 11/11 PASS.
  - `go test -v ./internal/harnessprofile/...`: 18/18 PASS.
  - `go test -v ./internal/policy/...`: PASS.
  - `go test -v ./cmd/fak -run 'TestGuardNativeHarness|TestAgentMCP|TestHarnessProfile|TestGuardMCP'`: 15/15 PASS.
  - `go vet ./internal/agent/... ./internal/harnessprofile/... ./internal/policy/... ./cmd/fak/...`: 0 diagnostics.

# Scratch / last-refusal
- All witness test suites passed cleanly with exit code 0.
