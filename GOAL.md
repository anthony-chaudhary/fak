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
- [ ] 1. Set default posture to `PostureDefaultOpen` (permissive allow) in `internal/agent/tools.go` and `cmd/fak/agent.go` with explicit opt-in strict profile/posture support.
- [ ] 2. Register native harness (`fak` / `fak-agent`) as a first-class profile in `internal/harnessprofile/harnessprofile.go` and enable MCP registration in `cmd/fak/guard_mcp.go`.
- [ ] 3. Implement default MCP tools arming (`agent.ArmMCPTools` / `gateway.MCPFloorToolDefs`) and in-process execution (`fak_read`, `fak_tools_search`, `fak_adjudicate`, `fak_syscall`) in `internal/agent` and `cmd/fak/agent.go`.
- [ ] 4. Register `fak-native` in `internal/policy/harness-profiles.json`, `cmd/fak/guard-default-policy.json`, and `cmd/fak/guard_harness_profiles_test.go`.
- [ ] 5. Write comprehensive unit and integration tests verifying default permissive allow, MCP tool execution, guard wrapping of native harness, and fail-closed security invariants.
- [ ] 6. Run test witnesses and verification gates.

# Scratch / last-refusal
