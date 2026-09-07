// Package guard benchmarks measure tool reconciliation, capability profile
// authorization checks, ruleset token encoding, and sandbox environment
// synthesis for execution security gates.
//
// Authorization queries execute on the hot path for every tool invocation,
// requiring zero allocations and low nanosecond latency to avoid degrading
// model-tool turnaround times.
package guard

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var (
	benchReconcileSink ReconciliationResult
	benchBoolSink      bool
	benchStringSink    string
	benchSpecSink      RulesetSpec
	benchStringsSink   []string
	benchReasonSink    FailOpenReason
)

// BenchmarkReconcileCatalog_Standard measures catalog reconciliation between a standard 7-tool catalog and standard capability profile.
// Operating envelope: 7 catalog tools against an allowed set of 4 tools with 1 known alias.
// Allocation budget: <= 10 allocs/op and <= 1 KB/op for ReconciliationResult slices.
// Latency ceiling: P50 < 2µs, P99 < 10µs with linear scaling on tool count.
func BenchmarkReconcileCatalog_Standard(b *testing.B) {
	cat := ToolCatalog{
		Version: "v1.2",
		Harness: "claude",
		Tools: []string{
			"bash", "read_file", "write_file", "edit_file",
			"exec_command", "view_image", "custom_script",
		},
	}
	profile := CapabilityProfile{
		Name:    "standard-agent",
		Version: "v1.2",
		AllowedTools: []string{
			"bash", "read_file", "write_file", "edit_file",
		},
		KnownAliases: map[string]string{
			"exec_command": "bash",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReconcileSink = ReconcileCatalog(cat, profile)
	}
}

// BenchmarkReconcileCatalog_LargeMatrix measures catalog reconciliation under a large 64-tool fleet matrix.
// Operating envelope: 64 catalog tools evaluated against 32 allowed tools and 16 alias mappings.
// Allocation budget: <= 20 allocs/op and <= 5 KB/op.
// Latency ceiling: P50 < 15µs, P99 < 60µs.
func BenchmarkReconcileCatalog_LargeMatrix(b *testing.B) {
	tools := make([]string, 64)
	for i := range tools {
		tools[i] = fmt.Sprintf("tool_%d", i)
	}
	allowed := make([]string, 32)
	for i := range allowed {
		allowed[i] = fmt.Sprintf("tool_%d", i*2)
	}
	aliases := make(map[string]string, 16)
	for i := 0; i < 16; i++ {
		aliases[fmt.Sprintf("tool_alias_%d", i)] = fmt.Sprintf("tool_%d", i)
	}

	cat := ToolCatalog{
		Version: "v2.0",
		Harness: "codex-fleet",
		Tools:   tools,
	}
	profile := CapabilityProfile{
		Name:         "fleet-profile",
		Version:      "v2.0",
		AllowedTools: allowed,
		KnownAliases: aliases,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReconcileSink = ReconcileCatalog(cat, profile)
	}
}

// BenchmarkCapabilityProfile_Allows measures fast-path authorization checks for tool invocation requests.
// Operating envelope: cyclic authorization queries across 5 tools (direct match, alias match, and unauthorized).
// Allocation budget: 0 allocs/op on the membership evaluation path.
// Latency ceiling: P50 < 40ns, P99 < 200ns.
func BenchmarkCapabilityProfile_Allows(b *testing.B) {
	profile := CapabilityProfile{
		Name: "standard",
		AllowedTools: []string{
			"bash", "read_file", "write_file", "edit_file", "glob", "grep",
		},
		KnownAliases: map[string]string{
			"exec_command": "bash",
			"run_shell":    "bash",
			"view":         "read_file",
		},
	}
	testTools := []string{
		"bash", "read_file", "exec_command", "view", "unauthorized_tool",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = profile.Allows(testTools[i%len(testTools)])
	}
}

// BenchmarkRulesetSpec_Encode measures serialization of RulesetSpec containment rules into an opaque token.
// Operating envelope: single RulesetSpec structure with 3 read-only directories.
// Allocation budget: <= 3 allocs/op and <= 256 B/op for encoded token formatting.
// Latency ceiling: P50 < 300ns, P99 < 1.5µs.
func BenchmarkRulesetSpec_Encode(b *testing.B) {
	spec := RulesetSpec{
		RepoRoot: "/var/work/fak",
		GitDir:   "/var/work/fak/.git",
		ReadOnlyDirs: []string{
			"/var/work/fak/.git/hooks",
			"/etc/git-hooks/core",
			"/var/shared/security/hooks",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = spec.Encode()
	}
}

// BenchmarkRulesetSpec_Decode measures parsing and validation of an encoded ruleset specification token.
// Operating envelope: encoded token containing repo root, git directory, and 3 read-only paths.
// Allocation budget: <= 8 allocs/op and <= 512 B/op.
// Latency ceiling: P50 < 500ns, P99 < 2µs.
func BenchmarkRulesetSpec_Decode(b *testing.B) {
	spec := RulesetSpec{
		RepoRoot: "/var/work/fak",
		GitDir:   "/var/work/fak/.git",
		ReadOnlyDirs: []string{
			"/var/work/fak/.git/hooks",
			"/etc/git-hooks/core",
			"/var/shared/security/hooks",
		},
	}
	tok := spec.Encode()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := DecodeSpec(tok)
		if err != nil {
			b.Fatal(err)
		}
		benchSpecSink = s
	}
}

// BenchmarkResolveSpec measures resolution of concrete ruleset specifications from repository and hooks paths.
// Operating envelope: repository path strings resolving default read-only filesystem paths.
// Allocation budget: <= 6 allocs/op and <= 512 B/op.
// Latency ceiling: P50 < 500ns, P99 < 2.5µs.
func BenchmarkResolveSpec(b *testing.B) {
	root := "/var/work/fak"
	gitDir := "/var/work/fak/.git"
	hooksPath := ".git/hooks"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSpecSink = ResolveSpec(root, gitDir, hooksPath, false)
	}
}

// BenchmarkTrampolineArgvAndSplit measures trampoline command-line argument encapsulation and reverse parsing.
// Operating envelope: 4-argument agent invocation command line wrapped with safety trampoline flags.
// Allocation budget: <= 10 allocs/op and <= 1 KB/op.
// Latency ceiling: P50 < 800ns, P99 < 3.5µs.
func BenchmarkTrampolineArgvAndSplit(b *testing.B) {
	spec := RulesetSpec{
		RepoRoot: "/var/work/fak",
		GitDir:   "/var/work/fak/.git",
		ReadOnlyDirs: []string{
			"/var/work/fak/.git/hooks",
			"/etc/git-hooks",
		},
	}
	agentArgv := []string{"claude", "--session", "sess_12345", "--dangerously-skip-permissions"}
	bin := "/usr/local/bin/fak"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		argv := TrampolineArgv(bin, spec, agentArgv)
		tok, agent, ok := SplitTrampolineArgs(argv[2:])
		if !ok {
			b.Fatal("SplitTrampolineArgs failed")
		}
		benchStringSink = tok
		benchStringsSink = agent
	}
}

// BenchmarkDecideFailOpen measures fail-open kernel decision evaluation across Landlock version and errno states.
// Operating envelope: cyclic evaluation across 5 kernel capability cases including ENOSYS and EOPNOTSUPP.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 15ns, P99 < 80ns.
func BenchmarkDecideFailOpen(b *testing.B) {
	cases := []struct {
		version int
		errno   int
	}{
		{version: 1, errno: 0},
		{version: 5, errno: 0},
		{version: -1, errno: errnoENOSYS},
		{version: -1, errno: errnoEOPNOTSUPP},
		{version: 0, errno: 0},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i%len(cases)]
		benchReasonSink = DecideFailOpen(c.version, c.errno)
	}
}

// BenchmarkOptedIn measures opt-in predicate resolution from injected environment lookup functions.
// Operating envelope: mock environment reader returning truthy opt-in flag.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 20ns, P99 < 100ns.
func BenchmarkOptedIn(b *testing.B) {
	getenv := func(key string) string {
		return "1"
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = OptedIn(getenv)
	}
}

// BenchmarkPolicyOriginEvidencePath measures formatting and canonicalization of policy evidence artifact paths.
// Operating envelope: alternating queries between default workspace trace path and explicit custom path.
// Allocation budget: <= 2 allocs/op and <= 128 B/op.
// Latency ceiling: P50 < 150ns, P99 < 800ns.
func BenchmarkPolicyOriginEvidencePath(b *testing.B) {
	root := "/work/fak"
	traceID := "trace-benchmark-987654321"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			benchStringSink = PolicyOriginEvidencePath(root, traceID, "")
		} else {
			benchStringSink = PolicyOriginEvidencePath(root, traceID, "/etc/policies/custom.json")
		}
	}
}

// BenchmarkEnsureOriginEvidence_Existing measures verification and path resolution for existing evidence files.
// Operating envelope: temporary filesystem file verified through stat calls.
// Allocation budget: <= 2 allocs/op and <= 128 B/op.
// Latency ceiling: P50 < 1µs, P99 < 5µs.
func BenchmarkEnsureOriginEvidence_Existing(b *testing.B) {
	tempDir := b.TempDir()
	path := filepath.Join(tempDir, "existing-evidence.json")
	if err := os.WriteFile(path, []byte(`{"trace":"bench"}`), 0o600); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, ok := EnsureOriginEvidence(path)
		if !ok {
			b.Fatal("EnsureOriginEvidence failed")
		}
		benchStringSink = p
	}
}

// BenchmarkLandlockChildEnv measures child process environment variable synthesis for sandboxed execution.
// Operating envelope: extra role and run-id environment variables appended to sanitized environment.
// Allocation budget: <= 2 allocs/op and <= 256 B/op.
// Latency ceiling: P50 < 100ns, P99 < 500ns.
func BenchmarkLandlockChildEnv(b *testing.B) {
	extra := []string{
		"FAK_AGENT_ROLE=worker",
		"FAK_RUN_ID=run-bench-123",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringsSink = landlockChildEnv(extra...)
	}
}
