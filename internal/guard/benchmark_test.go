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
