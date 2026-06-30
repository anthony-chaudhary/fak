package codexmemory

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeConfig writes a config.toml into a fresh temp Codex home and returns the
// home path.
func writeConfig(t *testing.T, body string) string {
	t.Helper()
	home := t.TempDir()
	if body != "" {
		if err := os.WriteFile(filepath.Join(home, "config.toml"), []byte(body), 0o644); err != nil {
			t.Fatalf("write config.toml: %v", err)
		}
	}
	return home
}

func findingCodes(p Posture) map[string]Finding {
	m := map[string]Finding{}
	for _, f := range p.Findings {
		m[f.Code] = f
	}
	return m
}

func TestMissingConfig(t *testing.T) {
	home := t.TempDir() // no config.toml
	p := Doctor(Options{CodexHome: home})
	if p.ConfigExists {
		t.Fatalf("expected ConfigExists=false for a home with no config.toml")
	}
	if p.MemoriesEnabled.Set {
		t.Fatalf("expected MemoriesEnabled unset when no config present")
	}
	if !p.OK {
		t.Fatalf("a bare home with no memories is healthy posture; got OK=false findings=%+v", p.Findings)
	}
	if _, ok := findingCodes(p)["config-absent"]; !ok {
		t.Fatalf("expected a config-absent finding, got %+v", p.Findings)
	}
}

func TestEmptyHomeUnresolved(t *testing.T) {
	// No flag, env returns nothing, home dir resolution fails -> unresolved.
	p := Doctor(Options{
		Env:     func(string) string { return "" },
		HomeDir: func() (string, error) { return "", os.ErrNotExist },
	})
	if p.CodexHome != "" {
		t.Fatalf("expected unresolved CodexHome, got %q", p.CodexHome)
	}
	if !p.OK {
		t.Fatalf("unresolved home is not itself a risk; got OK=false")
	}
	if _, ok := findingCodes(p)["home-unresolved"]; !ok {
		t.Fatalf("expected home-unresolved finding, got %+v", p.Findings)
	}
}

func TestEnvAndDefaultResolution(t *testing.T) {
	envHome := t.TempDir()
	p := Doctor(Options{Env: func(k string) string {
		if k == "CODEX_HOME" {
			return envHome
		}
		return ""
	}})
	if p.HomeSource != "env" || p.CodexHome != envHome {
		t.Fatalf("CODEX_HOME not honored: source=%q home=%q", p.HomeSource, p.CodexHome)
	}

	fakeHome := t.TempDir()
	p2 := Doctor(Options{
		Env:     func(string) string { return "" },
		HomeDir: func() (string, error) { return fakeHome, nil },
	})
	if p2.HomeSource != "default" {
		t.Fatalf("expected default home source, got %q", p2.HomeSource)
	}
	if p2.CodexHome != filepath.Join(fakeHome, ".codex") {
		t.Fatalf("default home should be <userhome>/.codex, got %q", p2.CodexHome)
	}

	// Explicit flag wins over env.
	flagHome := t.TempDir()
	p3 := Doctor(Options{CodexHome: flagHome, Env: func(string) string { return envHome }})
	if p3.HomeSource != "flag" || p3.CodexHome != flagHome {
		t.Fatalf("--codex-home should win: source=%q home=%q", p3.HomeSource, p3.CodexHome)
	}
}

func TestMemoriesDisabled(t *testing.T) {
	home := writeConfig(t, `
[features]
memories = false
`)
	p := Doctor(Options{CodexHome: home})
	if !p.MemoriesEnabled.Set || p.MemoriesEnabled.Value {
		t.Fatalf("expected memories explicitly false; got %+v", p.MemoriesEnabled)
	}
	if !p.OK {
		t.Fatalf("memories disabled is healthy; got findings %+v", p.Findings)
	}
}

func TestMemoriesEnabledExternalExcluded(t *testing.T) {
	home := writeConfig(t, `
[features]
memories = true

[memories]
use_memories = true
generate_memories = true
disable_on_external_context = true
min_rate_limit_remaining_percent = 25
extract_model = "gpt-extract"
consolidation_model = "gpt-consolidate"
`)
	p := Doctor(Options{CodexHome: home})
	if !p.MemoriesEnabled.Value || !p.UseMemories.Value || !p.GenerateMemories.Value {
		t.Fatalf("expected all three enabled; got enabled=%+v use=%+v gen=%+v",
			p.MemoriesEnabled, p.UseMemories, p.GenerateMemories)
	}
	if !p.DisableOnExternalContext.Set || !p.DisableOnExternalContext.Value {
		t.Fatalf("expected disable_on_external_context=true; got %+v", p.DisableOnExternalContext)
	}
	if !p.RateLimitFloorSet || p.RateLimitFloor == nil || *p.RateLimitFloor != 25 {
		t.Fatalf("expected rate-limit floor 25; got set=%v val=%v", p.RateLimitFloorSet, p.RateLimitFloor)
	}
	if p.ExtractModel != "gpt-extract" || p.ConsolidationModel != "gpt-consolidate" {
		t.Fatalf("model keys not read: extract=%q consolidation=%q", p.ExtractModel, p.ConsolidationModel)
	}
	if _, ok := findingCodes(p)["external-context-included"]; ok {
		t.Fatalf("external-context-included risk should NOT fire when excluded; findings=%+v", p.Findings)
	}
	if !p.OK {
		t.Fatalf("safe external-context exclusion should be healthy; findings=%+v", p.Findings)
	}
}

func TestMemoriesEnabledExternalIncluded(t *testing.T) {
	home := writeConfig(t, `
[features]
memories = true

[memories]
use_memories = true
generate_memories = true
disable_on_external_context = false
`)
	p := Doctor(Options{CodexHome: home})
	f, ok := findingCodes(p)["external-context-included"]
	if !ok {
		t.Fatalf("expected external-context-included finding; got %+v", p.Findings)
	}
	if !f.Risk {
		t.Fatalf("external-context-included must be a risk finding")
	}
	if p.OK {
		t.Fatalf("expected risky posture (OK=false) when external context is included")
	}
}

func TestLargeMemoryDirectory(t *testing.T) {
	home := writeConfig(t, "[features]\nmemories = true\n")
	memDir := filepath.Join(home, "memories")
	if err := os.MkdirAll(memDir, 0o755); err != nil {
		t.Fatal(err)
	}
	big := make([]byte, largeMemoryBytes+1024)
	if err := os.WriteFile(filepath.Join(memDir, "huge.md"), big, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(memDir, "small.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := Doctor(Options{CodexHome: home})
	if !p.Memories.Exists || p.Memories.Files != 2 {
		t.Fatalf("expected 2 memory files; got exists=%v files=%d", p.Memories.Exists, p.Memories.Files)
	}
	if p.Memories.Bytes < largeMemoryBytes {
		t.Fatalf("expected bytes >= large threshold; got %d", p.Memories.Bytes)
	}
	if p.Memories.LargestBytes < largeMemoryBytes {
		t.Fatalf("expected largest file recorded; got %d", p.Memories.LargestBytes)
	}
	f, ok := findingCodes(p)["large-memory-store"]
	if !ok || !f.Risk {
		t.Fatalf("expected risky large-memory-store finding; got %+v", p.Findings)
	}
	// Artifact names are surfaced; raw content never is.
	joined := strings.Join(p.Memories.Artifacts, ",")
	if !strings.Contains(joined, "huge.md") || !strings.Contains(joined, "small.md") {
		t.Fatalf("expected artifact names listed, got %q", joined)
	}
}

func TestChroniclePresent(t *testing.T) {
	home := writeConfig(t, "[features]\nmemories = true\n")
	chronDir := filepath.Join(home, "memories_extensions", "chronicle")
	if err := os.MkdirAll(chronDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"screen1.json", "screen2.json"} {
		if err := os.WriteFile(filepath.Join(chronDir, n), []byte(`{"derived":"screen"}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	p := Doctor(Options{CodexHome: home})
	if !p.Chronicle.Exists || p.Chronicle.Files != 2 {
		t.Fatalf("expected 2 chronicle files; got exists=%v files=%d", p.Chronicle.Exists, p.Chronicle.Files)
	}
	f, ok := findingCodes(p)["chronicle-present"]
	if !ok || !f.Risk {
		t.Fatalf("expected risky chronicle-present finding; got %+v", p.Findings)
	}
	if p.OK {
		t.Fatalf("Chronicle present should mark posture risky")
	}
}

func TestRepoGuidanceBoundary(t *testing.T) {
	home := writeConfig(t, "[features]\nmemories = true\n")

	// Repo without AGENTS.md -> advisory finding (not a risk).
	repoNo := t.TempDir()
	p := Doctor(Options{CodexHome: home, RepoRoot: repoNo})
	if p.AgentsMD {
		t.Fatalf("expected AgentsMD=false for repo without AGENTS.md")
	}
	if _, ok := findingCodes(p)["agents-md-absent"]; !ok {
		t.Fatalf("expected agents-md-absent finding; got %+v", p.Findings)
	}

	// Repo with AGENTS.md -> present, no finding.
	repoYes := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoYes, "AGENTS.md"), []byte("# agents"), 0o644); err != nil {
		t.Fatal(err)
	}
	p2 := Doctor(Options{CodexHome: home, RepoRoot: repoYes})
	if !p2.AgentsMD {
		t.Fatalf("expected AgentsMD=true when AGENTS.md present")
	}
	if _, ok := findingCodes(p2)["agents-md-absent"]; ok {
		t.Fatalf("agents-md-absent should not fire when AGENTS.md present; got %+v", p2.Findings)
	}
}

func TestRenderIsContentFree(t *testing.T) {
	home := writeConfig(t, "[features]\nmemories = true\n\n[memories]\ndisable_on_external_context = false\n")
	memDir := filepath.Join(home, "memories")
	if err := os.MkdirAll(memDir, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := "TOP-SECRET-MEMORY-BODY-do-not-print"
	if err := os.WriteFile(filepath.Join(memDir, "note.md"), []byte(secret), 0o644); err != nil {
		t.Fatal(err)
	}
	p := Doctor(Options{CodexHome: home})
	out := Render(p)
	if strings.Contains(out, secret) {
		t.Fatalf("Render leaked raw memory content:\n%s", out)
	}
	if !strings.Contains(out, "codex-memory doctor:") {
		t.Fatalf("Render missing header:\n%s", out)
	}
	if !strings.Contains(out, "WARN") {
		t.Fatalf("expected a WARN tag for external-context inclusion:\n%s", out)
	}
}

func TestParseFlatTOMLComments(t *testing.T) {
	kv := parseFlatTOML(`
# a comment line
[memories]
use_memories = true # trailing comment
extract_model = "gpt#hash" # '#' inside quotes is not a comment
`)
	if kv["memories.use_memories"] != "true" {
		t.Fatalf("trailing comment not stripped: %q", kv["memories.use_memories"])
	}
	if got := unquote(kv["memories.extract_model"]); got != "gpt#hash" {
		t.Fatalf("quoted '#' mishandled: %q", got)
	}
}
