package codexmemory

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var (
	benchPostureSink Posture
	benchStringSink  string
	benchSourceSink  string
	benchMapSink     map[string]string
	benchInvSink     DirInventory
)

const sampleConfigTOML = `
# Codex core memory configuration
[features]
memories = true # enable persistent memories across sessions

# Memories subsystem fine-tuning
[memories]
use_memories = true
generate_memories = true
disable_on_external_context = true # exclude MCP/web threads
min_rate_limit_remaining_percent = 20
extract_model = "gpt-4o-extract#v1" # hash in string must be preserved
consolidation_model = 'gpt-4o-consolidate#v2' # single quote test

# Section to be ignored
[model]
name = "o3-mini" # default reasoning model
temperature = 0.2
`

const denseCommentTOML = `
# Header comment line 1
# Header comment line 2
[features]
# Feature toggle
memories = true # inline comment
# Another comment line
[memories]
# Memory options
use_memories = true # comment
generate_memories = true # comment
disable_on_external_context = false # comment
# Rate limit config
min_rate_limit_remaining_percent = 50 # comment
extract_model = "model#123" # comment
consolidation_model = "model#456" # comment
# Trailing comment
`

// TestBenchmarkSanity ensures all benchmarked paths execute cleanly and produce valid output.
func TestBenchmarkSanity(t *testing.T) {
	// Sanity test for ParseFlatTOML
	parsed := parseFlatTOML(sampleConfigTOML)
	if parsed["features.memories"] != "true" {
		t.Fatalf("expected features.memories=true, got %q", parsed["features.memories"])
	}
	if parsed["memories.extract_model"] != `"gpt-4o-extract#v1"` {
		t.Fatalf("expected preserved quoted hash, got %q", parsed["memories.extract_model"])
	}

	// Sanity test for ResolveHome
	home := t.TempDir()
	pFlag := Doctor(Options{CodexHome: home})
	if pFlag.CodexHome != home || pFlag.HomeSource != "flag" {
		t.Fatalf("unexpected flag resolution: %+v", pFlag)
	}

	// Sanity test for RenderGuidance
	rendered := Render(pFlag)
	if rendered == "" {
		t.Fatal("expected non-empty rendered string")
	}

	// Sanity test for inventory
	inv := inventory(home)
	if !inv.Exists {
		t.Fatal("expected inventory Exists=true for created dir")
	}

	// Sanity test for deriveFindings
	p := Posture{
		MemoriesEnabled:          triBool{Set: true, Value: true},
		DisableOnExternalContext: triBool{Set: true, Value: false},
	}
	deriveFindings(&p)
	if len(p.Findings) == 0 {
		t.Fatal("expected findings for external context inclusion")
	}
}

// BenchmarkResolveHome measures Codex home directory resolution across flag, env, default, and unresolved paths.
func BenchmarkResolveHome(b *testing.B) {
	b.Run("Flag", func(b *testing.B) {
		opts := Options{CodexHome: "/explicit/codex/path"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink, benchSourceSink = resolveHome(opts)
		}
	})
	b.Run("Env", func(b *testing.B) {
		opts := Options{
			Env: func(k string) string {
				if k == "CODEX_HOME" {
					return "/env/codex/path"
				}
				return ""
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink, benchSourceSink = resolveHome(opts)
		}
	})
	b.Run("Default", func(b *testing.B) {
		opts := Options{
			Env: func(string) string { return "" },
			HomeDir: func() (string, error) {
				return "/home/operator", nil
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink, benchSourceSink = resolveHome(opts)
		}
	})
	b.Run("Unresolved", func(b *testing.B) {
		opts := Options{
			Env: func(string) string { return "" },
			HomeDir: func() (string, error) {
				return "", os.ErrNotExist
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink, benchSourceSink = resolveHome(opts)
		}
	})
}

// BenchmarkParseFlatTOMLComments measures TOML parsing with comments, quotes, and section tables.
func BenchmarkParseFlatTOMLComments(b *testing.B) {
	b.Run("Standard", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(sampleConfigTOML)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchMapSink = parseFlatTOML(sampleConfigTOML)
		}
	})
	b.Run("DenseComments", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(denseCommentTOML)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchMapSink = parseFlatTOML(denseCommentTOML)
		}
	})
}

// BenchmarkRenderGuidance measures formatting the diagnostic posture report into the human-readable string.
func BenchmarkRenderGuidance(b *testing.B) {
	rateLimit := 25
	base := Posture{
		Schema:                   Schema,
		CodexHome:                "/home/developer/.codex",
		HomeSource:               "default",
		ConfigPath:               "/home/developer/.codex/config.toml",
		ConfigExists:             true,
		MemoriesEnabled:          triBool{Set: true, Value: true},
		GenerateMemories:         triBool{Set: true, Value: true},
		UseMemories:              triBool{Set: true, Value: true},
		DisableOnExternalContext: triBool{Set: true, Value: true},
		RateLimitFloor:           &rateLimit,
		RateLimitFloorSet:        true,
		ExtractModel:             "gpt-extract",
		ConsolidationModel:       "gpt-consolidate",
		Memories: DirInventory{
			Path:   "/home/developer/.codex/memories",
			Exists: true,
			Files:  42,
			Bytes:  128 * 1024,
		},
		Chronicle: DirInventory{
			Path:   "/home/developer/.codex/memories_extensions/chronicle",
			Exists: false,
		},
		RepoRoot:     "/work/fak",
		AgentsMD:     true,
		AgentsMDPath: "/work/fak/AGENTS.md",
		Findings: []Finding{
			{Code: "rate-limit-floor-default", Message: "advisory notice", Risk: false},
		},
		OK: true,
	}

	b.Run("Present", func(b *testing.B) {
		p := base
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Render(p)
		}
	})

	b.Run("Absent", func(b *testing.B) {
		p := base
		p.AgentsMD = false
		p.AgentsMDPath = ""
		p.Findings = append([]Finding{}, base.Findings...)
		p.Findings = append(p.Findings, Finding{
			Code:    "agents-md-absent",
			Message: "no AGENTS.md in the repo — team/repo invariants should be checked in",
			Risk:    false,
		})
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Render(p)
		}
	})

	b.Run("RiskFindings", func(b *testing.B) {
		p := base
		p.DisableOnExternalContext = triBool{Set: true, Value: false}
		p.Chronicle = DirInventory{
			Path:   "/home/developer/.codex/memories_extensions/chronicle",
			Exists: true,
			Files:  8,
			Bytes:  1024 * 1024,
		}
		p.Findings = []Finding{
			{Code: "external-context-included", Message: "external context enabled", Risk: true},
			{Code: "chronicle-present", Message: "chronicle present", Risk: true},
		}
		p.OK = false
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Render(p)
		}
	})
}

// BenchmarkInventory measures scanning and summarizing memory directory structures.
func BenchmarkInventory(b *testing.B) {
	dir := b.TempDir()
	subDir := filepath.Join(dir, "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		b.Fatal(err)
	}
	content := []byte("memory file payload")
	for i := 0; i < 10; i++ {
		p := filepath.Join(dir, fmt.Sprintf("file_%02d.md", i))
		if err := os.WriteFile(p, content, 0o644); err != nil {
			b.Fatal(err)
		}
	}
	for i := 0; i < 5; i++ {
		p := filepath.Join(subDir, fmt.Sprintf("subfile_%02d.md", i))
		if err := os.WriteFile(p, content, 0o644); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchInvSink = inventory(dir)
	}
}

// BenchmarkApplyConfig measures folding parsed TOML key-values into Posture.
func BenchmarkApplyConfig(b *testing.B) {
	parsed := parseFlatTOML(sampleConfigTOML)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var p Posture
		applyConfig(&p, parsed)
		benchPostureSink = p
	}
}

// BenchmarkDeriveFindings measures finding derivation across healthy and risky postures.
func BenchmarkDeriveFindings(b *testing.B) {
	b.Run("Healthy", func(b *testing.B) {
		base := Posture{
			MemoriesEnabled:          triBool{Set: true, Value: true},
			DisableOnExternalContext: triBool{Set: true, Value: true},
			RateLimitFloorSet:        true,
			RepoRoot:                 "/repo",
			AgentsMD:                 true,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := base
			deriveFindings(&p)
			benchPostureSink = p
		}
	})

	b.Run("Risky", func(b *testing.B) {
		base := Posture{
			MemoriesEnabled:          triBool{Set: true, Value: true},
			DisableOnExternalContext: triBool{Set: true, Value: false},
			Memories:                 DirInventory{Bytes: largeMemoryBytes + 1024},
			Chronicle:                DirInventory{Exists: true, Files: 4, Bytes: 2048},
			RepoRoot:                 "/repo",
			AgentsMD:                 false,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p := base
			deriveFindings(&p)
			benchPostureSink = p
		}
	})
}

// BenchmarkDoctor measures full end-to-end Doctor evaluation against disk.
func BenchmarkDoctor(b *testing.B) {
	home := b.TempDir()
	configPath := filepath.Join(home, "config.toml")
	if err := os.WriteFile(configPath, []byte(sampleConfigTOML), 0o644); err != nil {
		b.Fatal(err)
	}
	memDir := filepath.Join(home, "memories")
	if err := os.MkdirAll(memDir, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(memDir, "fact.md"), []byte("fact"), 0o644); err != nil {
		b.Fatal(err)
	}

	repoDir := b.TempDir()
	if err := os.WriteFile(filepath.Join(repoDir, "AGENTS.md"), []byte("# agents"), 0o644); err != nil {
		b.Fatal(err)
	}

	opts := Options{
		CodexHome: home,
		RepoRoot:  repoDir,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPostureSink = Doctor(opts)
	}
}
