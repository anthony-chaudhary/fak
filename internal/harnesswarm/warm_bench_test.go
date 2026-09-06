package harnesswarm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var benchSink any

func TestBenchmarkSanity(t *testing.T) {
	if AllStages == nil || len(AllStages) == 0 {
		t.Fatal("expected AllStages to be defined")
	}
}

func setupBenchWorkspace(b *testing.B) string {
	b.Helper()
	dir := b.TempDir()

	files := map[string]string{
		"go.mod":     "module example.com/benchproject\n\ngo 1.26\n",
		"go.sum":     "example.com/benchproject v0.1.0 h1:xyz=\n",
		".gitignore": "bin/\n*.tmp\nvendor/\n.build/\n*.log\n",
		".fakignore": "scratch/\n*.bak\n",
		"dos.toml":   "[lanes]\ndefault = \"*\"\n",
		"README.md":  "# Bench Workspace\nPerformance test workspace.\n",
		"main.go": `package main

import "fmt"

type ServerConfig struct {
	Port int
	Host string
}

var DefaultPort = 8080

func RunServer(cfg ServerConfig) error {
	fmt.Println("running")
	return nil
}

func (s *ServerConfig) Addr() string {
	return fmt.Sprintf("%s:%d", s.Host, s.Port)
}

func main() {}
`,
		"pkg/service.go": `package pkg

type Processor interface {
	Process(data []byte) error
}

type DefaultProcessor struct {
	Retries int
}

func (d *DefaultProcessor) Process(data []byte) error {
	return nil
}

func NewProcessor(retries int) *DefaultProcessor {
	return &DefaultProcessor{Retries: retries}
}
`,
		"scripts/worker.py": `def handle_event(event):
    pass

def parse_args():
    return {}

class TaskWorker:
    def __init__(self):
        pass

    def run(self):
        pass
`,
		"frontend/view.ts": `export function renderHeader() {}

export function mountView() {}

export class DashboardController {
    refresh() {}
}
`,
		"native/lib.rs": `pub fn calculate_hash(data: &[u8]) -> u64 {
    0
}

pub struct StateBuffer {
    size: usize,
}
`,
		"nested/.gitignore": "local.cache\n",
	}

	for rel, content := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			b.Fatalf("mkdir %s: %v", filepath.Dir(full), err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			b.Fatalf("write %s: %v", full, err)
		}
	}

	// Add additional nested files to simulate a multi-file workspace inventory.
	for i := 0; i < 50; i++ {
		p := filepath.Join(dir, fmt.Sprintf("src/module%02d/file%02d.go", i/5, i%5))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			b.Fatalf("mkdir: %v", err)
		}
		content := fmt.Sprintf("package module%02d\n\nfunc Operation%02d() int { return %d }\n", i/5, i%5, i)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			b.Fatalf("write: %v", err)
		}
	}

	return dir
}

func BenchmarkEngineSnapshot(b *testing.B) {
	ws := setupBenchWorkspace(b)
	engine := NewEngine(ws, Options{})
	defer engine.Close()

	ctx := context.Background()
	engine.Start(ctx)
	for _, stage := range AllStages {
		if err := engine.WaitStage(ctx, stage); err != nil {
			b.Fatalf("WaitStage(%s): %v", stage, err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap := engine.Snapshot()
		if !snap.AllWarm {
			b.Fatal("expected snapshot AllWarm == true")
		}
		benchSink = snap
	}
}

func BenchmarkEngineIsWarm(b *testing.B) {
	ws := setupBenchWorkspace(b)
	engine := NewEngine(ws, Options{})
	defer engine.Close()

	ctx := context.Background()
	engine.Start(ctx)
	for _, stage := range AllStages {
		if err := engine.WaitStage(ctx, stage); err != nil {
			b.Fatalf("WaitStage(%s): %v", stage, err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !engine.IsWarm(StageFiles) {
			b.Fatal("expected StageFiles warm")
		}
	}
}

func BenchmarkClassifyPathStages(b *testing.B) {
	engine := NewEngine(".", Options{})
	defer engine.Close()

	cases := []struct {
		name  string
		paths []string
	}{
		{"Manifest", []string{"go.mod", "package.json", "dos.toml"}},
		{"Source", []string{"main.go", "pkg/service.go", "frontend/view.ts"}},
		{"Ignore", []string{".gitignore", ".fakignore", ".ignore"}},
		{"Mixed", []string{"go.mod", "main.go", ".gitignore", "README.md"}},
		{"Wildcard", []string{""}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res := engine.classifyPathStagesLocked(tc.paths)
				benchSink = res
			}
		})
	}
}

func BenchmarkExecuteManifestsStage(b *testing.B) {
	engine := NewEngine(".", Options{})
	defer engine.Close()

	files := make([]string, 0, 200)
	files = append(files, "go.mod", "go.sum", "dos.toml", "package.json", "Cargo.toml")
	for i := 0; i < 195; i++ {
		files = append(files, fmt.Sprintf("src/file_%03d.go", i))
	}

	engine.mu.Lock()
	engine.files = files
	engine.mu.Unlock()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := engine.executeManifestsStage(); err != nil {
			b.Fatalf("executeManifestsStage: %v", err)
		}
	}
}

func BenchmarkExecuteIgnoreStage(b *testing.B) {
	ws := setupBenchWorkspace(b)
	engine := NewEngine(ws, Options{})
	defer engine.Close()

	// Populate files so nested ignore detection is exercised.
	if err := engine.executeFilesStage(); err != nil {
		b.Fatalf("executeFilesStage: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := engine.executeIgnoreStage(); err != nil {
			b.Fatalf("executeIgnoreStage: %v", err)
		}
	}
}

func BenchmarkExtractGoSymbols(b *testing.B) {
	ws := setupBenchWorkspace(b)
	filePath := filepath.Join(ws, "main.go")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		syms := extractGoSymbols(filePath)
		if len(syms) == 0 {
			b.Fatal("expected extracted symbols")
		}
		benchSink = syms
	}
}

func BenchmarkExtractLineSymbols(b *testing.B) {
	ws := setupBenchWorkspace(b)

	targets := []struct {
		name string
		rel  string
		ext  string
	}{
		{"Python", "scripts/worker.py", ".py"},
		{"TypeScript", "frontend/view.ts", ".ts"},
		{"Rust", "native/lib.rs", ".rs"},
	}

	for _, tc := range targets {
		b.Run(tc.name, func(b *testing.B) {
			path := filepath.Join(ws, tc.rel)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				syms := extractLineSymbols(path, tc.ext)
				if len(syms) == 0 {
					b.Fatal("expected extracted symbols")
				}
				benchSink = syms
			}
		})
	}
}

func BenchmarkFullWarming(b *testing.B) {
	ws := setupBenchWorkspace(b)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine := NewEngine(ws, Options{})
		engine.Start(ctx)
		for _, stage := range AllStages {
			if err := engine.WaitStage(ctx, stage); err != nil {
				b.Fatalf("WaitStage(%s): %v", stage, err)
			}
		}
		engine.Close()
	}
}
