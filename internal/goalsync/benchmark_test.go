package goalsync

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

var (
	benchArtifactsSink []Artifact
	benchStatusSink    *SyncStatus
	benchReportSink    *SyncReport
)

func setupBenchmarkEnv(b *testing.B, fileCount int) (wsRoot, targetDir, regPath, parkDir string) {
	b.Helper()
	wsRoot = b.TempDir()
	targetDir = b.TempDir()

	goalsDir := filepath.Join(wsRoot, "goals")
	subDir := filepath.Join(goalsDir, "subagents")
	fakDir := filepath.Join(wsRoot, ".fak")
	parkDir = filepath.Join(fakDir, "goal-park")

	if err := os.MkdirAll(subDir, 0755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(parkDir, 0755); err != nil {
		b.Fatal(err)
	}

	// 1. Goal specs
	for i := 0; i < fileCount; i++ {
		name := fmt.Sprintf("GOAL-%03d.md", i)
		content := fmt.Sprintf("# Goal %d\n\nAutomated goal specification %d with description and requirements.\n", i, i)
		if err := os.WriteFile(filepath.Join(goalsDir, name), []byte(content), 0644); err != nil {
			b.Fatal(err)
		}
	}

	// 2. Subagent specs
	for i := 0; i < fileCount; i++ {
		name := fmt.Sprintf("SUB-%03d.md", i)
		content := fmt.Sprintf("# Subagent %d\n\nSubagent specification %d details.\n", i, i)
		if err := os.WriteFile(filepath.Join(subDir, name), []byte(content), 0644); err != nil {
			b.Fatal(err)
		}
	}

	// 3. Registry file
	regPath = filepath.Join(fakDir, "goals.json")
	if err := os.WriteFile(regPath, []byte(`{"schema":"fak-goal-registry/1","goals":[{"id":"g1","status":"active"}]}`), 0644); err != nil {
		b.Fatal(err)
	}

	// 4. Goal park records
	for i := 0; i < fileCount; i++ {
		name := fmt.Sprintf("park-%03d.json", i)
		content := fmt.Sprintf(`{"schema":"fak.goal-park.v1","goal":"g%d","reason":"checkpoint"}`, i)
		if err := os.WriteFile(filepath.Join(parkDir, name), []byte(content), 0644); err != nil {
			b.Fatal(err)
		}
	}

	// Target directory setup
	// Half of specs copied identically
	for i := 0; i < fileCount/2; i++ {
		name := fmt.Sprintf("GOAL-%03d.md", i)
		srcData, err := os.ReadFile(filepath.Join(goalsDir, name))
		if err != nil {
			b.Fatal(err)
		}
		destPath := filepath.Join(targetDir, "goals", name)
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			b.Fatal(err)
		}
		if err := os.WriteFile(destPath, srcData, 0644); err != nil {
			b.Fatal(err)
		}
	}
	// The other half copied with different content
	for i := fileCount / 2; i < fileCount; i++ {
		name := fmt.Sprintf("GOAL-%03d.md", i)
		destPath := filepath.Join(targetDir, "goals", name)
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			b.Fatal(err)
		}
		if err := os.WriteFile(destPath, []byte("different target content"), 0644); err != nil {
			b.Fatal(err)
		}
		future := time.Now().Add(1 * time.Hour)
		_ = os.Chtimes(destPath, future, future)
	}

	// Subagents copied identically
	for i := 0; i < fileCount; i++ {
		name := fmt.Sprintf("SUB-%03d.md", i)
		srcData, err := os.ReadFile(filepath.Join(subDir, name))
		if err != nil {
			b.Fatal(err)
		}
		destPath := filepath.Join(targetDir, "goals", "subagents", name)
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			b.Fatal(err)
		}
		if err := os.WriteFile(destPath, srcData, 0644); err != nil {
			b.Fatal(err)
		}
	}

	// Registry copied identically
	regData, err := os.ReadFile(regPath)
	if err != nil {
		b.Fatal(err)
	}
	tgtReg := filepath.Join(targetDir, "registry", "goals.json")
	if err := os.MkdirAll(filepath.Dir(tgtReg), 0755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(tgtReg, regData, 0644); err != nil {
		b.Fatal(err)
	}

	// Extra files only in target
	for i := 0; i < fileCount/2; i++ {
		extraPath := filepath.Join(targetDir, "goals", fmt.Sprintf("TARGET-EXTRA-%03d.md", i))
		if err := os.WriteFile(extraPath, []byte("extra target only"), 0644); err != nil {
			b.Fatal(err)
		}
	}

	return wsRoot, targetDir, regPath, parkDir
}

func BenchmarkDiscoverSource(b *testing.B) {
	wsRoot, _, regPath, parkDir := setupBenchmarkEnv(b, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := DiscoverSource(wsRoot, regPath, parkDir)
		if err != nil {
			b.Fatalf("DiscoverSource failed: %v", err)
		}
		benchArtifactsSink = res
	}
}

func BenchmarkDiscoverTarget(b *testing.B) {
	_, targetDir, _, _ := setupBenchmarkEnv(b, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := DiscoverTarget(targetDir)
		if err != nil {
			b.Fatalf("DiscoverTarget failed: %v", err)
		}
		benchArtifactsSink = res
	}
}

func BenchmarkDiscoverSourceAndTarget(b *testing.B) {
	wsRoot, targetDir, regPath, parkDir := setupBenchmarkEnv(b, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srcs, err := DiscoverSource(wsRoot, regPath, parkDir)
		if err != nil {
			b.Fatalf("DiscoverSource failed: %v", err)
		}
		tgts, err := DiscoverTarget(targetDir)
		if err != nil {
			b.Fatalf("DiscoverTarget failed: %v", err)
		}
		benchArtifactsSink = srcs
		if len(tgts) == 0 {
			b.Fatal("empty targets")
		}
	}
}

func BenchmarkStatus(b *testing.B) {
	wsRoot, targetDir, regPath, parkDir := setupBenchmarkEnv(b, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		st, err := Status(wsRoot, targetDir, regPath, parkDir)
		if err != nil {
			b.Fatalf("Status failed: %v", err)
		}
		benchStatusSink = st
	}
}

func BenchmarkPush_DryRun(b *testing.B) {
	wsRoot, targetDir, regPath, parkDir := setupBenchmarkEnv(b, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Push(wsRoot, targetDir, regPath, parkDir, false, false, true)
		if err != nil {
			b.Fatalf("Push dry-run failed: %v", err)
		}
		benchReportSink = report
	}
}

func BenchmarkPull_DryRun(b *testing.B) {
	wsRoot, targetDir, regPath, parkDir := setupBenchmarkEnv(b, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Pull(wsRoot, targetDir, regPath, parkDir, false, true)
		if err != nil {
			b.Fatalf("Pull dry-run failed: %v", err)
		}
		benchReportSink = report
	}
}
