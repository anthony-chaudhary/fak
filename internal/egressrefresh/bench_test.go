package egressrefresh

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

func seedBenchDir(b *testing.B, url, seedText string) string {
	b.Helper()
	dir := b.TempDir()
	src := egresslist.Source{
		Name:          "testlist",
		URL:           url,
		Format:        "hosts",
		Description:   "refresh benchmark fixture",
		LastRefreshed: "2026-01-01T00:00:00Z",
	}
	list := egresslist.NewBuilder().AddFilterText(src.Name, seedText).Build()
	block, allow := list.Counts()
	artifact := egresslist.RenderArtifact(src, list)
	src.SHA256 = egresslist.Checksum(artifact)
	src.Rules = block + allow

	man := egresslist.Manifest{Version: egresslist.ManifestVersion, Lists: []egresslist.Source{src}}
	out, err := man.Render()
	if err != nil {
		b.Fatalf("render seed manifest: %v", err)
	}
	if err := os.WriteFile(ManifestPath(dir), out, 0o644); err != nil {
		b.Fatalf("write seed manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "testlist.txt"), []byte(artifact), 0o644); err != nil {
		b.Fatalf("write seed artifact: %v", err)
	}
	return dir
}

// BenchmarkEgressRefresh exercises the egress list refresh pipeline under a simulated upstream fetch.
func BenchmarkEgressRefresh(b *testing.B) {
	const benchText = "0.0.0.0 old-a.example\n0.0.0.0 old-b.example\n"
	const upstream = "0.0.0.0 new-evil.example\n0.0.0.0 new-tracker.example\n||anchor.example^\n"
	dir := seedBenchDir(b, "https://upstream.invalid/list.txt", benchText)
	fetcher := staticFetcher{body: upstream}
	opts := Options{
		Dir:     dir,
		Fetcher: fetcher,
		DryRun:  true,
		Now:     time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC),
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := Refresh(ctx, opts)
		if err != nil {
			b.Fatalf("Refresh failed: %v", err)
		}
		if len(results) == 0 || results[0].Status != StatusUpdated {
			b.Fatalf("unexpected benchmark results: %+v", results)
		}
	}
}

// TestBenchmarkEgressRefreshRuns verifies that the benchmark logic executes without error.
func TestBenchmarkEgressRefreshRuns(t *testing.T) {
	const text = "0.0.0.0 old-a.example\n"
	dir := seedDir(t, "https://upstream.invalid/list.txt", text)
	opts := Options{
		Dir:     dir,
		Fetcher: staticFetcher{body: "0.0.0.0 new-evil.example\n"},
		DryRun:  true,
	}
	results, err := Refresh(context.Background(), opts)
	if err != nil {
		t.Fatalf("Refresh failed: %v", err)
	}
	if len(results) == 0 || results[0].Status != StatusUpdated {
		t.Fatalf("unexpected results: %+v", results)
	}
}
