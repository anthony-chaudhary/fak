package sessionmine

import (
	"context"
	"testing"
)

func TestRefreshBenchmarkStructuralSentinel(t *testing.T) {
	r, err := BenchmarkRefresh(context.Background(), RefreshBenchmarkOptions{Sizes: []int{20}, Repetitions: 2, WorkDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schema != RefreshBenchmarkSchema || len(r.Scales) != 1 {
		t.Fatalf("report=%+v", r)
	}
	s := r.Scales[0]
	if len(s.Phases) != 3 || s.Phases[0].Name != "cold" || s.Phases[1].Reused != 20 || s.Phases[1].Rebuilt != 0 || s.Phases[2].Rebuilt != 1 {
		t.Fatalf("scale=%+v", s)
	}
	if s.IndexBytes == 0 || s.BytesPerSession <= 0 || s.PeakHeapBytes == 0 {
		t.Fatalf("missing resource evidence: %+v", s)
	}
}
func TestRefreshBenchmarkRejectsUnboundedScale(t *testing.T) {
	if _, err := BenchmarkRefresh(context.Background(), RefreshBenchmarkOptions{Sizes: []int{100001}, Repetitions: 1}); err == nil {
		t.Fatal("expected bound refusal")
	}
}
