package runtimeobs

import (
	"runtime"
	"slices"
	"testing"
)

func TestReadFillsClosedFieldSet(t *testing.T) {
	runtime.GC()
	fields := Read().Fields()
	got := make([]string, len(fields))
	for i, f := range fields {
		got[i] = f.Name
	}
	want := []string{
		"heap_live_bytes", "heap_goal_bytes", "memlimit_bytes", "gogc",
		"gc_cycles_total", "gc_pause_p99_seconds", "total_mapped_bytes",
		"released_bytes", "stack_bytes", "goroutines",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("field set = %v, want %v", got, want)
	}
	for _, f := range fields {
		if !f.OK {
			t.Errorf("field %s unavailable on the live runtime", f.Name)
		}
	}
}

func TestReadReflectsLiveRuntime(t *testing.T) {
	runtime.GC()
	s := Read()
	if s.GCCyclesTotal.V < 1 {
		t.Errorf("gc_cycles_total = %v after runtime.GC, want >= 1", s.GCCyclesTotal.V)
	}
	if s.Goroutines.V < 1 {
		t.Errorf("goroutines = %v, want >= 1", s.Goroutines.V)
	}
	if s.HeapGoalBytes.V <= 0 || s.TotalMappedBytes.V <= 0 {
		t.Errorf("heap goal %v / total mapped %v, want > 0", s.HeapGoalBytes.V, s.TotalMappedBytes.V)
	}
}
