package taskmgr

import (
	"encoding/json"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// TestSampleRuntimeBracketsMemStats pins the MemStats equivalence of the
// runtime/metrics mapping: with the heap quiesced (GC off after a completed
// cycle and sweep) the cumulative heap fields only grow, so each value the
// default sampler reads must fall between two ReadMemStats readings taken
// around it. A wrong class mapping (e.g. counting stacks in HeapSys) breaks
// the bracket.
func TestSampleRuntimeBracketsMemStats(t *testing.T) {
	runtime.GC()
	old := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(old)
	start := time.Now()
	_ = SampleRuntime(start, start) // warm the collector and its allocation paths

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	s := SampleRuntime(start, time.Now())
	runtime.ReadMemStats(&after)

	if len(s.Unavailable) != 0 {
		t.Fatalf("live %s runtime left fields unavailable: %v", runtime.Version(), s.Unavailable)
	}
	for _, c := range []struct {
		name        string
		got, lo, hi uint64
	}{
		{"heap_alloc_bytes", s.HeapAllocBytes, before.HeapAlloc, after.HeapAlloc},
		{"heap_inuse_bytes", s.HeapInuseBytes, before.HeapInuse, after.HeapInuse},
		{"heap_sys_bytes", s.HeapSysBytes, before.HeapSys, after.HeapSys},
		{"sys_bytes", s.SysBytes, before.Sys, after.Sys},
	} {
		if c.got < c.lo || c.got > c.hi {
			t.Errorf("%s = %d, want within MemStats bracket [%d, %d]", c.name, c.got, c.lo, c.hi)
		}
	}
	if s.CPUSeconds <= 0 || s.Goroutines <= 0 {
		t.Fatalf("cpu_s=%v goroutines=%d, want both observed", s.CPUSeconds, s.Goroutines)
	}
}

// TestResourceSampleNamesUnavailableFields proves a metric the toolchain
// cannot supply is named, not reported as an observed zero, and that every
// field derived from it degrades with it.
func TestResourceSampleNamesUnavailableFields(t *testing.T) {
	r := runtimeobs.Receipt{}
	r.CPU.TotalSeconds = runtimeobs.F64{V: 3, OK: true}
	r.Memory.HeapUnusedBytes = runtimeobs.U64{V: 10, OK: true}
	r.Memory.HeapFreeBytes = runtimeobs.U64{V: 20, OK: true}
	r.Memory.HeapReleasedBytes = runtimeobs.U64{V: 30, OK: true}
	r.Memory.TotalMappedBytes = runtimeobs.U64{V: 1000, OK: true}
	// heap/objects is missing: HeapAlloc, HeapInuse and HeapSys all need it.
	start := time.Unix(100, 0)
	s := resourceSampleFromReceipt(r, start, start.Add(2*time.Second))
	want := []string{"heap_alloc_bytes", "heap_inuse_bytes", "heap_sys_bytes"}
	if !reflect.DeepEqual(s.Unavailable, want) {
		t.Fatalf("unavailable = %v, want %v", s.Unavailable, want)
	}
	if s.HeapAllocBytes != 0 || s.HeapInuseBytes != 0 || s.HeapSysBytes != 0 {
		t.Fatalf("unavailable fields carried values: %+v", s)
	}
	if s.CPUSeconds != 3 || s.SysBytes != 1000 || s.WallSeconds != 2 {
		t.Fatalf("available fields lost: %+v", s)
	}
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"unavailable":["heap_alloc_bytes","heap_inuse_bytes","heap_sys_bytes"]`) {
		t.Fatalf("unavailable fields not named in JSON: %s", b)
	}

	r.Memory.HeapObjectsBytes = runtimeobs.U64{V: 5, OK: true}
	full := resourceSampleFromReceipt(r, start, start)
	if len(full.Unavailable) != 0 || full.HeapAllocBytes != 5 || full.HeapInuseBytes != 15 || full.HeapSysBytes != 65 {
		t.Fatalf("class mapping = %+v, want alloc 5, inuse 15, sys 65", full)
	}
}

// TestResourceDeltaRefusesUnobservedEndpoints: a field unobserved at either
// end of a window has no interval value, even though both zeros subtract.
func TestResourceDeltaRefusesUnobservedEndpoints(t *testing.T) {
	start := ResourceSample{CPUSeconds: 1, HeapAllocBytes: 0, SysBytes: 100, Unavailable: []string{"heap_alloc_bytes"}}
	cur := ResourceSample{CPUSeconds: 4, HeapAllocBytes: 900, SysBytes: 160, Unavailable: []string{"cpu_s"}}
	d := resourceDelta(start, cur)
	if d.HeapAllocBytes != 0 || d.CPUSeconds != 0 {
		t.Fatalf("delta subtracted an unobserved endpoint: %+v", d)
	}
	if !reflect.DeepEqual(d.Unavailable, []string{"cpu_s", "heap_alloc_bytes"}) {
		t.Fatalf("delta unavailable = %v", d.Unavailable)
	}
	if d.SysBytes != 60 {
		t.Fatalf("observed delta lost: %+v", d)
	}
	if clean := resourceDelta(ResourceSample{CPUSeconds: 1}, ResourceSample{CPUSeconds: 2}); clean.Unavailable != nil || clean.CPUSeconds != 1 {
		t.Fatalf("fully observed delta = %+v", clean)
	}
}

// BenchmarkSampleRuntime is the default sampler's per-transition cost
// (#10182 1% budget); compare BenchmarkReadMemStats, the stop-the-world call
// it replaced.
func BenchmarkSampleRuntime(b *testing.B) {
	start := time.Now()
	b.ReportAllocs()
	for b.Loop() {
		_ = SampleRuntime(start, start)
	}
}

func BenchmarkReadMemStats(b *testing.B) {
	var ms runtime.MemStats
	b.ReportAllocs()
	for b.Loop() {
		runtime.ReadMemStats(&ms)
	}
}
