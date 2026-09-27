package taskmgr

import (
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// ResourceSample field names, as they appear in its JSON and in the
// Unavailable lists of ResourceSample and ResourceDelta.
const (
	resourceCPUSeconds     = "cpu_s"
	resourceHeapAllocBytes = "heap_alloc_bytes"
	resourceHeapInuseBytes = "heap_inuse_bytes"
	resourceHeapSysBytes   = "heap_sys_bytes"
	resourceSysBytes       = "sys_bytes"
)

// runtimeCollector is the default sampler's availability-checked
// runtime/metrics vector (#10182). It replaces runtime.ReadMemStats, which
// stops the world on every task and step transition.
var runtimeCollector = sync.OnceValue(func() *runtimeobs.Collector { return runtimeobs.New() })

// SampleRuntime is the default Sampler: one runtime/metrics read, mapped onto
// the MemStats-shaped fields this package has always reported. A field the
// running toolchain cannot supply is named in Unavailable instead of being
// reported as an observed zero.
func SampleRuntime(processStart, now time.Time) ResourceSample {
	return resourceSampleFromReceipt(runtimeCollector().Sample(), processStart, now)
}

// resourceSampleFromReceipt maps the runtime/metrics memory classes onto the
// MemStats fields they equal (runtime/mstats.go, Go 1.26):
//
//	HeapAlloc = heap/objects
//	HeapInuse = heap/objects + heap/unused
//	HeapSys   = heap/objects + heap/unused + heap/free + heap/released
//	Sys       = classes/total
func resourceSampleFromReceipt(r runtimeobs.Receipt, processStart, now time.Time) ResourceSample {
	s := ResourceSample{
		TSUnixNano:  now.UnixNano(),
		WallSeconds: seconds(now.Sub(processStart)),
		Goroutines:  runtime.NumGoroutine(),
	}
	var missing []string
	if cpu := r.CPU.TotalSeconds; cpu.OK {
		s.CPUSeconds = cpu.V
	} else {
		missing = append(missing, resourceCPUSeconds)
	}
	m := r.Memory
	if m.HeapObjectsBytes.OK {
		s.HeapAllocBytes = m.HeapObjectsBytes.V
	} else {
		missing = append(missing, resourceHeapAllocBytes)
	}
	inuse, inuseOK := sumU64(m.HeapObjectsBytes, m.HeapUnusedBytes)
	if inuseOK {
		s.HeapInuseBytes = inuse
	} else {
		missing = append(missing, resourceHeapInuseBytes)
	}
	if sys, ok := sumU64(m.HeapObjectsBytes, m.HeapUnusedBytes, m.HeapFreeBytes, m.HeapReleasedBytes); ok {
		s.HeapSysBytes = sys
	} else {
		missing = append(missing, resourceHeapSysBytes)
	}
	if m.TotalMappedBytes.OK {
		s.SysBytes = m.TotalMappedBytes.V
	} else {
		missing = append(missing, resourceSysBytes)
	}
	s.Unavailable = missing
	return s
}

// sumU64 adds observed values; any unavailable operand makes the sum unavailable.
func sumU64(vs ...runtimeobs.U64) (uint64, bool) {
	var total uint64
	for _, v := range vs {
		if !v.OK {
			return 0, false
		}
		total += v.V
	}
	return total, true
}

// withoutUnavailable names every field unavailable on either side of a delta
// and zeroes its difference: subtracting an unobserved zero from an observed
// value is not an interval.
func withoutUnavailable(d ResourceDelta, start, current ResourceSample) ResourceDelta {
	if len(start.Unavailable) == 0 && len(current.Unavailable) == 0 {
		return d
	}
	seen := map[string]bool{}
	for _, f := range append(append([]string(nil), start.Unavailable...), current.Unavailable...) {
		if seen[f] {
			continue
		}
		seen[f] = true
		d.Unavailable = append(d.Unavailable, f)
		switch f {
		case resourceCPUSeconds:
			d.CPUSeconds = 0
		case resourceHeapAllocBytes:
			d.HeapAllocBytes = 0
		case resourceHeapInuseBytes:
			d.HeapInuseBytes = 0
		case resourceHeapSysBytes:
			d.HeapSysBytes = 0
		case resourceSysBytes:
			d.SysBytes = 0
		}
	}
	sort.Strings(d.Unavailable)
	return d
}
