// Package runtimeobs is the stable public view of the engine's Go runtime
// reader: a fixed, closed set of memory/GC/scheduler gauges sampled from
// runtime/metrics. Each field carries an OK bit, so a metric the running
// toolchain does not export is reported as unavailable, never as zero.
package runtimeobs

import (
	"math"
	"sync"

	inner "github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// Gauge is one observed value or an explicit unavailable marker.
type Gauge struct {
	V  float64
	OK bool
}

// Snapshot is one sample of the closed gauge set.
type Snapshot struct {
	HeapLiveBytes    Gauge
	HeapGoalBytes    Gauge
	MemoryLimitBytes Gauge
	// GOGCPercent is -1 when GOGC=off.
	GOGCPercent       Gauge
	GCCyclesTotal     Gauge
	GCPauseP99Seconds Gauge
	TotalMappedBytes  Gauge
	ReleasedBytes     Gauge
	// StackBytes is heap-allocated plus OS-allocated goroutine stack memory.
	StackBytes Gauge
	Goroutines Gauge
}

// Field is one named gauge of a Snapshot.
type Field struct {
	Name string
	Gauge
}

// Names is the closed, ordered field-name set returned by Snapshot.Fields.
// Adding or renaming a name is a contract change.
var Names = []string{
	"heap_live_bytes",
	"heap_goal_bytes",
	"memlimit_bytes",
	"gogc",
	"gc_cycles_total",
	"gc_pause_p99_seconds",
	"total_mapped_bytes",
	"released_bytes",
	"stack_bytes",
	"goroutines",
}

// Fields returns the snapshot as named gauges, in Names order.
func (s Snapshot) Fields() []Field {
	gs := []Gauge{
		s.HeapLiveBytes, s.HeapGoalBytes, s.MemoryLimitBytes, s.GOGCPercent,
		s.GCCyclesTotal, s.GCPauseP99Seconds, s.TotalMappedBytes,
		s.ReleasedBytes, s.StackBytes, s.Goroutines,
	}
	out := make([]Field, len(gs))
	for i, g := range gs {
		out[i] = Field{Name: Names[i], Gauge: g}
	}
	return out
}

var (
	once      sync.Once
	collector *inner.Collector
)

// Read samples the live Go runtime. It is safe for concurrent use.
func Read() Snapshot {
	once.Do(func() { collector = inner.New() })
	return fromReceipt(collector.Sample())
}

func fromReceipt(r inner.Receipt) Snapshot {
	s := Snapshot{
		HeapLiveBytes:    u64(r.GC.HeapLiveBytes),
		HeapGoalBytes:    u64(r.GC.HeapGoalBytes),
		MemoryLimitBytes: u64(r.GC.MemoryLimitBytes),
		GOGCPercent:      u64(r.GC.GOGCPercent),
		GCCyclesTotal:    u64(r.GC.Cycles),
		TotalMappedBytes: u64(r.Memory.TotalMappedBytes),
		ReleasedBytes:    u64(r.Memory.HeapReleasedBytes),
		Goroutines:       u64(r.Sched.Goroutines),
	}
	if r.GC.GOGCOff {
		s.GOGCPercent = Gauge{V: -1, OK: true}
	}
	if v, ok := r.GC.Pauses.Quantile(0.99); ok && !math.IsInf(v, 0) && !math.IsNaN(v) {
		s.GCPauseP99Seconds = Gauge{V: v, OK: true}
	}
	hs, os := r.Memory.HeapStacksBytes, r.Memory.OSStacksBytes
	if hs.OK && os.OK {
		s.StackBytes = Gauge{V: float64(hs.V) + float64(os.V), OK: true}
	}
	return s
}

func u64(u inner.U64) Gauge {
	if !u.OK {
		return Gauge{}
	}
	return Gauge{V: float64(u.V), OK: true}
}
