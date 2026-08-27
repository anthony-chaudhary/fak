// Package localadmission turns measured task envelopes and live host pressure into readiness decisions.
package localadmission

import (
	"sync"
	"time"
)

type Readiness string

const (
	Ready                  Readiness = "ready"
	ReadyDegraded          Readiness = "ready_degraded"
	TemporarilyUnavailable Readiness = "temporarily_unavailable"
	Unsupported            Readiness = "unsupported"
)

type Hardware struct {
	Chip        string `json:"chip"`
	MemoryBytes int64  `json:"memory_bytes"`
	OSVersion   int    `json:"os_version"`
}
type TaskEnvelope struct {
	Task            string        `json:"task"`
	MinOS           int           `json:"min_os"`
	Chips           []string      `json:"chips"`
	PeakMemoryBytes int64         `json:"peak_memory_bytes"`
	DiskBytes       int64         `json:"disk_bytes"`
	Quality         float64       `json:"quality"`
	MinQuality      float64       `json:"min_quality"`
	TTFT            time.Duration `json:"ttft"`
	MaxTTFT         time.Duration `json:"max_ttft"`
	DecodeTPS       float64       `json:"decode_tps"`
	MinDecodeTPS    float64       `json:"min_decode_tps"`
}
type Limits struct {
	MemoryHighWater      float64       `json:"memory_high_water"`
	MaxQueue             int           `json:"max_queue"`
	MaxConcurrent        int           `json:"max_concurrent"`
	MaxForegroundLatency time.Duration `json:"max_foreground_latency"`
	RequireAC            bool          `json:"require_ac"`
	AllowLowPower        bool          `json:"allow_low_power"`
	MaxCrashes           int           `json:"max_crashes"`
	CrashWindow          time.Duration `json:"crash_window"`
}
type Signals struct {
	MemoryUsedBytes   int64         `json:"memory_used_bytes"`
	DiskFreeBytes     int64         `json:"disk_free_bytes"`
	Queue             int           `json:"queue"`
	Concurrent        int           `json:"concurrent"`
	ForegroundLatency time.Duration `json:"foreground_latency"`
	OnBattery         bool          `json:"on_battery"`
	LowPower          bool          `json:"low_power"`
	Thermal           string        `json:"thermal"`
	Cancelled         bool          `json:"cancelled"`
}
type Decision struct {
	Task          string    `json:"task"`
	Readiness     Readiness `json:"readiness"`
	Reason        string    `json:"reason"`
	Downshift     bool      `json:"downshift"`
	QueuePosition int       `json:"queue_position,omitempty"`
}
type Governor struct {
	mu      sync.Mutex
	now     func() time.Time
	crashes map[string][]time.Time
}

func New() *Governor { return &Governor{now: time.Now, crashes: map[string][]time.Time{}} }
func includes(v []string, w string) bool {
	for _, x := range v {
		if x == w {
			return true
		}
	}
	return false
}
func (g *Governor) RecordCrash(task string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.crashes[task] = append(g.crashes[task], g.now())
}
func (g *Governor) crashOpen(task string, l Limits) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	cut := g.now().Add(-l.CrashWindow)
	kept := g.crashes[task][:0]
	for _, at := range g.crashes[task] {
		if !at.Before(cut) {
			kept = append(kept, at)
		}
	}
	g.crashes[task] = kept
	return l.MaxCrashes > 0 && len(kept) >= l.MaxCrashes
}
func (g *Governor) Admit(h Hardware, t TaskEnvelope, l Limits, s Signals) Decision {
	d := Decision{Task: t.Task, Readiness: Ready, Reason: "measured_envelope_pass"}
	unsupported := func(r string) Decision { d.Readiness = Unsupported; d.Reason = r; return d }
	unavailable := func(r string) Decision { d.Readiness = TemporarilyUnavailable; d.Reason = r; return d }
	degraded := func(r string) Decision { d.Readiness = ReadyDegraded; d.Reason = r; d.Downshift = true; return d }
	if !includes(t.Chips, h.Chip) {
		return unsupported("chip")
	}
	if h.OSVersion < t.MinOS {
		return unsupported("os")
	}
	if t.PeakMemoryBytes > h.MemoryBytes {
		return unsupported("memory_capacity")
	}
	if t.Quality < t.MinQuality {
		return unsupported("fixture_quality")
	}
	if t.TTFT > t.MaxTTFT {
		return unsupported("ttft")
	}
	if t.DecodeTPS < t.MinDecodeTPS {
		return unsupported("decode_rate")
	}
	if s.Cancelled {
		return unavailable("cancelled")
	}
	if g.crashOpen(t.Task, l) {
		return unavailable("crash_loop")
	}
	if s.DiskFreeBytes < t.DiskBytes {
		return unavailable("disk_reservation")
	}
	if float64(s.MemoryUsedBytes+t.PeakMemoryBytes) > float64(h.MemoryBytes)*l.MemoryHighWater {
		return unavailable("memory_high_water")
	}
	if s.Queue >= l.MaxQueue {
		return unavailable("queue_full")
	}
	if s.Concurrent >= l.MaxConcurrent {
		d = degraded("queued")
		d.QueuePosition = s.Queue + 1
		return d
	}
	if s.ForegroundLatency > l.MaxForegroundLatency {
		return unavailable("foreground_latency")
	}
	if l.RequireAC && s.OnBattery {
		return unavailable("ac_required")
	}
	if s.LowPower && !l.AllowLowPower {
		return unavailable("low_power_mode")
	}
	if s.Thermal == "critical" {
		return unavailable("thermal_critical")
	}
	if s.Thermal == "serious" || s.OnBattery || s.LowPower {
		return degraded("resource_downshift")
	}
	return d
}
