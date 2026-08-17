package main

import (
	"runtime"
	"sync"
	"time"
)

// resourceUsage describes process/runtime cost over one benchmark run. CPUPercent
// is normalized to the whole machine (100% means every logical CPU was busy),
// while CPUCoreEquivalent is the average number of fully occupied logical CPUs.
type resourceUsage struct {
	LogicalCPUs       int     `json:"logical_cpus"`
	WallMS            float64 `json:"wall_ms"`
	CPUAvailable      bool    `json:"cpu_available"`
	CPUUserMS         float64 `json:"cpu_user_ms,omitempty"`
	CPUSystemMS       float64 `json:"cpu_system_ms,omitempty"`
	CPUTotalMS        float64 `json:"cpu_total_ms,omitempty"`
	CPUPercent        float64 `json:"cpu_percent,omitempty"`
	CPUCoreEquivalent float64 `json:"cpu_core_equivalent,omitempty"`
	HeapStartBytes    uint64  `json:"heap_start_bytes"`
	HeapEndBytes      uint64  `json:"heap_end_bytes"`
	HeapPeakBytes     uint64  `json:"heap_peak_bytes"`
	HeapDeltaBytes    int64   `json:"heap_delta_bytes"`
	SysStartBytes     uint64  `json:"sys_start_bytes"`
	SysEndBytes       uint64  `json:"sys_end_bytes"`
	SysPeakBytes      uint64  `json:"sys_peak_bytes"`
	TotalAllocBytes   uint64  `json:"total_alloc_bytes"`
	Mallocs           uint64  `json:"mallocs"`
	Frees             uint64  `json:"frees"`
	GCCycles          uint32  `json:"gc_cycles"`
	GCPauseMS         float64 `json:"gc_pause_ms"`
	GoroutinesStart   int     `json:"goroutines_start"`
	GoroutinesEnd     int     `json:"goroutines_end"`
	GoroutinesPeak    int     `json:"goroutines_peak"`
	Samples           int     `json:"samples"`
	SampleIntervalMS  float64 `json:"sample_interval_ms"`
}

type resourceMonitor struct {
	started  time.Time
	cpu      processCPU
	before   runtime.MemStats
	startG   int
	stop     chan struct{}
	done     chan struct{}
	stopOnce sync.Once

	mu       sync.Mutex
	heapPeak uint64
	sysPeak  uint64
	goPeak   int
	samples  int
}

const resourceSampleInterval = 100 * time.Millisecond

func startResourceMonitor() *resourceMonitor {
	m := &resourceMonitor{
		started: time.Now(),
		cpu:     readProcessCPU(),
		startG:  runtime.NumGoroutine(),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
	runtime.ReadMemStats(&m.before)
	m.heapPeak = m.before.HeapAlloc
	m.sysPeak = m.before.Sys
	m.goPeak = m.startG
	m.sample()
	go m.loop()
	return m
}

func (m *resourceMonitor) loop() {
	defer close(m.done)
	ticker := time.NewTicker(resourceSampleInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			m.sample()
		case <-m.stop:
			return
		}
	}
}

func (m *resourceMonitor) sample() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	g := runtime.NumGoroutine()
	m.mu.Lock()
	if ms.HeapAlloc > m.heapPeak {
		m.heapPeak = ms.HeapAlloc
	}
	if ms.Sys > m.sysPeak {
		m.sysPeak = ms.Sys
	}
	if g > m.goPeak {
		m.goPeak = g
	}
	m.samples++
	m.mu.Unlock()
}

func (m *resourceMonitor) close() {
	m.stopOnce.Do(func() { close(m.stop) })
}

func (m *resourceMonitor) finish() resourceUsage {
	m.close()
	<-m.done
	m.sample()
	wall := time.Since(m.started)
	cpuAfter := readProcessCPU()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	endG := runtime.NumGoroutine()

	m.mu.Lock()
	heapPeak, sysPeak, goPeak, samples := m.heapPeak, m.sysPeak, m.goPeak, m.samples
	m.mu.Unlock()

	u := resourceUsage{
		LogicalCPUs:      runtime.NumCPU(),
		WallMS:           milliseconds(wall),
		HeapStartBytes:   m.before.HeapAlloc,
		HeapEndBytes:     after.HeapAlloc,
		HeapPeakBytes:    heapPeak,
		HeapDeltaBytes:   int64(after.HeapAlloc) - int64(m.before.HeapAlloc),
		SysStartBytes:    m.before.Sys,
		SysEndBytes:      after.Sys,
		SysPeakBytes:     sysPeak,
		TotalAllocBytes:  after.TotalAlloc - m.before.TotalAlloc,
		Mallocs:          after.Mallocs - m.before.Mallocs,
		Frees:            after.Frees - m.before.Frees,
		GCCycles:         after.NumGC - m.before.NumGC,
		GCPauseMS:        float64(after.PauseTotalNs-m.before.PauseTotalNs) / float64(time.Millisecond),
		GoroutinesStart:  m.startG,
		GoroutinesEnd:    endG,
		GoroutinesPeak:   goPeak,
		Samples:          samples,
		SampleIntervalMS: milliseconds(resourceSampleInterval),
	}
	if m.cpu.ok && cpuAfter.ok {
		user := cpuAfter.user - m.cpu.user
		system := cpuAfter.system - m.cpu.system
		total := user + system
		u.CPUAvailable = true
		u.CPUUserMS = milliseconds(user)
		u.CPUSystemMS = milliseconds(system)
		u.CPUTotalMS = milliseconds(total)
		if wall > 0 {
			u.CPUCoreEquivalent = float64(total) / float64(wall)
			u.CPUPercent = 100 * u.CPUCoreEquivalent / float64(u.LogicalCPUs)
		}
	}
	return u
}

func milliseconds(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
