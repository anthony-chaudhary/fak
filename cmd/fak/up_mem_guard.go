package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// memGuardGovernor stops a resident turnkey server once its own memory footprint
// has stayed above a configured ceiling for a sustained window, so the OS
// memory-pressure cascade (compressor saturation + swap exhaustion) cannot be
// driven by an unbounded-growth resident process.
//
// Motivating defect (macOS M3 Pro, 36 GiB): `fak up / fak-native up --headless
// --model 27B --context 20480` grew to a ~27 GiB physical footprint (peak
// 31 GiB), leaving 20 GiB of compressor and 8.3 GiB of swap in use with under
// 200 MiB free physical RAM. launchd's KeepAlive=true restarts the process on
// exit, so a bounded stop RELEASES the residency and lets a fresh (low-water)
// process take over instead of pinning the machine in swap.
//
// The guard is a full stop through the SAME graceful path as the GPU idle-exit
// governor (#13135): residency and GPU lease are released together. It is
// operator-visible (a stderr line naming the ceiling and the observed peak) and
// reversible (--max-rss 0 disables it, preserving the historical unbounded
// holder).
type memGuardGovernor struct {
	mu       sync.Mutex
	limit    uint64
	interval time.Duration
	sustain  time.Duration
	closed   bool
	doneCh   chan struct{}
	shutdown func()
	logf     func(format string, args ...any)
	// overSince is the zero time while under the ceiling; it records when the
	// process first went over so the guard fires only after a SUSTAINED breach.
	overSince time.Time
	peak      uint64

	// readRSS and now are seams; nil uses the real process RSS and clock.
	readRSS func() uint64
	now     func() time.Time
}

// defaultMemGuardInterval is how often the guard samples process RSS. Slow
// enough to be negligible against a multi-minute model load, fast enough to
// catch runaway growth well before swap exhaustion.
const defaultMemGuardInterval = 30 * time.Second

// defaultMemGuardSustain is how long the footprint must stay above the ceiling
// before the guard stops the server. It absorbs the model load's own transient
// high-water (weights + KV pool mapped at once) without tripping on it.
const defaultMemGuardSustain = 2 * time.Minute

// newMemGuardGovernor builds a memory-ceiling governor. A zero limit or nil
// shutdown disables it and returns nil, preserving the historical unbounded
// holder byte-for-byte.
func newMemGuardGovernor(limit uint64, interval, sustain time.Duration, readRSS func() uint64, shutdown func(), logf func(format string, args ...any)) *memGuardGovernor {
	if limit == 0 || shutdown == nil {
		return nil
	}
	if interval <= 0 {
		interval = defaultMemGuardInterval
	}
	if sustain <= 0 {
		sustain = defaultMemGuardSustain
	}
	if readRSS == nil {
		readRSS = processRSSBytes
	}
	if logf == nil {
		logf = func(format string, args ...any) {}
	}
	g := &memGuardGovernor{
		limit:    limit,
		interval: interval,
		sustain:  sustain,
		doneCh:   make(chan struct{}),
		readRSS:  readRSS,
		now:      time.Now,
		shutdown: shutdown,
		logf:     logf,
	}
	go g.run()
	return g
}

// run samples process RSS on the governor's interval and stops the server after
// a sustained breach. It exits when the governor is closed.
func (g *memGuardGovernor) run() {
	ticker := time.NewTicker(g.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			g.sample()
		case <-g.doneCh:
			return
		}
	}
}

// sample takes one RSS reading and applies the sustained-breach rule.
func (g *memGuardGovernor) sample() {
	if g == nil {
		return
	}
	rss := g.readRSS()
	g.mu.Lock()
	if g.closed {
		g.mu.Unlock()
		return
	}
	if rss > g.peak {
		g.peak = rss
	}
	now := g.now()
	if rss <= g.limit {
		g.overSince = time.Time{}
		g.mu.Unlock()
		return
	}
	if g.overSince.IsZero() {
		g.overSince = now
		g.mu.Unlock()
		return
	}
	if now.Sub(g.overSince) < g.sustain {
		g.mu.Unlock()
		return
	}
	g.closed = true
	overSince := g.overSince
	peak := g.peak
	limit := g.limit
	shutdown := g.shutdown
	logf := g.logf
	g.mu.Unlock()
	close(g.doneCh)
	logf("fak up: process RSS stayed above the %s ceiling for %s (observed peak %s); stopping the resident server so its model residency and GPU lease are released and a fresh process restarts (--max-rss 0 disables this guard)",
		formatBytes(limit), now.Sub(overSince).Round(time.Second), formatBytes(peak))
	shutdown()
}

// close stops the governor without triggering a shutdown (an operator-initiated
// or signal-initiated stop is already in progress).
func (g *memGuardGovernor) close() {
	if g == nil {
		return
	}
	g.mu.Lock()
	if !g.closed {
		g.closed = true
		close(g.doneCh)
	}
	g.mu.Unlock()
}

// processRSSBytes reads this process's current resident set size. It prefers the
// OS-maintained current RSS (darwin mach task info via the platform helper);
// when that is unavailable it falls back to the Go runtime's total bytes
// obtained from the OS, which still bounds an unbounded-growth process because
// this guard only ever compares the value to a ceiling and stops on a sustained
// breach.
func processRSSBytes() uint64 {
	if rss := platformCurrentRSS(); rss > 0 {
		return rss
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.Sys
}

// formatBytes renders a byte count compactly for the operator-visible log line.
func formatBytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := uint64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	units := "KMGTPE"
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), units[exp])
}
