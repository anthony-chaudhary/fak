package ggufload

import (
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Default memory limits and pacing thresholds for GGUF model loading.
const (
	// DefaultMemoryLimitFraction sets Go runtime memory limit to 85% of host physical RAM.
	DefaultMemoryLimitFraction = 0.85

	// DefaultLargeTensorThreshold triggers GC pacing after decoding any tensor >= 200 MiB.
	DefaultLargeTensorThreshold = int64(200 * 1024 * 1024)

	// DefaultTensorGCPacingInterval triggers GC pacing every 16 tensors under tight envelopes.
	DefaultTensorGCPacingInterval = 16
)

var memLimitMu sync.Mutex

// ConfigureHostMemoryLimit bounds Go runtime memory consumption by setting debug.SetMemoryLimit
// to a safe fraction of available host physical memory, preventing macOS Jetsam and Linux OOM kills.
//
// If an explicit memory limit is already active (via GOMEMLIMIT environment variable or prior
// explicit debug.SetMemoryLimit calls), ConfigureHostMemoryLimit preserves the caller's limit
// without overwriting it and reports changed=false.
func ConfigureHostMemoryLimit(fraction float64) (applied int64, changed bool) {
	memLimitMu.Lock()
	defer memLimitMu.Unlock()

	current := debug.SetMemoryLimit(-1)

	// If GOMEMLIMIT is set or an explicit memory limit (< MaxInt64) is already active, preserve it.
	if os.Getenv("GOMEMLIMIT") != "" || (current > 0 && current < math.MaxInt64) {
		return current, false
	}

	if fraction <= 0 {
		fraction = DefaultMemoryLimitFraction
	} else if fraction > 1.0 {
		fraction = 1.0
	}

	if envFrac := os.Getenv("FAK_MEMORY_LIMIT_FRACTION"); envFrac != "" {
		if f, err := strconv.ParseFloat(strings.TrimSpace(envFrac), 64); err == nil && f > 0 && f <= 1.0 {
			fraction = f
		}
	}

	total, _, known := compute.HostSystemMemoryInfo()
	if !known || total <= 0 {
		return current, false
	}

	target := int64(float64(total) * fraction)
	debug.SetMemoryLimit(target)
	return target, true
}

// ConfigureRuntimeMemoryLimit is an alias for ConfigureHostMemoryLimit.
func ConfigureRuntimeMemoryLimit(fraction float64) (applied int64, changed bool) {
	return ConfigureHostMemoryLimit(fraction)
}

// PreserveMemoryLimit captures the current runtime memory limit and returns a restore function.
func PreserveMemoryLimit() func() {
	orig := debug.SetMemoryLimit(-1)
	return func() {
		debug.SetMemoryLimit(orig)
	}
}

// IsTightMemoryEnvelope reports whether the host operates under unified or tight memory constraints
// where GGUF decompression slices can quickly trigger kernel process termination.
func IsTightMemoryEnvelope() bool {
	total, free, known := compute.HostSystemMemoryInfo()
	if !known || total <= 0 {
		return false
	}
	// Apple Silicon with unified memory or hosts with <= 36 GiB RAM.
	if runtime.GOOS == "darwin" || total <= 36*1024*1024*1024 {
		return true
	}
	// Low free headroom (< 16 GiB).
	if free > 0 && free < 16*1024*1024*1024 {
		return true
	}
	return false
}

// GCPacerOption configures a GCPacer.
type GCPacerOption func(*GCPacer)

// WithGCPacingEnabled enables or disables GC pacing.
func WithGCPacingEnabled(enabled bool) GCPacerOption {
	return func(p *GCPacer) { p.enabled = enabled }
}

// WithLargeTensorThreshold sets the byte size threshold that triggers synchronous GC.
func WithLargeTensorThreshold(bytes int64) GCPacerOption {
	return func(p *GCPacer) { p.largeTensorThreshold = bytes }
}

// WithPacingInterval sets the tensor count interval that triggers synchronous GC.
func WithPacingInterval(interval int) GCPacerOption {
	return func(p *GCPacer) { p.pacingInterval = interval }
}

// GCPacer provides cooperative garbage collection pacing during high-allocation model loads.
type GCPacer struct {
	mu                   sync.Mutex
	enabled              bool
	largeTensorThreshold int64
	pacingInterval       int
	tensorCount          int
	gcCount              int
}

// NewGCPacer constructs a GCPacer initialized with default thresholds and environment overrides.
func NewGCPacer(opts ...GCPacerOption) *GCPacer {
	enabled := IsTightMemoryEnvelope()

	if env := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_GC_PACING"))); env != "" {
		switch env {
		case "1", "true", "yes", "on":
			enabled = true
		case "0", "false", "no", "off":
			enabled = false
		case "auto":
			enabled = IsTightMemoryEnvelope()
		}
	}

	p := &GCPacer{
		enabled:              enabled,
		largeTensorThreshold: DefaultLargeTensorThreshold,
		pacingInterval:       DefaultTensorGCPacingInterval,
	}

	if envMB := os.Getenv("FAK_GC_PACING_MB"); envMB != "" {
		if mb, err := strconv.ParseInt(strings.TrimSpace(envMB), 10, 64); err == nil && mb > 0 {
			p.largeTensorThreshold = mb * 1024 * 1024
		}
	}
	if envInterval := os.Getenv("FAK_GC_PACING_INTERVAL"); envInterval != "" {
		if interval, err := strconv.Atoi(strings.TrimSpace(envInterval)); err == nil && interval > 0 {
			p.pacingInterval = interval
		}
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Pace records tensor progress and invokes runtime.GC() when memory thresholds are met.
// Returns true if a garbage collection was executed.
func (p *GCPacer) Pace(tensorName string, tensorBytes int64) bool {
	if p == nil {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.enabled {
		return false
	}

	p.tensorCount++
	shouldGC := false

	if p.largeTensorThreshold > 0 && tensorBytes >= p.largeTensorThreshold {
		shouldGC = true
	} else if p.pacingInterval > 0 && p.tensorCount%p.pacingInterval == 0 {
		shouldGC = true
	}

	if shouldGC {
		runtime.GC()
		p.gcCount++
		return true
	}
	return false
}

// GCCount returns the number of GC invocations triggered by this pacer.
func (p *GCPacer) GCCount() int {
	if p == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.gcCount
}

// Reset clears the pacer tensor and GC counters.
func (p *GCPacer) Reset() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tensorCount = 0
	p.gcCount = 0
}

// LoadMemoryGovernor coordinates runtime memory limits, temporary GOGC pacing, and GC synchronization
// during the lifecycle of a large GGUF model load.
type LoadMemoryGovernor struct {
	origLimit     int64
	origGCPercent int
	limitChanged  bool
	pacer         *GCPacer
}

// StartLoadMemoryGovernor initializes memory limits and temporary GC percent for model loading.
// Caller should defer gov.Close() to restore previous runtime settings.
func StartLoadMemoryGovernor(fraction float64, tempGCPercent int, opts ...GCPacerOption) *LoadMemoryGovernor {
	origLimit := debug.SetMemoryLimit(-1)
	origGC := -1

	if tempGCPercent > 0 {
		origGC = debug.SetGCPercent(tempGCPercent)
	} else {
		origGC = debug.SetGCPercent(-1)
	}

	_, changed := ConfigureHostMemoryLimit(fraction)
	pacer := NewGCPacer(opts...)

	return &LoadMemoryGovernor{
		origLimit:     origLimit,
		origGCPercent: origGC,
		limitChanged:  changed,
		pacer:         pacer,
	}
}

// Pace delegates tensor progress checking to the internal GCPacer.
func (g *LoadMemoryGovernor) Pace(tensorName string, tensorBytes int64) bool {
	if g == nil || g.pacer == nil {
		return false
	}
	return g.pacer.Pace(tensorName, tensorBytes)
}

// Pacer returns the underlying GCPacer.
func (g *LoadMemoryGovernor) Pacer() *GCPacer {
	if g == nil {
		return nil
	}
	return g.pacer
}

// Close restores the previous memory limit and GOGC percent, performing a final GC.
func (g *LoadMemoryGovernor) Close() {
	if g == nil {
		return
	}
	debug.SetMemoryLimit(g.origLimit)
	if g.origGCPercent > 0 {
		debug.SetGCPercent(g.origGCPercent)
	}
	runtime.GC()
}
