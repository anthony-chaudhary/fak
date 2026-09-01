package workerworktree

import (
	"strings"
	"time"
)

const (
	landProgressSchema = "fak-worker-land-progress/1"
	landCostSchema     = "fak-worker-land-cost/1"
)

// LandProgressEvent is one bounded state transition from the worker-land
// state machine. Events describe phases, never individual files.
type LandProgressEvent struct {
	Schema         string `json:"schema"`
	Phase          string `json:"phase"`
	Status         string `json:"status"`
	Attempt        int    `json:"attempt,omitempty"`
	LandElapsedMS  int64  `json:"land_elapsed_ms"`
	PhaseElapsedMS int64  `json:"phase_elapsed_ms,omitempty"`
	ScannedFiles   int    `json:"scanned_files,omitempty"`
	ScannedBytes   int64  `json:"scanned_bytes,omitempty"`
}

// LandPhaseCost records one completed phase. Attempt distinguishes the bounded
// isolated-index CAS retries without expanding progress to per-file events.
type LandPhaseCost struct {
	Phase     string `json:"phase"`
	Attempt   int    `json:"attempt,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms"`
}

// PatchScopeFiles counts non-empty paths emitted by git diff --name-only for
// the captured land patch. It is patch metadata, not whole-tree scan I/O.
type PatchScopeFiles int

// PatchScopeBytes is the byte length of the captured git diff used by Land.
// It is patch metadata, not whole-tree scan I/O.
type PatchScopeBytes int64

// LandCostReceipt is the terminal cost attribution for one Land call.
// ResourceState explains whether this platform exposed process CPU and peak RSS.
type LandCostReceipt struct {
	Schema          string          `json:"schema"`
	WallTimeMS      int64           `json:"wall_time_ms"`
	CPUTimeMS       *int64          `json:"cpu_time_ms,omitempty"`
	PeakRSSBytes    *int64          `json:"peak_rss_bytes,omitempty"`
	ResourceState   string          `json:"resource_state"`
	ResourceReason  string          `json:"resource_reason,omitempty"`
	PatchScopeFiles PatchScopeFiles `json:"patch_scope_files"`
	PatchScopeBytes PatchScopeBytes `json:"patch_scope_bytes"`
	CacheState      string          `json:"cache_state"`
	Reused          bool            `json:"reused"`
	SlowestPhase    string          `json:"slowest_phase,omitempty"`
	SlowestPhaseMS  int64           `json:"slowest_phase_ms"`
	PhaseTotalMS    int64           `json:"phase_total_ms"`
	UnattributedMS  int64           `json:"unattributed_ms"`
	Phases          []LandPhaseCost `json:"phases"`
}

type landResourceSample struct {
	cpuTime      time.Duration
	peakRSSBytes int64
	cpuAvailable bool
	rssAvailable bool
	reason       string
}

type landPhase struct {
	name    string
	attempt int
	started time.Time
}

type landProgressTracker struct {
	now             func() time.Time
	resources       func() landResourceSample
	emit            func(LandProgressEvent)
	started         time.Time
	resourceStart   landResourceSample
	phases          []LandPhaseCost
	patchScopeFiles int
	patchScopeBytes int64
	cacheState      string
	reused          bool
}

func newLandProgressTracker(cfg landConfig) *landProgressTracker {
	now := cfg.now
	if now == nil {
		now = time.Now
	}
	resources := cfg.resources
	if resources == nil {
		resources = currentLandResources
	}
	t := &landProgressTracker{
		now:        now,
		resources:  resources,
		emit:       cfg.progress,
		started:    now(),
		cacheState: "not-built",
		phases:     []LandPhaseCost{},
	}
	t.resourceStart = resources()
	return t
}

func (t *landProgressTracker) start(name string, attempt int) landPhase {
	p := landPhase{name: name, attempt: attempt, started: t.now()}
	t.emitEvent(LandProgressEvent{
		Schema: landProgressSchema, Phase: name, Status: "started", Attempt: attempt,
		LandElapsedMS: elapsedMillis(t.started, p.started),
	})
	return p
}

func (t *landProgressTracker) complete(p landPhase) {
	finished := t.now()
	d := elapsedMillis(p.started, finished)
	t.phases = append(t.phases, LandPhaseCost{Phase: p.name, Attempt: p.attempt, ElapsedMS: d})
	t.emitEvent(LandProgressEvent{
		Schema: landProgressSchema, Phase: p.name, Status: "completed", Attempt: p.attempt,
		LandElapsedMS: elapsedMillis(t.started, finished), PhaseElapsedMS: d,
		ScannedFiles: t.patchScopeFiles, ScannedBytes: t.patchScopeBytes,
	})
}

func (t *landProgressTracker) emitEvent(event LandProgressEvent) {
	if t.emit != nil {
		t.emit(event)
	}
}

func (t *landProgressTracker) setPatchScope(files int, bytes int64) {
	t.patchScopeFiles = files
	t.patchScopeBytes = bytes
}

func (t *landProgressTracker) setCache(state string, reused bool) {
	t.cacheState = state
	t.reused = reused
}

func (t *landProgressTracker) receipt() *LandCostReceipt {
	finished := t.now()
	endResources := t.resources()
	wall := elapsedMillis(t.started, finished)
	receipt := &LandCostReceipt{
		Schema: landCostSchema, WallTimeMS: wall,
		ResourceState: "unavailable", PatchScopeFiles: PatchScopeFiles(t.patchScopeFiles), PatchScopeBytes: PatchScopeBytes(t.patchScopeBytes),
		CacheState: t.cacheState, Reused: t.reused, Phases: append([]LandPhaseCost(nil), t.phases...),
	}
	cpuAvailable := t.resourceStart.cpuAvailable && endResources.cpuAvailable
	if cpuAvailable {
		cpuMS := (endResources.cpuTime - t.resourceStart.cpuTime).Milliseconds()
		receipt.CPUTimeMS = &cpuMS
	}
	if endResources.rssAvailable {
		peakRSS := endResources.peakRSSBytes
		receipt.PeakRSSBytes = &peakRSS
	}
	switch {
	case cpuAvailable && endResources.rssAvailable:
		receipt.ResourceState = "available"
	case cpuAvailable || endResources.rssAvailable:
		receipt.ResourceState = "partial"
	default:
		receipt.ResourceState = "unavailable"
	}
	if endResources.reason != "" {
		receipt.ResourceReason = endResources.reason
	} else if t.resourceStart.reason != "" {
		receipt.ResourceReason = t.resourceStart.reason
	}
	for _, phase := range receipt.Phases {
		receipt.PhaseTotalMS += phase.ElapsedMS
		if phase.ElapsedMS > receipt.SlowestPhaseMS || receipt.SlowestPhase == "" {
			receipt.SlowestPhase = phase.Phase
			receipt.SlowestPhaseMS = phase.ElapsedMS
		}
	}
	if wall > receipt.PhaseTotalMS {
		receipt.UnattributedMS = wall - receipt.PhaseTotalMS
	}
	return receipt
}

func elapsedMillis(start, end time.Time) int64 {
	if end.Before(start) {
		return 0
	}
	return end.Sub(start).Milliseconds()
}

func countPatchScopeFiles(names string) int {
	count := 0
	for _, name := range strings.Split(names, "\n") {
		if strings.TrimSpace(name) != "" {
			count++
		}
	}
	return count
}

func beginLandPhase(t *landProgressTracker, name string, attempt int) func() {
	if t == nil {
		return func() {}
	}
	p := t.start(name, attempt)
	return func() { t.complete(p) }
}

// WithLandProgress streams bounded state-machine transitions as they happen.
func WithLandProgress(emit func(LandProgressEvent)) LandOption {
	return func(c *landConfig) { c.progress = emit }
}

// WithLandClock and withLandResourceSampler are deterministic test seams.
func WithLandClock(now func() time.Time) LandOption {
	return func(c *landConfig) { c.now = now }
}

func withLandResourceSampler(sample func() landResourceSample) LandOption {
	return func(c *landConfig) { c.resources = sample }
}
