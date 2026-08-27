package workerworktree

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

const (
	// LandProgressSchema versions the streaming phase-boundary contract.
	LandProgressSchema = "fak-worker-land-progress/1"
	// LandCostSchema versions the terminal cost-receipt contract.
	LandCostSchema = "fak-worker-land-cost/1"
)

// LandProgressEvent is a bounded phase boundary emitted while Land is running.
// It deliberately carries counts rather than paths: progress must never disclose or
// accidentally sweep peer-owned paths from a dirty trunk.
type LandProgressEvent struct {
	Schema         string `json:"schema"`
	Phase          string `json:"phase"`
	State          string `json:"state"` // started | completed
	Outcome        string `json:"outcome,omitempty"`
	Attempt        int    `json:"attempt,omitempty"`
	ElapsedMS      int64  `json:"elapsed_ms"`
	PhaseElapsedMS int64  `json:"phase_elapsed_ms,omitempty"`
}

// LandPhaseCost is one completed, disjoint phase included in the terminal receipt.
type LandPhaseCost struct {
	Phase     string `json:"phase"`
	Outcome   string `json:"outcome"`
	Attempt   int    `json:"attempt,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms"`
}

// LandCostReceipt accounts for one Land invocation. ScannedBytes includes the
// captured exact-land patch plus bytes materialized by each whole-tree analysis;
// ScanScope names which of those sources ran. It does not claim Git's internal I/O.
// PeakRSS is process-lifetime OS peak because that is what Getrusage and
// GetProcessMemoryInfo can portably witness.
type LandCostReceipt struct {
	Schema             string          `json:"schema"`
	Outcome            string          `json:"outcome"`
	WallMS             int64           `json:"wall_ms"`
	CPUms              int64           `json:"cpu_ms"`
	CPUObserved        bool            `json:"cpu_observed"`
	CPUScope           string          `json:"cpu_scope"`
	PeakRSSBytes       uint64          `json:"peak_rss_bytes"`
	PeakRSSObserved    bool            `json:"peak_rss_observed"`
	PeakRSSScope       string          `json:"peak_rss_scope"`
	ScannedFiles       int             `json:"scanned_files"`
	ScannedBytes       int64           `json:"scanned_bytes"`
	ScanScope          string          `json:"scan_scope"`
	CacheReuse         string          `json:"cache_reuse"`
	SlowestPhase       string          `json:"slowest_phase,omitempty"`
	SlowestPhaseMS     int64           `json:"slowest_phase_ms,omitempty"`
	AccountedPhaseMS   int64           `json:"accounted_phase_ms"`
	UnattributedWallMS int64           `json:"unattributed_wall_ms"`
	Phases             []LandPhaseCost `json:"phases"`
}

// LandProgressSink receives low-volume phase boundaries. A sink cannot alter the
// land verdict; logging failure must never weaken or wedge the transaction.
type LandProgressSink func(LandProgressEvent)

// WithLandProgress streams phase boundaries for this Land invocation. The terminal
// cost receipt is returned on Result regardless of whether a sink is installed.
func WithLandProgress(sink LandProgressSink) LandOption {
	return func(c *landConfig) { c.progressSink = sink }
}

type landResourceSample struct {
	cpu             time.Duration
	cpuObserved     bool
	peakRSSBytes    uint64
	peakRSSObserved bool
}

type landResourceProbe interface{ stop() landResourceSample }

type runtimeLandProbe struct {
	sampler    *harnessres.Sampler
	initialCPU time.Duration
	initialOK  bool
}

func newRuntimeLandProbe() landResourceProbe {
	s := harnessres.New()
	// Start samples immediately. The long interval avoids observer churn; Stop takes
	// a final sample, so short and long lands both receive two resource readings.
	s.Start(time.Hour)
	initial := s.Snapshot().Kernel
	return &runtimeLandProbe{
		sampler:    s,
		initialCPU: initial.CPUUser + initial.CPUSys,
		initialOK:  initial.HaveCPU,
	}
}

func (p *runtimeLandProbe) stop() landResourceSample {
	if p == nil || p.sampler == nil {
		return landResourceSample{}
	}
	snap := p.sampler.Stop().Kernel
	cpu := snap.CPUUser + snap.CPUSys
	if p.initialOK && snap.HaveCPU && cpu >= p.initialCPU {
		cpu -= p.initialCPU
	} else {
		cpu = 0
	}
	return landResourceSample{
		cpu:             cpu,
		cpuObserved:     p.initialOK && snap.HaveCPU,
		peakRSSBytes:    snap.PeakRSSBytes,
		peakRSSObserved: snap.HavePeakRSS,
	}
}

type landRecorder struct {
	now            func() time.Time
	sink           LandProgressSink
	started        time.Time
	probe          landResourceProbe
	phases         []LandPhaseCost
	scannedFiles   int
	scannedBytes   int64
	wholeTreeScans int
	finished       bool
}

type activeLandPhase struct {
	recorder *landRecorder
	name     string
	attempt  int
	started  time.Time
	finished bool
}

func newLandRecorder(cfg landConfig) *landRecorder {
	now := cfg.progressNow
	if now == nil {
		now = time.Now
	}
	probeFactory := cfg.progressProbe
	if probeFactory == nil {
		probeFactory = newRuntimeLandProbe
	}
	return &landRecorder{now: now, sink: cfg.progressSink, started: now(), probe: probeFactory()}
}

func (r *landRecorder) begin(name string, attempt int) *activeLandPhase {
	if r == nil {
		return &activeLandPhase{}
	}
	now := r.now()
	r.emit(LandProgressEvent{
		Schema: LandProgressSchema, Phase: name, State: "started", Attempt: attempt,
		ElapsedMS: nonnegativeDuration(now.Sub(r.started)).Milliseconds(),
	})
	return &activeLandPhase{recorder: r, name: name, attempt: attempt, started: now}
}

func (p *activeLandPhase) complete(outcome string) {
	if p == nil || p.finished || p.recorder == nil {
		return
	}
	p.finished = true
	if outcome == "" {
		outcome = "ok"
	}
	now := p.recorder.now()
	elapsed := nonnegativeDuration(now.Sub(p.started))
	cost := LandPhaseCost{Phase: p.name, Outcome: outcome, Attempt: p.attempt, ElapsedMS: elapsed.Milliseconds()}
	p.recorder.phases = append(p.recorder.phases, cost)
	p.recorder.emit(LandProgressEvent{
		Schema: LandProgressSchema, Phase: p.name, State: "completed", Outcome: outcome, Attempt: p.attempt,
		ElapsedMS: nonnegativeDuration(now.Sub(p.recorder.started)).Milliseconds(), PhaseElapsedMS: elapsed.Milliseconds(),
	})
}

func (r *landRecorder) emit(event LandProgressEvent) {
	if r == nil || r.sink == nil {
		return
	}
	// An observer is evidence-only. Even a caller-supplied sink that panics cannot
	// change commit semantics or strand the resource sampler.
	func() {
		defer func() { _ = recover() }()
		r.sink(event)
	}()
}

func (r *landRecorder) setScan(files int, bytes int64) {
	if r == nil {
		return
	}
	r.scannedFiles = files
	r.scannedBytes = bytes
}

func (r *landRecorder) addWholeTreeScan(files int, bytes int64) {
	if r == nil {
		return
	}
	r.wholeTreeScans++
	r.scannedFiles += files
	r.scannedBytes += bytes
}

func (r *landRecorder) finish(ok bool) *LandCostReceipt {
	if r == nil || r.finished {
		return nil
	}
	r.finished = true
	wall := nonnegativeDuration(r.now().Sub(r.started)).Milliseconds()
	resources := landResourceSample{}
	if r.probe != nil {
		resources = r.probe.stop()
	}
	phases := append([]LandPhaseCost(nil), r.phases...)
	accounted := int64(0)
	slowest := LandPhaseCost{}
	for _, phase := range phases {
		accounted += phase.ElapsedMS
		if phase.ElapsedMS > slowest.ElapsedMS || slowest.Phase == "" {
			slowest = phase
		}
	}
	unattributed := wall - accounted
	if unattributed < 0 {
		// Millisecond rounding can make a sum of disjoint sub-millisecond phases one
		// tick larger than wall. Clamp the presentation; raw phases remain visible.
		unattributed = 0
	}
	outcome := "failed"
	if ok {
		outcome = "ok"
	}
	scanScope := "exact_land_patch"
	if r.wholeTreeScans > 0 {
		scanScope = "exact_land_patch_plus_whole_tree_analyses"
	}
	return &LandCostReceipt{
		Schema: LandCostSchema, Outcome: outcome, WallMS: wall,
		CPUms: resources.cpu.Milliseconds(), CPUObserved: resources.cpuObserved, CPUScope: "lander_process",
		PeakRSSBytes: resources.peakRSSBytes, PeakRSSObserved: resources.peakRSSObserved,
		PeakRSSScope: "process_lifetime", ScannedFiles: r.scannedFiles, ScannedBytes: r.scannedBytes,
		ScanScope: scanScope, CacheReuse: "none", SlowestPhase: slowest.Phase,
		SlowestPhaseMS: slowest.ElapsedMS, AccountedPhaseMS: accounted,
		UnattributedWallMS: unattributed, Phases: phases,
	}
}

func nonnegativeDuration(d time.Duration) time.Duration {
	if d < 0 {
		return 0
	}
	return d
}
