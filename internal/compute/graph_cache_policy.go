package compute

import (
	"errors"
	"sync"
)

// GraphCachePolicy is a host-side keyed CUDA graph cache policy adapted from
// TensorFold's zig/src/cuda/graph_cache.zig Cache.run at pin
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209 (Apache-2.0; see
// THIRD_PARTY_NOTICES.md, "TensorFold CUDA graph-cache policy"). It
// re-expresses the upstream closed contract in idiomatic Go against an
// injected engine so the whole policy is testable on host CPU with a fake
// engine, beside cudaBackend.GraphBegin/GraphEndLaunch in cuda.go:
//
//   - capture-once per key, replay on hit;
//   - a changed address fingerprint drops every cached executable at once
//     (captured graphs bind to buffer addresses);
//   - bounded LRU retention under Cap with a MinKeep floor: a room-miss sheds
//     least-recently-used entries down to MinKeep before capturing;
//   - Hold disables eviction entirely, so nothing is ever evicted and the
//     cache may grow past Cap;
//   - an all-rank agree vote gates every miss; a declined vote runs eager,
//     and hits never consult voters;
//   - a failed BeginCapture (for example OOM) falls back to eager so the
//     token still completes;
//   - RecordPeerFailure disables the cache: it releases every stored
//     executable and runs everything eager from then on;
//   - On=false mirrors the upstream environment-off switch.
//
// Upstream bound preserved: the only concrete callers of the source cache are
// experimental DeepSeek-V4.1 paths and no release binary serves them, so this
// policy ships as an unqualified component. It is not yet wired into the live
// cudaBackend capture path; physical capture/replay qualification remains open.
type GraphCachePolicy struct {
	eng            GraphCacheEngine
	cap            int
	minKeep        int
	hold           bool
	voters         []func() bool
	on             bool
	disabled       bool
	fingerprint    uint64
	hasFingerprint bool
	entries        map[string]graphPolicyEntry
	order          []string // keys, least recently used first
	mu             sync.Mutex
}

// GraphRunOutcome reports which path a GraphCachePolicy.Run took.
type GraphRunOutcome int

const (
	// GraphRunReplayed launched a previously captured executable.
	GraphRunReplayed GraphRunOutcome = iota
	// GraphRunCaptured captured and stored a new executable this run.
	GraphRunCaptured
	// GraphRunEager executed the step without capture or replay.
	GraphRunEager
	// GraphRunCaptureFailed means the capture attempt failed after the miss
	// was admitted; the open capture was aborted and nothing was stored.
	GraphRunCaptureFailed
)

// String renders the outcome for logs and receipts.
func (o GraphRunOutcome) String() string {
	switch o {
	case GraphRunReplayed:
		return "replayed"
	case GraphRunCaptured:
		return "captured"
	case GraphRunEager:
		return "eager"
	case GraphRunCaptureFailed:
		return "capture-failed"
	default:
		return "unknown"
	}
}

// GraphCacheExecutable is a captured-graph handle owned by the engine. The
// policy releases it when its entry is evicted, dropped, or the cache is
// disabled.
type GraphCacheExecutable interface {
	Release()
}

// GraphCacheEngine performs the device-side half of one token step. The
// policy decides which path to take; the engine executes it.
type GraphCacheEngine interface {
	// BeginCapture opens a stream capture for the upcoming op stream.
	BeginCapture() error
	// EndCapture instantiates the open capture into an executable.
	EndCapture() (GraphCacheExecutable, error)
	// AbortCapture ends and discards an open capture, releasing any partially
	// instantiated executable. It must be safe to call after EndCapture
	// failed.
	AbortCapture()
	// Replay launches a stored executable.
	Replay(exec GraphCacheExecutable) error
	// RunEager executes the step without capture or replay.
	RunEager() error
}

// GraphCachePolicyOptions configures NewGraphCachePolicy.
type GraphCachePolicyOptions struct {
	// On is the master switch; false mirrors the upstream environment-off
	// setting and forces every run eager.
	On bool
	// Cap bounds retained executables. Required positive.
	Cap int
	// MinKeep is the eviction floor a room-miss sheds down to before
	// capturing. Must satisfy 0 <= MinKeep < Cap.
	MinKeep int
	// Hold disables eviction: nothing is ever evicted and the cache may grow
	// past Cap.
	Hold bool
	// Voters are per-rank agree votes consulted on misses only. Every voter
	// must return true before a capture is attempted; nil or empty means a
	// single rank that always agrees. A nonempty list must not contain nil.
	Voters []func() bool
}

var (
	// ErrGraphCacheNilEngine reports a nil engine passed to NewGraphCachePolicy.
	ErrGraphCacheNilEngine = errors.New("compute: graph cache policy requires a non-nil engine")
	// ErrGraphCacheBadCap reports a non-positive Cap.
	ErrGraphCacheBadCap = errors.New("compute: graph cache policy cap must be positive")
	// ErrGraphCacheBadMinKeep reports MinKeep outside [0, Cap).
	ErrGraphCacheBadMinKeep = errors.New("compute: graph cache policy min_keep must be below cap")
	// ErrGraphCacheNilVoter reports an absent callback in a nonempty voter list.
	ErrGraphCacheNilVoter = errors.New("compute: graph cache policy voter must be non-nil")
)

type graphPolicyEntry struct {
	exec GraphCacheExecutable
}

// NewGraphCachePolicy validates opts and returns an empty policy. It owns a copy
// of the voter list; callers must not mutate that list during construction.
// State captured by each callback remains the caller's responsibility.
func NewGraphCachePolicy(eng GraphCacheEngine, opts GraphCachePolicyOptions) (*GraphCachePolicy, error) {
	if eng == nil {
		return nil, ErrGraphCacheNilEngine
	}
	if opts.Cap <= 0 {
		return nil, ErrGraphCacheBadCap
	}
	if opts.MinKeep < 0 || opts.MinKeep >= opts.Cap {
		return nil, ErrGraphCacheBadMinKeep
	}
	voters := append([]func() bool(nil), opts.Voters...)
	for _, vote := range voters {
		if vote == nil {
			return nil, ErrGraphCacheNilVoter
		}
	}
	return &GraphCachePolicy{
		eng:     eng,
		cap:     opts.Cap,
		minKeep: opts.MinKeep,
		hold:    opts.Hold,
		voters:  voters,
		on:      opts.On,
		entries: make(map[string]graphPolicyEntry),
	}, nil
}

// Run executes one token step under the cache policy for key at the given
// address fingerprint. On a miss that admits capture, body issues the op
// stream while the capture is open; on every eager path body is not called
// and the engine runs the step itself. Callers that share one policy across
// goroutines rely on its internal lock, which is held across engine calls
// the way cudaBackend holds cudaMu across its graph C calls.
func (p *GraphCachePolicy) Run(key string, fingerprint uint64, body func() error) (GraphRunOutcome, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.on || p.disabled {
		return GraphRunEager, p.eng.RunEager()
	}

	if p.hasFingerprint && fingerprint != p.fingerprint {
		// Executables bind to buffer addresses; a moved footprint invalidates
		// every captured graph at once.
		p.dropAllLocked()
	}
	p.fingerprint = fingerprint
	p.hasFingerprint = true

	if entry, ok := p.entries[key]; ok {
		p.touchLocked(key)
		return GraphRunReplayed, p.eng.Replay(entry.exec)
	}

	// Miss: the all-rank vote gates capture; hits never consult voters.
	for _, vote := range p.voters {
		if !vote() {
			return GraphRunEager, p.eng.RunEager()
		}
	}

	if !p.hold && len(p.entries) >= p.cap {
		// Room-miss: shed least-recently-used entries down to the floor so the
		// new capture lands without breaching Cap.
		for len(p.entries) > p.minKeep {
			p.evictOldestLocked()
		}
	}

	if err := p.eng.BeginCapture(); err != nil {
		// Capture admission failed (for example OOM): the token still has to
		// complete, so fall back to the eager path.
		return GraphRunEager, p.eng.RunEager()
	}
	if err := body(); err != nil {
		p.eng.AbortCapture()
		return GraphRunCaptureFailed, err
	}
	exec, err := p.eng.EndCapture()
	if err != nil {
		p.eng.AbortCapture()
		return GraphRunCaptureFailed, err
	}
	p.entries[key] = graphPolicyEntry{exec: exec}
	p.order = append(p.order, key)
	return GraphRunCaptured, nil
}

// RecordPeerFailure records that a peer rank's capture failed. A peer failure
// poisons the collective assumption behind shared captures, so the policy
// disables itself: every stored executable is released and every later run is
// eager.
func (p *GraphCachePolicy) RecordPeerFailure() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.disabled = true
	p.dropAllLocked()
}

// Len reports the number of retained executables.
func (p *GraphCachePolicy) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.entries)
}

// Disabled reports whether a peer failure switched the policy to eager-only.
func (p *GraphCachePolicy) Disabled() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.disabled
}

// touchLocked moves key to most-recently-used position.
func (p *GraphCachePolicy) touchLocked(key string) {
	for i, k := range p.order {
		if k == key {
			copy(p.order[i:], p.order[i+1:])
			p.order = append(p.order[:len(p.order)-1], key)
			return
		}
	}
}

// evictOldestLocked releases and removes the least-recently-used entry.
func (p *GraphCachePolicy) evictOldestLocked() {
	key := p.order[0]
	p.order = p.order[1:]
	entry := p.entries[key]
	delete(p.entries, key)
	entry.exec.Release()
}

// dropAllLocked releases every stored executable and empties the cache.
func (p *GraphCachePolicy) dropAllLocked() {
	for _, key := range p.order {
		p.entries[key].exec.Release()
	}
	p.entries = make(map[string]graphPolicyEntry)
	p.order = nil
}
