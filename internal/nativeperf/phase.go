package nativeperf

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// Phase is a bounded fak-native execution phase. Values are stable metric labels.
type Phase string

const (
	PhaseQueueAdmission  Phase = "queue_admission"
	PhaseModelLoad       Phase = "model_load"
	PhaseTokenization    Phase = "tokenization"
	PhasePrefill         Phase = "prefill"
	PhaseDecode          Phase = "decode"
	PhaseKVLookup        Phase = "kv_lookup"
	PhaseKVRestore       Phase = "kv_restore"
	PhaseKVEvict         Phase = "kv_evict"
	PhaseHostUpload      Phase = "host_device_upload"
	PhaseHostDownload    Phase = "host_device_download"
	PhaseKernel          Phase = "kernel"
	PhaseSynchronization Phase = "synchronization"
	PhaseSampling        Phase = "sampling"
	PhaseOutput          Phase = "output"
)

var phaseOrder = [...]Phase{
	PhaseQueueAdmission, PhaseModelLoad, PhaseTokenization, PhasePrefill, PhaseDecode,
	PhaseKVLookup, PhaseKVRestore, PhaseKVEvict, PhaseHostUpload, PhaseHostDownload,
	PhaseKernel, PhaseSynchronization, PhaseSampling, PhaseOutput,
}

// Phases returns the stable, bounded phase label set.
func Phases() []Phase { return append([]Phase(nil), phaseOrder[:]...) }

// PhaseTiming partitions an exclusive phase duration. Active and Wait must
// reconcile to Wall; callers use Wait for queueing, synchronization, and
// blocking transfers, and Active for owned CPU/device work.
type PhaseTiming struct {
	Wall   time.Duration `json:"wall"`
	Active time.Duration `json:"active"`
	Wait   time.Duration `json:"wait"`
}

// PhaseAccounting carries both inclusive and exclusive timing. Inclusive may
// contain nested child work. Only Exclusive participates in request totals and
// exported counters, preventing nested or asynchronous work from being counted
// twice.
type PhaseAccounting struct {
	Phase     Phase       `json:"phase"`
	Parent    Phase       `json:"parent,omitempty"`
	Inclusive PhaseTiming `json:"inclusive"`
	Exclusive PhaseTiming `json:"exclusive"`
}

// PhaseReceipt is the bounded per-request fak-native accounting receipt.
type PhaseReceipt struct {
	Engine         string            `json:"engine"`
	Backend        string            `json:"backend"`
	ForwardPath    string            `json:"forward_path"`
	FallbackActive bool              `json:"fallback_active"`
	Wall           time.Duration     `json:"wall"`
	Phases         []PhaseAccounting `json:"phases"`
}

// Validate proves that the receipt identifies fak-native execution, uses only
// bounded phases, reconciles every timing partition, and has no exclusive
// nested double-counting. tolerance admits clock quantization only.
func (r PhaseReceipt) Validate(tolerance time.Duration) error {
	if boundedEngine(r.Engine) == "other" || r.FallbackActive {
		return errors.New("phase receipt is not non-fallback fak-native execution")
	}
	if r.Wall < 0 || tolerance < 0 {
		return errors.New("negative wall time or tolerance")
	}
	allowed := make(map[Phase]bool, len(phaseOrder))
	for _, phase := range phaseOrder {
		allowed[phase] = true
	}
	seen := make(map[Phase]bool, len(r.Phases))
	var total time.Duration
	for _, p := range r.Phases {
		if !allowed[p.Phase] || seen[p.Phase] {
			return fmt.Errorf("unbounded or duplicate phase %q", p.Phase)
		}
		seen[p.Phase] = true
		if p.Parent != "" && !allowed[p.Parent] {
			return fmt.Errorf("phase %q has unbounded parent %q", p.Phase, p.Parent)
		}
		if err := validateTiming(p.Inclusive, tolerance); err != nil {
			return fmt.Errorf("phase %q inclusive: %w", p.Phase, err)
		}
		if err := validateTiming(p.Exclusive, tolerance); err != nil {
			return fmt.Errorf("phase %q exclusive: %w", p.Phase, err)
		}
		if p.Exclusive.Wall > p.Inclusive.Wall+tolerance {
			return fmt.Errorf("phase %q exclusive exceeds inclusive", p.Phase)
		}
		if total > time.Duration(math.MaxInt64)-p.Exclusive.Wall {
			return errors.New("phase wall overflow")
		}
		total += p.Exclusive.Wall
	}
	if len(seen) != len(phaseOrder) {
		return fmt.Errorf("phase receipt has %d phases, want complete bounded set of %d", len(seen), len(phaseOrder))
	}
	if absDuration(total-r.Wall) > tolerance {
		return fmt.Errorf("exclusive phase wall %s does not reconcile request wall %s", total, r.Wall)
	}
	return nil
}

func validateTiming(t PhaseTiming, tolerance time.Duration) error {
	if t.Wall < 0 || t.Active < 0 || t.Wait < 0 {
		return errors.New("negative timing")
	}
	if absDuration(t.Active+t.Wait-t.Wall) > tolerance {
		return fmt.Errorf("active %s plus wait %s does not reconcile wall %s", t.Active, t.Wait, t.Wall)
	}
	return nil
}

func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}

// SortedPhases returns a stable copy for receipts and deterministic tests.
func (r PhaseReceipt) SortedPhases() []PhaseAccounting {
	out := append([]PhaseAccounting(nil), r.Phases...)
	sort.Slice(out, func(i, j int) bool { return out[i].Phase < out[j].Phase })
	return out
}

// WorkKind describes whether an interval owns execution or waits for another
// resource. It is intentionally a closed label set.
type WorkKind uint8

const (
	WorkActive WorkKind = iota + 1
	WorkWait
)

type phaseInterval struct {
	phase  Phase
	parent Phase
	start  time.Duration
	end    time.Duration
	kind   WorkKind
	order  int
}

// PhaseRecorder builds a receipt from monotonic offsets relative to one
// request. Intervals may be nested or asynchronous. Finalize partitions the
// timeline once, assigning each instant to the deepest active interval (then
// stable insertion order), so exclusive totals cannot double-count overlap.
type PhaseRecorder struct {
	engine      string
	backend     string
	forwardPath string
	fallback    bool
	wall        time.Duration
	intervals   []phaseInterval
}

func NewPhaseRecorder(engine, backend, forwardPath string, wall time.Duration) *PhaseRecorder {
	return &PhaseRecorder{engine: engine, backend: backend, forwardPath: forwardPath, wall: wall}
}

func (r *PhaseRecorder) SetFallbackActive(active bool) { r.fallback = active }

func (r *PhaseRecorder) Add(phase, parent Phase, start, end time.Duration, kind WorkKind) error {
	if r == nil {
		return errors.New("nil phase recorder")
	}
	if start < 0 || end < start || end > r.wall {
		return fmt.Errorf("phase %q interval [%s,%s] outside request wall %s", phase, start, end, r.wall)
	}
	if kind != WorkActive && kind != WorkWait {
		return fmt.Errorf("phase %q has invalid work kind", phase)
	}
	if !knownPhase(phase) || (parent != "" && !knownPhase(parent)) {
		return fmt.Errorf("unbounded phase %q or parent %q", phase, parent)
	}
	r.intervals = append(r.intervals, phaseInterval{phase: phase, parent: parent, start: start, end: end, kind: kind, order: len(r.intervals)})
	return nil
}

func (r *PhaseRecorder) Finalize(tolerance time.Duration) (PhaseReceipt, error) {
	if r == nil {
		return PhaseReceipt{}, errors.New("nil phase recorder")
	}
	byPhase := make(map[Phase][]phaseInterval)
	parent := make(map[Phase]Phase)
	points := []time.Duration{0, r.wall}
	for _, iv := range r.intervals {
		if prior, ok := parent[iv.phase]; !ok {
			parent[iv.phase] = iv.parent
		} else if prior != iv.parent {
			// A bounded phase such as kernel may occur under both prefill and
			// decode. Its aggregate parent is intentionally omitted; each
			// interval still carries its parent for exclusive-depth ordering.
			parent[iv.phase] = ""
		}
		byPhase[iv.phase] = append(byPhase[iv.phase], iv)
		points = append(points, iv.start, iv.end)
	}
	sort.Slice(points, func(i, j int) bool { return points[i] < points[j] })
	points = compactDurations(points)
	accounts := make(map[Phase]*PhaseAccounting, len(phaseOrder))
	for _, phase := range phaseOrder {
		accounts[phase] = &PhaseAccounting{Phase: phase, Parent: parent[phase]}
	}
	for phase, intervals := range byPhase {
		accounts[phase].Inclusive = unionTiming(intervals)
	}
	for i := 0; i+1 < len(points); i++ {
		start, end := points[i], points[i+1]
		if end == start {
			continue
		}
		winner := -1
		winnerDepth := -1
		for j, iv := range r.intervals {
			if iv.start > start || iv.end < end {
				continue
			}
			depth, err := intervalDepth(iv, parent)
			if err != nil {
				return PhaseReceipt{}, err
			}
			if depth > winnerDepth || (depth == winnerDepth && (winner < 0 || iv.order < r.intervals[winner].order)) {
				winner, winnerDepth = j, depth
			}
		}
		if winner < 0 {
			return PhaseReceipt{}, fmt.Errorf("unattributed request interval [%s,%s]", start, end)
		}
		iv := r.intervals[winner]
		addTiming(&accounts[iv.phase].Exclusive, end-start, iv.kind)
	}
	phases := make([]PhaseAccounting, 0, len(phaseOrder))
	for _, phase := range phaseOrder {
		phases = append(phases, *accounts[phase])
	}
	receipt := PhaseReceipt{Engine: r.engine, Backend: r.backend, ForwardPath: r.forwardPath, FallbackActive: r.fallback, Wall: r.wall, Phases: phases}
	if err := receipt.Validate(tolerance); err != nil {
		return PhaseReceipt{}, err
	}
	return receipt, nil
}

func knownPhase(want Phase) bool {
	for _, phase := range phaseOrder {
		if phase == want {
			return true
		}
	}
	return false
}

func intervalDepth(iv phaseInterval, parents map[Phase]Phase) (int, error) {
	seen := map[Phase]bool{iv.phase: true}
	depth := 1
	phase := iv.parent
	for phase != "" {
		if seen[phase] {
			return 0, fmt.Errorf("phase parent cycle at %q", phase)
		}
		seen[phase] = true
		phase = parents[phase]
		depth++
	}
	return depth, nil
}

func compactDurations(in []time.Duration) []time.Duration {
	out := in[:0]
	for _, d := range in {
		if len(out) == 0 || out[len(out)-1] != d {
			out = append(out, d)
		}
	}
	return out
}

func unionTiming(intervals []phaseInterval) PhaseTiming {
	if len(intervals) == 0 {
		return PhaseTiming{}
	}
	points := make([]time.Duration, 0, len(intervals)*2)
	for _, iv := range intervals {
		points = append(points, iv.start, iv.end)
	}
	sort.Slice(points, func(i, j int) bool { return points[i] < points[j] })
	points = compactDurations(points)
	var out PhaseTiming
	for i := 0; i+1 < len(points); i++ {
		start, end := points[i], points[i+1]
		kind := WorkKind(0)
		for _, iv := range intervals {
			if iv.start <= start && iv.end >= end {
				if iv.kind == WorkActive {
					kind = WorkActive
					break
				}
				kind = WorkWait
			}
		}
		if kind != 0 {
			addTiming(&out, end-start, kind)
		}
	}
	return out
}

func addTiming(t *PhaseTiming, duration time.Duration, kind WorkKind) {
	t.Wall += duration
	if kind == WorkWait {
		t.Wait += duration
	} else {
		t.Active += duration
	}
}
