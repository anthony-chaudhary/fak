package model

import (
	"sync"
	"time"
)

// v41_expert_fault_attribution.go — phase-scoped (prefill vs decode) routed-expert
// fault attribution for the DeepSeek V4.1 native forward (fak#13294 DoD item 1).
//
// The physical Strix Halo strix3 run measured 0.1 tok/s prefill (30 tok in
// 492.02s) with the tier's ExpertCheckpointStats reporting only a GLOBAL
// running total — no phase dimension, no f32 dequant accounting — so the log
// cannot say "prefill faulted N experts over T tokens = faults/token". This
// ledger gives the forward one, and the in-kernel planner surfaces it at the
// end of the serve's per-request summary so the next run can attribute the
// 492s.
//
// WHY TWO SEPARATE LEDGERS, NOT ONE LEDGER + A PHASE FIELD: the value the
// witness needs is a RATE PER PHASE — faults/token differs by an order of
// magnitude between prefill (whole prompt thorough every routed expert once,
// with thrash at the token boundaries) and decode (one token, re-faulting what
// the bounded host pool may already have dropped). Folding a single running
// total across phases loses exactly the divisor that makes the number legible:
// faults is meaningless without the token count collected under the SAME
// phase. Separate ledgers keep each phase's denominator with its numerator.
//
// WHY THE RESIDENT ARM COUNTS HITS AND NOT FAULTS: on the tier-only arm a
// routed expert read resolves either from a resident store (m.residentF32Mat —
// the resident f32/quant twin already in RAM, i.e. ZERO tier IO and zero
// dequant) or from an expertCheckpoint.fault (real slab IO + f32
// materialization). A resident-read token would otherwise be invisible
// (all-zero phase → an unattributable run), and counting it as a fault would
// fabricate IO that did not happen. So resident lands in ResidentHits, tier in
// Faults/FaultedBytes/DequantBytes, and ResidentHitFraction =
// residentHits/(residentHits+faults) reads as "what share of routed-expert
// reads this phase served from residency" — 1.0 when the tier is never touched
// at all.
//
// PURITY: the ledger is nil until v41SetExpertFaultPhase is first called, so a
// model that never runs the V4.1 instrumented path allocates nothing and every
// method tolerates a nil receiver. All of it is observation only — no routing,
// arithmetic, or byte layout reads it.

// V41ExpertPhase is the forward phase the attribution attributes faults to.
type V41ExpertPhase int

const (
	// V41PhaseUnknown is the zero value: no phase was ever set on this model,
	// so the ledger is inert and V41ExpertFaultAttribution returns zero costs.
	V41PhaseUnknown V41ExpertPhase = iota
	// V41PhasePrefill is the multi-token prompt pass (one token per id).
	V41PhasePrefill
	// V41PhaseDecode is the single-token generate step (one token per call).
	V41PhaseDecode
)

// String names the phase for log clauses. Never panicked on; the unknown value
// prints as "unknown".
func (p V41ExpertPhase) String() string {
	switch p {
	case V41PhasePrefill:
		return "prefill"
	case V41PhaseDecode:
		return "decode"
	default:
		return "unknown"
	}
}

// v41ExpertFaultPhaseLedger is one phase's accounting: the tokens observed
// under this phase, the routed-expert reads resolved from the checkpoint tier
// (each moving its stride bytes and materializing an f32 block), the reads
// resolved from a resident store (tier IO-free hits), and — #13299 — the
// WALL-CLOCK split across the three real call boundaries (fault door, f32
// dequant, scalar contraction) that the byte counts alone cannot separate.
//
// The three duration fields are separate by construction, never one total
// copied: a fault is the tier's per-expert range read, a dequant is the f32
// materialization of the faulted stride, and a contraction is the SwiGLU the
// routed pick applies. They nest in TIME (each fault is followed by its
// dequant, each pick by its contraction) but are summed into three independent
// accumulators so a run can say which one dominates.
type v41ExpertFaultPhaseLedger struct {
	Tokens                   int     `json:"tokens"`
	Faults                   int     `json:"faults"`
	FaultedBytes             int64   `json:"faulted_bytes"`
	DequantBytes             int64   `json:"dequant_bytes"`
	ResidentHits             int     `json:"resident_hits"`
	Contractions             int     `json:"contractions"`
	FaultDoorNanos           int64   `json:"fault_nanos"`
	DequantNanos             int64   `json:"dequant_nanos"`
	ContractionNanos         int64   `json:"contraction_nanos"`
	FaultNanosPerToken       float64 `json:"fault_nanos_per_token"`
	DequantNanosPerToken     float64 `json:"dequant_nanos_per_token"`
	ContractionNanosPerToken float64 `json:"contraction_nanos_per_token"`
	FaultsPerToken           float64 `json:"faults_per_token"`
	FaultedBytesPerToken     float64 `json:"faulted_bytes_per_token"`
	DequantBytesPerToken     float64 `json:"dequant_bytes_per_token"`
	ResidentHitFraction      float64 `json:"resident_hit_fraction"`
	// ContractionBackend names the engine the contraction ran on ("host" or
	// "vulkan"), observed from the model's selection at the last contraction —
	// an identity, not a device receipt.
	ContractionBackend string `json:"contraction_backend"`
}

// V41ExpertFaultAttribution is the model-level snapshot served to the agent
// planner's execution-summary log: prefill and decode ledgers are INDEPENDENT
// running totals over the model's lifetime (the serve resets per request), so
// the consumer derives per-request attribution by differencing consecutive
// snapshots the same way it already differences the tier stats.
type V41ExpertFaultAttribution struct {
	Prefill v41ExpertFaultPhaseLedger `json:"prefill"`
	Decode  v41ExpertFaultPhaseLedger `json:"decode"`
}

// v41ExpertFaultLedger is the mutex-guarded accumulator a V4.1 model's
// routed-expert reads record into: one guard byte is deliberately cheaper than
// per-field atomics for a diagnostic that fires once per request summary, and
// a plain struct keeps snapshot() a simple consistent copy. The ledger lives
// in a package registry keyed by *Model (below) rather than as a Model field,
// so adding this observation touches no shared struct layout; the *Model is
// process-lifetime on every serve that carries a checkpoint tier, so
// key-by-pointer retention matches the object it observes.
type v41ExpertFaultLedger struct {
	mu    sync.Mutex
	phase V41ExpertPhase
	pre   v41ExpertFaultPhaseLedger
	dec   v41ExpertFaultPhaseLedger

	// nowNanos reads a monotonic clock in nanoseconds; nil means time.Now. It
	// is the seam a deterministic test drives so the measured durations are
	// exact instead of wall-clock noise.
	nowNanos func() int64
	// backend names the contraction engine the last contraction observed
	// ("host" default, "vulkan" when selected); see ContractionBackend.
	backend string
}

// v41ExpertFaultLedgers maps *Model -> *v41ExpertFaultLedger, installed ONLY
// by v41SetExpertFaultPhase (the instrumented V4.1 wiring) and read by
// everything else. A model that never declares a phase is never inserted, so
// the uninstrumented default allocates nothing and its note*/snapshot are one
// map miss. sync.Map because the common regime is install-once, read-many and
// inserts from setPhase never race with model construction.
var v41ExpertFaultLedgers sync.Map

// setPhase declares which phase the V4.1 forward is running, so subsequent
// note* calls land in the right ledger. nil- and default-safe: every method
// tolerates a nil receiver.
func (l *v41ExpertFaultLedger) setPhase(p V41ExpertPhase) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.phase = p
}

// noteTierFault records one routed-expert read resolved from the checkpoint
// tier under the CURRENT phase: raw is the faulted byte stride the tier moved
// (the entry stride the weight rebuild carries) and dequant the f32 bytes the
// dequant materialized for it. A phase-less ledger routes nowhere, so the
// note is silently dropped under the inert default.
func (l *v41ExpertFaultLedger) noteTierFault(raw, dequant int64) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	led := l.ledgerLocked()
	if led == nil {
		return
	}
	led.Faults++
	led.FaultedBytes += raw
	led.DequantBytes += dequant
}

// noteResidentHit records one routed-expert read resolved from a resident
// store (no tier IO, no dequant) under the CURRENT phase. A phase-less ledger
// routes nowhere, so the note is silently dropped under the inert default.
func (l *v41ExpertFaultLedger) noteResidentHit() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	led := l.ledgerLocked()
	if led == nil {
		return
	}
	led.ResidentHits++
}

// noteToken records one token processed under the CURRENT phase: prefill notes
// one per id (the forward is a batch over the prompt), decode one per Step.
// A phase-less ledger routes nowhere, so the note is silently dropped under
// the inert default.
func (l *v41ExpertFaultLedger) noteToken(n int) {
	if l == nil || n <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	led := l.ledgerLocked()
	if led == nil {
		return
	}
	led.Tokens += n
}

// ledgerLocked returns the ledger the current phase selects. The caller must
// hold l.mu. An unset/unknown phase routes nowhere (the inert default), so a
// stray note from a path that never declared its phase cannot corrupt a
// phase's denominator.
func (l *v41ExpertFaultLedger) ledgerLocked() *v41ExpertFaultPhaseLedger {
	if l.phase == V41PhasePrefill {
		return &l.pre
	}
	if l.phase == V41PhaseDecode {
		return &l.dec
	}
	return nil
}

// nowLocked reads the ledger's monotonic clock (nanoseconds). The caller must
// hold l.mu. A nil seam reads time.Now, so the default path needs no
// installation and the instrument is inert when never consulted.
func (l *v41ExpertFaultLedger) nowLocked() int64 {
	if l.nowNanos != nil {
		return l.nowNanos()
	}
	return time.Now().UnixNano()
}

// noteFaultDoor records one tier fault door's wall-clock cost under the CURRENT
// phase: the elapsed (fault door close - open) the caller measured around the
// checkpoint range read. A phase-less ledger routes nowhere and drops the note.
func (l *v41ExpertFaultLedger) noteFaultDoor(nanos int64) {
	l.noteDuration(nanos, func(led *v41ExpertFaultPhaseLedger) { led.FaultDoorNanos += nanos })
}

// noteDequant records one f32 materialization's wall-clock cost under the
// CURRENT phase. A phase-less ledger routes nowhere.
func (l *v41ExpertFaultLedger) noteDequant(nanos int64) {
	l.noteDuration(nanos, func(led *v41ExpertFaultPhaseLedger) { led.DequantNanos += nanos })
}

// noteContraction records one routed-pick contraction under the CURRENT phase:
// its wall-clock cost and the backend that ran it. A phase-less ledger routes
// nowhere.
func (l *v41ExpertFaultLedger) noteContraction(nanos int64) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	led := l.ledgerLocked()
	if led == nil {
		return
	}
	led.Contractions++
	if nanos > 0 {
		led.ContractionNanos += nanos
	}
	led.ContractionBackend = l.backendNameLocked()
}

// noteDuration is the shared guard/route for a single timed boundary. Negative
// durations are dropped: a non-monotonic reading must not subtract time.
func (l *v41ExpertFaultLedger) noteDuration(nanos int64, add func(*v41ExpertFaultPhaseLedger)) {
	if l == nil || nanos < 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	led := l.ledgerLocked()
	if led == nil {
		return
	}
	add(led)
}

// backendNameLocked is the contraction-backend identity, defaulting to "host"
// for the scalar arm the V4.1 routed contraction runs on today. The caller must
// hold l.mu.
func (l *v41ExpertFaultLedger) backendNameLocked() string {
	if l.backend == "" {
		return v41ContractionBackendHost
	}
	return l.backend
}

// setBackend records the contraction-backend identity the notes observe.
func (l *v41ExpertFaultLedger) setBackend(name string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.backend = name
}

// snapshot returns the attributed ledgers with all four derived rates
// computed. Denominator-zero phases report 0 (never NaN), so a phase the run
// has not entered yet prints 0s rather than Inf and fails no JSON consumer.
func (l *v41ExpertFaultLedger) snapshot() V41ExpertFaultAttribution {
	out := V41ExpertFaultAttribution{}
	if l == nil {
		return out
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	out.Prefill = l.deriveLocked(l.pre)
	out.Decode = l.deriveLocked(l.dec)
	return out
}

// deriveLocked copies one phase ledger and fills the derived per-token rates
// and the resident hit fraction. The caller must hold l.mu.
func (l *v41ExpertFaultLedger) deriveLocked(led v41ExpertFaultPhaseLedger) v41ExpertFaultPhaseLedger {
	if led.Tokens > 0 {
		ft := float64(led.Tokens)
		led.FaultsPerToken = float64(led.Faults) / ft
		led.FaultedBytesPerToken = float64(led.FaultedBytes) / ft
		led.DequantBytesPerToken = float64(led.DequantBytes) / ft
		led.FaultNanosPerToken = float64(led.FaultDoorNanos) / ft
		led.DequantNanosPerToken = float64(led.DequantNanos) / ft
		led.ContractionNanosPerToken = float64(led.ContractionNanos) / ft
	}
	if denom := led.ResidentHits + led.Faults; denom > 0 {
		led.ResidentHitFraction = float64(led.ResidentHits) / float64(denom)
	}
	// The contraction-backend identity is only meaningful once a contraction
	// actually ran: an inert ledger must stay the exact zero value (the purity
	// contract the #13294 witness pins), so the backend is defaulted here only
	// when Contractions > 0. noteContraction also stamps it at record time.
	if led.Contractions > 0 && led.ContractionBackend == "" {
		led.ContractionBackend = v41ContractionBackendHost
	}
	return led
}

// v41ExpertFaultLedgerOf returns the model's ledger, inserting one ONLY when
// install is set. note*/snapshot pass install=false so they never create
// state; setPhase passes install=true. nil- and default-safe.
func v41ExpertFaultLedgerOf(m *Model, install bool) *v41ExpertFaultLedger {
	if m == nil {
		return nil
	}
	if !install {
		if l, ok := v41ExpertFaultLedgers.Load(m); ok {
			return l.(*v41ExpertFaultLedger)
		}
		return nil
	}
	l, _ := v41ExpertFaultLedgers.LoadOrStore(m, &v41ExpertFaultLedger{})
	return l.(*v41ExpertFaultLedger)
}

// v41SetExpertFaultPhase declares the phase the V4.1 forward is in (creating
// the model's ledger on first use — this is the ONLY creator), or clears it
// back to the inert default with phase == V41PhaseUnknown (the defer-restore
// after a phase ends). Nil-model safe. Package-internal wiring; the agent
// planner only ever READS through V41ExpertFaultAttribution.
//
// This is deliberately on *Model (not Session): the tier the reads resolve
// through hangs off the Model, so concurrent sessions over one model attribute
// to one shared ledger — which is the honest count over the model's lifetime;
// the per-request differencing that makes that legible happens in the caller.
func (m *Model) v41SetExpertFaultPhase(p V41ExpertPhase) {
	l := v41ExpertFaultLedgerOf(m, true)
	l.setPhase(p)
}

// v41NoteExpertFaultToken records n tokens under the currently-set phase.
// nil-safe.
func (m *Model) v41NoteExpertFaultToken(n int) {
	v41ExpertFaultLedgerOf(m, false).noteToken(n)
}

// v41NoteExpertTierFault records one tier-faulted routed-expert read under the
// currently-set phase: rawBytes the faulted stride, dequantBytes the f32 block
// the dequant materialized. nil-safe.
func (m *Model) v41NoteExpertTierFault(rawBytes, dequantBytes int64) {
	v41ExpertFaultLedgerOf(m, false).noteTierFault(rawBytes, dequantBytes)
}

// v41NoteExpertResidentHit records one resident-store resolved routed-expert
// read under the currently-set phase. nil-safe.
func (m *Model) v41NoteExpertResidentHit() {
	v41ExpertFaultLedgerOf(m, false).noteResidentHit()
}

// V41ExpertFaultAttribution returns the phase-split routed-expert fault
// attribution snapshot for this model (the zero-value attribution when the
// V4.1 instrumented path never ran, and never NaN). This is the read surface
// the agent planner's execution-summary logs through an interface assertion so
// non-V4.1 models stay byte-for-byte unchanged.
func (m *Model) V41ExpertFaultAttribution() V41ExpertFaultAttribution {
	return v41ExpertFaultLedgerOf(m, false).snapshot()
}

// v41ContractionBackendHost is the default contraction-backend identity: the
// scalar host SwiGLU the V4.1 routed-expert contraction runs on.
const v41ContractionBackendHost = "host"

// v41SetExpertTimeClock installs the monotonic clock (nanoseconds) the phase
// ledger reads to time the fault/dequant/contraction boundaries. nil restores
// time.Now. It creates the model's ledger on first use (like setPhase), so a
// test can install the clock before any phase is set. nil-model safe.
func (m *Model) v41SetExpertTimeClock(now func() int64) {
	l := v41ExpertFaultLedgerOf(m, true)
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.nowNanos = now
}

// v41NowNanos reads the model's installed clock (nil => time.Now), so a caller
// times a boundary with the same clock the ledger accumulates into. It is
// inert (returns 0) when the model has no live ledger, so the default path
// measures nothing.
func (m *Model) v41NowNanos() int64 {
	l := v41ExpertFaultLedgerOf(m, false)
	if l == nil {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.nowLocked()
}

// v41SetContractionBackend declares the engine identity the routed-expert
// contraction notes observe ("host" or "vulkan"). Empty resets to the host
// default. This names the selection honestly; it does not run a device
// contraction. nil-model safe.
func (m *Model) v41SetContractionBackend(name string) {
	l := v41ExpertFaultLedgerOf(m, true)
	if l == nil {
		return
	}
	if name == "" {
		name = v41ContractionBackendHost
	}
	l.setBackend(name)
}

// v41NoteExpertFaultDoorNanos records the wall-clock cost of one tier fault
// door under the currently-set phase. nil-safe.
func (m *Model) v41NoteExpertFaultDoorNanos(nanos int64) {
	v41ExpertFaultLedgerOf(m, false).noteFaultDoor(nanos)
}

// v41NoteExpertDequantNanos records the wall-clock cost of one f32
// materialization under the currently-set phase. nil-safe.
func (m *Model) v41NoteExpertDequantNanos(nanos int64) {
	v41ExpertFaultLedgerOf(m, false).noteDequant(nanos)
}

// v41NoteExpertContraction records one routed-pick contraction under the
// currently-set phase (count and backend; duration via the elapsed variant).
// nil-safe.
func (m *Model) v41NoteExpertContraction() {
	v41ExpertFaultLedgerOf(m, false).noteContraction(0)
}

// v41NoteExpertContractionNanos records one routed-pick contraction with its
// wall-clock cost under the currently-set phase. nil-safe.
func (m *Model) v41NoteExpertContractionNanos(nanos int64) {
	v41ExpertFaultLedgerOf(m, false).noteContraction(nanos)
}
