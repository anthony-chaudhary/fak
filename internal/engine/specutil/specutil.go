// Package specutil carries the bit-exact logprob parity witness (the G2
// qualification witness) and the (backend, model) qualification matrix that
// records which pairs passed that witness. It is the code-side deliverable for
// the Vulkan spec-decode default-on flip: the gateway consults the matrix on the
// hot path to admit a qualified pair to default-on MTP, and the parity harness
// is the replayable receipt that a pair was bit-exact.
//
// The physical G1 qualification (bit-exact parity on live silicon) is deferred
// — no live silicon is available in this build — so the built-in matrix carries
// a single MOCK-qualified pair. Hardware-qualified pairs are added as
// live-silicon witnesses land; the matrix structure and the default-on code
// path are the in-scope deliverable here.
package specutil

import (
	"math"
	"sync"
)

// TokenLogprob is one emitted token with its logprob, as observed by the
// bit-exact parity harness.
type TokenLogprob struct {
	TokenID int64
	Logprob float64
}

// ParityResult is the bit-exact logprob parity witness outcome (G2).
type ParityResult struct {
	Pass            bool
	DivergenceIndex int // first index where token ID or logprob differs; -1 if none
	SpecOnLength    int
	SpecOffLength   int
}

// CompareLogprobs reports whether two (tokenID, logprob) sequences are
// bit-exact identical. No tolerance: any difference in token ID or logprob
// (even a single ULP), or any length mismatch, fails. This is the G2 witness.
//
// DivergenceIndex is the first index at which a token ID or logprob differs,
// or -1 when the sequences are bit-exact. When the sequences share all
// positions up to the shorter length but differ in length, DivergenceIndex is
// the length of the shorter sequence — the first index the longer sequence
// holds that the shorter one lacks.
func CompareLogprobs(specOn, specOff []TokenLogprob) ParityResult {
	n := len(specOn)
	if len(specOff) < n {
		n = len(specOff)
	}
	for i := 0; i < n; i++ {
		// Bit-exact: compare the float64 bit pattern, not the value. == treats
		// +0.0 and -0.0 as equal (different bit patterns); a bit-exact witness
		// must reject that. No epsilon: a single ULP of drift fails.
		if specOn[i].TokenID != specOff[i].TokenID ||
			math.Float64bits(specOn[i].Logprob) != math.Float64bits(specOff[i].Logprob) {
			return ParityResult{
				Pass:            false,
				DivergenceIndex: i,
				SpecOnLength:    len(specOn),
				SpecOffLength:   len(specOff),
			}
		}
	}
	if len(specOn) != len(specOff) {
		return ParityResult{
			Pass:            false,
			DivergenceIndex: n,
			SpecOnLength:    len(specOn),
			SpecOffLength:   len(specOff),
		}
	}
	return ParityResult{
		Pass:            true,
		DivergenceIndex: -1,
		SpecOnLength:    len(specOn),
		SpecOffLength:   len(specOff),
	}
}

// QualificationKey identifies a (backend, model) pair for spec-decode
// qualification.
type QualificationKey struct {
	Backend string // compute.Backend.Name() — stable backend id, e.g. "vulkan"
	Model   string // model.Config.Name — stable checkpoint identity
}

// Qualification is the recorded qualification status for one (backend, model)
// pair.
type Qualification struct {
	Key           QualificationKey
	Qualified     bool
	Parity        ParityResult
	Date          string // RFC3339
	EngineSHA     string
	SampleCount   int
	EvidenceClass string // "HW-WITNESSED" or "MOCK"
}

// Matrix is the qualification matrix: the set of qualified (backend, model)
// pairs. It is the replayable receipt of which pairs passed the bit-exact
// parity witness.
//
// Matrix is safe for concurrent use: the gateway calls IsQualified on the hot
// path, so all reads and writes are guarded by an RWMutex.
type Matrix struct {
	mu      sync.RWMutex
	entries map[QualificationKey]Qualification
}

// NewMatrix returns an empty qualification matrix.
func NewMatrix() *Matrix {
	return &Matrix{entries: make(map[QualificationKey]Qualification)}
}

// Record adds or updates a qualification in the matrix.
func (m *Matrix) Record(q Qualification) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[q.Key] = q
}

// IsQualified reports whether the (backend, model) pair is qualified for
// default-on spec decode: it must be recorded, marked qualified, and carry
// live-silicon evidence (HW-WITNESSED). A MOCK entry is a placeholder and does
// NOT qualify a pair for default-on — the "no non-qualified pairs" rule:
// default-on applies only to pairs that passed the bit-exact parity witness on
// live silicon.
func (m *Matrix) IsQualified(backend, model string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	q, ok := m.entries[QualificationKey{Backend: backend, Model: model}]
	return ok && q.Qualified && q.EvidenceClass == "HW-WITNESSED"
}

// Get returns the qualification for the (backend, model) pair, if present.
func (m *Matrix) Get(backend, model string) (Qualification, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	q, ok := m.entries[QualificationKey{Backend: backend, Model: model}]
	return q, ok
}

// defaultMatrix is the built-in qualification matrix, initialized once at
// package load. It carries one mock-qualified pair (the qualification-matrix
// deliverable for this ticket); hardware-qualified pairs are added as
// live-silicon witnesses land.
var defaultMatrix = newDefaultMatrix()

// newDefaultMatrix builds the built-in matrix. The single entry is the
// mock-qualified (vulkan, qwen38:27b-q4) pair: the Qwen3.8 27B Q4_K_M
// checkpoint (the canonical modelreg alias "qwen38:27b-q4") on the Vulkan
// backend. EvidenceClass is MOCK because no live silicon is available in this
// build; the parity witness is the mock arm, fully unit-tested.
func newDefaultMatrix() *Matrix {
	m := NewMatrix()
	m.Record(Qualification{
		Key:           QualificationKey{Backend: "vulkan", Model: "qwen38:27b-q4"},
		Qualified:     true,
		Parity:        ParityResult{Pass: true, DivergenceIndex: -1, SpecOnLength: 5, SpecOffLength: 5},
		Date:          "2026-11-15T00:00:00Z",
		EngineSHA:     "mock",
		SampleCount:   5,
		EvidenceClass: "MOCK",
	})
	return m
}

// DefaultMatrix returns the built-in qualification matrix. It carries one
// mock-qualified pair (the qualification matrix deliverable for this ticket);
// hardware-qualified pairs are added as live-silicon witnesses land.
func DefaultMatrix() *Matrix {
	return defaultMatrix
}
