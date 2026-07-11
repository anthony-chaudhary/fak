package l3kv

// audit.go adds an OPT-IN, interface-identical audit decorator over an
// abi.KVBackend (#3388). AuditKV wraps any abi.KVBackend and implements the EXACT
// same interface — a drop-in replacement — timing each operation and recording its
// per-op latency and a bytes-moved estimate into a caller-provided Recorder before
// forwarding the call, unchanged, to the inner backend.
//
// Opt-in / zero-cost-when-unused. AuditKV is only ever reached when a caller
// constructs it explicitly with NewAuditKV. Nothing in this leaf registers it,
// wires it into a factory, or reads an env var — a default build never allocates
// one, so an unwrapped abi.KVBackend keeps byte-identical behavior and pays zero
// overhead. Wrapping changes no return value and no error: every method returns
// exactly what the inner backend returned.
//
// Bytes-moved estimate (documented per op). The recorder is told a magnitude for
// each call; the estimate is deliberately cheap (no extra allocation, no re-hash)
// and is derived from the argument or result already in hand:
//
//   - Len         → 0. A live-length query moves no span bytes; it returns a count.
//   - Prefill     → len(logits)*4. Prefill returns the next-token logits vector
//                   ([]float32, 4 bytes each) — the dominant payload it produces.
//   - Evict       → int64(positionsRemoved). Eviction moves no bytes off-box; the
//                   count of renumbered/removed positions the inner backend reports
//                   is the closest movement magnitude.
//   - ModelID     → len(id). The returned id string's length in bytes.
//   - StageSpan   → res.BytesMoved. The AUTHORITATIVE bytes-moved the residency
//                   outcome already carries (0 for the in-process no-op, real bytes
//                   for a durable/remote tier).
//   - RestoreSpan → res.BytesMoved. Same authoritative field on the restore outcome.

import (
	"context"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Op names the audited operation in a recorded entry. They are stable string
// constants so a Recorder can key metrics by op without matching on free text.
const (
	OpLen         = "Len"
	OpPrefill     = "Prefill"
	OpEvict       = "Evict"
	OpModelID     = "ModelID"
	OpStageSpan   = "StageSpan"
	OpRestoreSpan = "RestoreSpan"
)

// Recorder is the caller-provided sink AuditKV reports each timed operation to. It
// is called exactly once per audited call, AFTER the inner backend returns, with
// the op name, the wall-clock latency, and the estimated bytes moved. A Recorder
// must be safe for the concurrency its caller drives the backend at; the shipped
// MemRecorder is mutex-guarded, NopRecorder is trivially safe.
type Recorder interface {
	Record(op string, dur time.Duration, bytes int64)
}

// AuditEntry is one observation MemRecorder retains: which op ran, how long it
// took, and the bytes-moved estimate for that call.
type AuditEntry struct {
	Op    string
	Dur   time.Duration
	Bytes int64
}

// MemRecorder is a simple in-memory Recorder that appends every observation to a
// slice for a test or an off-path aggregation to read back. It is mutex-guarded so
// it can back a backend driven concurrently.
type MemRecorder struct {
	mu      sync.Mutex
	entries []AuditEntry
}

// NewMemRecorder returns an empty in-memory recorder.
func NewMemRecorder() *MemRecorder { return &MemRecorder{} }

// Record appends one observation.
func (r *MemRecorder) Record(op string, dur time.Duration, bytes int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries = append(r.entries, AuditEntry{Op: op, Dur: dur, Bytes: bytes})
}

// Entries returns a copy of the observations recorded so far, in call order.
func (r *MemRecorder) Entries() []AuditEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]AuditEntry(nil), r.entries...)
}

// Len reports how many observations have been recorded.
func (r *MemRecorder) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.entries)
}

// nopRecorder discards every observation — the zero-overhead default when a caller
// wraps a backend without wanting to keep the entries.
type nopRecorder struct{}

func (nopRecorder) Record(string, time.Duration, int64) {}

// NopRecorder is the shared do-nothing Recorder used when NewAuditKV is given a nil
// recorder. It records nothing and allocates nothing per call.
var NopRecorder Recorder = nopRecorder{}

// AuditKV is an interface-identical decorator over an abi.KVBackend: it implements
// every abi.KVBackend method as a timed forward to inner, recording per-op latency
// and a bytes-moved estimate into rec, then returning the inner result verbatim.
type AuditKV struct {
	inner abi.KVBackend
	rec   Recorder
}

// Compile-time proof AuditKV is a drop-in abi.KVBackend (interface-identical).
var _ abi.KVBackend = (*AuditKV)(nil)

// NewAuditKV wraps inner with per-op timing reported to rec. A nil rec defaults to
// NopRecorder, so the decorator is always safe to construct. The result is a
// drop-in abi.KVBackend: nothing about inner's observable behavior changes.
func NewAuditKV(inner abi.KVBackend, rec Recorder) *AuditKV {
	if rec == nil {
		rec = NopRecorder
	}
	return &AuditKV{inner: inner, rec: rec}
}

// Len forwards the live-length query. No span bytes move: bytes=0.
func (a *AuditKV) Len() int {
	start := time.Now()
	out := a.inner.Len()
	a.rec.Record(OpLen, time.Since(start), 0)
	return out
}

// Prefill forwards the prefill. Bytes = len(logits)*4 (the returned []float32).
func (a *AuditKV) Prefill(ids []int) []float32 {
	start := time.Now()
	out := a.inner.Prefill(ids)
	a.rec.Record(OpPrefill, time.Since(start), int64(len(out))*4)
	return out
}

// Evict forwards the eviction. Bytes = positions removed (the movement proxy).
func (a *AuditKV) Evict(from, n int) int {
	start := time.Now()
	removed := a.inner.Evict(from, n)
	a.rec.Record(OpEvict, time.Since(start), int64(removed))
	return removed
}

// ModelID forwards the model-id read. Bytes = len(id).
func (a *AuditKV) ModelID() string {
	start := time.Now()
	id := a.inner.ModelID()
	a.rec.Record(OpModelID, time.Since(start), int64(len(id)))
	return id
}

// StageSpan forwards the residency stage. Bytes = res.BytesMoved (authoritative).
func (a *AuditKV) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	start := time.Now()
	res, err := a.inner.StageSpan(ctx, digest, from, n)
	a.rec.Record(OpStageSpan, time.Since(start), res.BytesMoved)
	return res, err
}

// RestoreSpan forwards the residency restore. Bytes = res.BytesMoved (authoritative).
func (a *AuditKV) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	start := time.Now()
	res, err := a.inner.RestoreSpan(ctx, digest)
	a.rec.Record(OpRestoreSpan, time.Since(start), res.BytesMoved)
	return res, err
}
