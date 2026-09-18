package model

import "fmt"

// dense_residency_bound.go — the RUNTIME consumer for the declared bounded streamed-dense
// working set (#13253). ggufload.WithStreamedDenseQ4KWorkingSet(hostBytes) declares that only
// hostBytes of dense host retention may be charged at load time, but nothing in the
// load/materialization path read the declaration: a lazy dense k-quant materialized on demand
// and MEMOIZED its whole payload into kQuantTensor.raw, growing host anon-RSS past the
// declared budget until the serve was kernel-OOM-killed.
//
// This file makes the declaration a MODEL-LEVEL retention ceiling for memoized lazy dense
// materialization. A Model whose bound was never declared (bound == 0) allocates no ledger and
// behaves byte-for-byte as before; a Model with a declared bound shares ONE ledger across all
// of its lazy dense tensors, so the ceiling is over the model's retained dense working set, not
// per tensor.

// denseResidentLedger is the shared retention accounting for one Model's memoized lazy dense
// materialization. bound is the declared ceiling (0 = unbounded, and then the ledger is never
// created). retained is the running total of dense payload bytes actually retained by memoized
// lazy tensors of this model. It is pointer-shared by every lazy dense tensor so the sum is
// model-wide rather than per tensor.
//
// Fail-closed: a materialization whose bytes would push retained past bound panics by name
// (see kQuantTensor.ensureRawCPU) rather than silently exceeding the declared working set.
type denseResidentLedger struct {
	bound    int64
	retained int64
}

// denseLedger returns this Model's dense-residency ledger, lazily creating it on first use.
// A Model with no declared bound (denseResidentBoundBytes == 0) returns nil and allocates
// nothing, so the default-off path is byte-for-byte unchanged.
func (m *Model) denseLedger() *denseResidentLedger {
	if m == nil || m.denseResidentBoundBytes <= 0 {
		return nil
	}
	if m.denseResidentLedger == nil {
		m.denseResidentLedger = &denseResidentLedger{bound: m.denseResidentBoundBytes}
	}
	return m.denseResidentLedger
}

// SetDenseResidentBound declares the model-level ceiling (in bytes) on retained memoized lazy
// dense materialization. bytes <= 0 is the unbounded default; it is clamped to 0 with no
// ledger ever created, matching the WithStreamedDenseQ4KWorkingSet contract (where a negative
// budget is refused at option resolution, so a negative reaching here can only mean unbounded
// intent — it is never silently accepted as a real ceiling). A nil receiver or nil model is
// ignored.
func (b *QuantBuilder) SetDenseResidentBound(bytes int64) {
	if b == nil || b.m == nil {
		return
	}
	if bytes < 0 {
		// Unbounded intent, matching WithStreamedDenseQ4KWorkingSet's refusal of a negative
		// declared budget at resolution; no ledger is created for <= 0.
		bytes = 0
	}
	b.m.denseResidentBoundBytes = bytes
}

// chargeRetained is the ledger-side admission check for one inbound materialization of
// incoming bytes. A nil ledger (no bound declared) is a no-op and returns true. A ledger with
// bound <= 0 is unbounded (also a no-op). Otherwise the charge is refused — and the caller
// panics by name — when retained+incoming would exceed the declared bound; on admission the
// running total is advanced by incoming. retained never exceeds bound across admitted charges.
func (l *denseResidentLedger) chargeRetained(incoming int64) bool {
	if l == nil || l.bound <= 0 {
		return true
	}
	if l.retained+incoming > l.bound {
		return false
	}
	l.retained += incoming
	return true
}

// denseResidentBoundPanic builds the named fail-closed diagnostic for a refused dense
// materialization. It matches the #13216/#13202 legibility style: a named panic the operator
// can route, not a silent overrun or a silent-zeros answer.
func denseResidentBoundPanic(op string, bound, incoming, retained int64) string {
	return fmt.Sprintf("model: dense resident bound %d exceeded: %s would retain %d more "+
		"(%d already retained) — the declared streamed-dense working set must bound retained "+
		"materialized dense bytes (#13253).", bound, op, incoming, retained)
}
