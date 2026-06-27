package callavoid

// classgate.go — the in-lane half of #819: a per-tool-class economics gate over
// ProveMemo, the admission predicate the vDSO tier-2 content cache should consult before
// it admits a tool CLASS (Read / Grep / Glob / a custom tool) into the cache.
//
// Why a class gate and not just ProveMemo. ProveMemo proves one KEY's calibration; the
// vDSO admits by tool CLASS. A class is worth caching iff its representative calibration
// — the per-class mutation rate, validate cost, and capture cost a caller draws from
// internal/vcachecal-style OBSERVATION, never hand-asserted — proves. The gate is the
// thin per-class wrapper: it runs the same break-even arithmetic and returns an admit /
// decline decision plus the full MemoProof, so the seam can log WHY a class was declined.
//
// Default behaviour is byte-identical until opted in. This leaf only DECIDES; it admits
// nothing. The caller (the kernel.Reap seam, deferred below) consults Admit and, only
// when a flag is set, skips tier-2 capture for a refuted class. With the gate off, the
// cache behaves exactly as today.
//
// DEFERRED (#819, defer the kernel wiring): the actual gate placement at the vDSO tier-2
// admission seam — kernel.Reap, between the EvDispatch emit and eng.Complete — is NOT in
// this lane (it touches internal/kernel, and callavoid is tier-1 import-free). This file
// supplies the pure per-class decision the seam calls; the seam owns the flag, the
// EvDispatch/Complete ordering, and feeding it the observed (m, v, c) calibration.

// ClassMemoInput is one tool class's representative calibration for the admission gate.
// The cost fields mirror MemoInput exactly (same EXECUTION-EQUIVALENT units); Class is
// the tool name the decision is about, and Accesses defaults to a probe of 2 (the
// smallest reuse that can ever clear the gate) when unset, so a class is judged on its
// per-reuse economics (D = 1 - m - v - m·c) rather than on a single observed window's k.
type ClassMemoInput struct {
	Class        string  `json:"class"`         // the tool class this calibration describes (e.g. "Read").
	Accesses     int     `json:"accesses"`      // k: representative reuse count; <2 is treated as the break-even probe of 2.
	ValidateCost float64 `json:"validate_cost"` // v: per-reuse re-validation cost, in execution-equivalents (observed).
	MutationRate float64 `json:"mutation_rate"` // m: observed share of reuses whose key was invalidated by an intervening write, in [0,1].
	CaptureCost  float64 `json:"capture_cost"`  // c: one-time capture/store cost on each execution (observed).
}

// ClassGateDecision is the gate's verdict for one tool class: whether to admit it to the
// tier-2 cache, plus the full MemoProof so the decline reason is auditable at the seam.
type ClassGateDecision struct {
	Class string    `json:"class"`
	Admit bool      `json:"admit"` // true: cache this class (the economics prove). false: leave it on always-execute.
	Proof MemoProof `json:"proof"` // the break-even proof the decision is read off — Admit == (Proof.Status == ProofProven).
	Note  string    `json:"note"`  // a one-line human reason, echoing the proof's verdict.
}

// GateClass runs the break-even gate for one tool class and returns the admit/decline
// decision. A class is ADMITTED iff its observed calibration PROVES — i.e. the per-reuse
// net gain D is positive AND the representative reuse count clears the break-even that
// amortizes capture. A volatile class (high m), a non-fingerprintable one (validate as
// costly as execution), or a single-use one is DECLINED, so the cache never pays to store
// an entry whose economics refute. Pure and deterministic: no cache, no clock, no I/O.
func GateClass(in ClassMemoInput) ClassGateDecision {
	k := in.Accesses
	if k < 2 {
		k = 2 // judge a class on its per-reuse economics, not on a degenerate single-access window.
	}
	proof := ProveMemo(MemoInput{
		Accesses:     k,
		ValidateCost: in.ValidateCost,
		MutationRate: in.MutationRate,
		CaptureCost:  in.CaptureCost,
	})
	admit := proof.Status == ProofProven
	note := "decline: " + proof.Reason
	if admit {
		note = "admit: " + proof.Reason
	}
	return ClassGateDecision{
		Class: in.Class,
		Admit: admit,
		Proof: proof,
		Note:  note,
	}
}

// GateClasses gates a batch of tool classes in one call, returning a decision per class
// in input order. It is the shape the kernel.Reap seam (deferred) consults once per
// session to decide its tier-2 admission set: the classes whose observed economics prove
// are the only ones it captures. Determinism is preserved — same inputs, same decisions,
// same order.
func GateClasses(in []ClassMemoInput) []ClassGateDecision {
	out := make([]ClassGateDecision, len(in))
	for i, c := range in {
		out[i] = GateClass(c)
	}
	return out
}

// AdmittedClasses is the convenience projection the seam actually wants: the set of class
// names the gate admitted, in input order. A caller builds its tier-2 allow-set from this
// directly. Declined classes are simply absent — their default disposition is to stay on
// the always-execute path, byte-identical to the gate-off behaviour.
func AdmittedClasses(in []ClassMemoInput) []string {
	var admitted []string
	for _, d := range GateClasses(in) {
		if d.Admit {
			admitted = append(admitted, d.Class)
		}
	}
	return admitted
}
