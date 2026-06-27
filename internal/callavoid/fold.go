package callavoid

// fold.go — the in-lane half of #818: map a finished session's kernel call-path
// tallies onto a callavoid.Tally so Account can report the realized amplification for
// a REAL run, not just a synthetic input.
//
// Layering. callavoid is tier-1 — it imports nothing internal — so it cannot reach up
// to internal/kernel for the live Counters type. Instead it defines a structural
// MIRROR of the kernel's counter shape here (Counters below) and the caller does the
// trivial field copy from kernel.Counters into it (a downward import from the tier-4
// guard/gateway caller into this leaf). The field names mirror internal/kernel.Counters
// on purpose so that copy is one obvious, total line:
//
//	callavoid.Counters{
//	    EngineCalls: int(c.EngineCalls),
//	    VDSOHits:    int(c.VDSOHits),
//	    Transforms:  int(c.Transforms),
//	    Denies:      int(c.Denies),
//	}
//
// DEFERRED (#818, peer-broken lane): the cmd/fak `fak guard` exit-summary fold that
// reads a live guard session's kernel.Counters (or gateway.AdjudicationSummary) and
// renders Account is NOT in this lane — cmd/fak does not compile against the current
// peer-dirty trunk. This leaf supplies the pure mapping the caller folds; the caller
// owns the print + the --json field + the reconcile-with-/metrics assertion.

// Counters is the tier-1 structural mirror of internal/kernel.Counters — the subset of
// the kernel's call-path tallies the amplification headline consumes. The caller copies
// the live kernel.Counters into this (a plain field-for-field assignment, no behaviour),
// keeping callavoid import-free. Counts are int here (the kernel's are int64); a guard
// session never overflows int, and Account guards every field non-negative regardless.
type Counters struct {
	EngineCalls int `json:"engine_calls"` // real engine dispatches            -> Tally.Execute.
	VDSOHits    int `json:"vdso_hits"`    // calls served from the vDSO        -> Tally.MemoHit.
	Transforms  int `json:"transforms"`   // malformed calls repaired in-syscall -> Tally.Repair.
	Denies      int `json:"denies"`       // adjudicator fast-rejects          -> Tally.HardDeny (see note).
}

// TallyFromCounters maps a session's kernel Counters onto a Tally for Account. It is the
// canonical realization of the mapping documented on Tally:
//
//	Execute   <- Counters.EngineCalls
//	MemoHit   <- Counters.VDSOHits
//	Repair    <- Counters.Transforms
//	HardDeny  <- Counters.Denies
//
// Every deny maps to HardDeny — the CONSERVATIVE default, because the live kernel does
// not yet emit the productive-deny / hard-deny split (#820): an un-witnessed deny is
// credited NOTHING (HardDeny is symmetric in Account). A caller that HAS witnessed
// productive-deny fan-out for the session feeds it through TallyFromCountersWitnessed,
// which moves exactly the witnessed denies out of HardDeny and into realized credit, so
// the two never double-count.
func TallyFromCounters(c Counters) Tally {
	return Tally{
		Execute:  nonNegInt(c.EngineCalls),
		MemoHit:  nonNegInt(c.VDSOHits),
		Repair:   nonNegInt(c.Transforms),
		HardDeny: nonNegInt(c.Denies),
	}
}

// TallyFromCountersWitnessed is TallyFromCounters with a set of WITNESSED productive
// denies folded in. Each witnessed redirect was structurally one of the session's denies,
// so it is MOVED out of the raw HardDeny count (HardDeny -= len(witnessed)) and carried
// as realized-credit fan-out — never added on top, which would double-count the deny as
// both a hard deny and a productive one. If more witnessed denies are supplied than the
// counters recorded (a caller bug or a stale witness), HardDeny floors at zero and the
// surplus witnessed denies are still credited from their own enumerated variants — the
// realized credit comes from the witness, not from the count it was netted against.
func TallyFromCountersWitnessed(c Counters, witnessed []WitnessedRedirect) Tally {
	t := TallyFromCounters(c)
	t.HardDeny = nonNegInt(t.HardDeny - len(witnessed))
	t.WitnessedRedirects = witnessed
	return t
}
