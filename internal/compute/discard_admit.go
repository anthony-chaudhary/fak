package compute

// discard_admit.go — DISCARD-AWARE ADMISSION: the scheduler-side decision that declines
// to compute a forward pass already known to be thrown away (issue #808).
//
// WHY THIS EXISTS — the concrete tension it resolves. When the kernel cancels a tool,
// supersedes a turn, or refutes a sub-agent branch, the in-flight (or still-queued)
// inference for that work is going to be IGNORED. But the cancellation lives one layer
// above the batch scheduler and never crosses the boundary, so today the scheduler holds
// the batch slot to completion: it prefills the suffix, runs the target every decode step,
// and only frees the KV slot when the request is finally cancelled from above — after the
// compute is already spent. The sharpest claim of the inference-front-end lens is that
// **the cheapest forward pass is the one never dispatched** (docs/notes/, 2026-06-25):
// speculative decoding still runs the target every step and prefix caching still prefills
// the suffix — only the layer that ISSUED the cancellation can decline to compute, and
// that decision cannot be made below the kernel.
//
// WHAT THIS FILE IS. The pure, deterministic, host-tractable DECISION the scheduler runs
// on the throughput-relevant projection of the kernel's per-turn intent — the strongest
// hint in session.TurnIntent (#807), `will_discard`, carried here as a plain bool so the
// tier-1 compute layer consumes the signal WITHOUT importing the session layer that
// produces it (the same shape as batchsched.ComposeBatches taking bare `lengths []int`).
// Given a sequence's discard signal and where it sits in the scheduler's lifecycle, it
// returns one of three verdicts:
//
//   AdmitProceed  — fail-open: no known-discard opinion. Schedule per the GPU-visible
//                   policy exactly as before this decision existed.
//   AdmitDrop     — the sequence is still QUEUED and is known-discard: never admit it.
//                   Its forward pass is the one never dispatched.
//   AdmitPreempt  — the sequence is already RUNNING and is known-discard: preempt it and
//                   free its KV slot (the mechanism the native scheduler already owns —
//                   internal/modelengine schedLane.Cancel -> finish reclaims the session).
//
// THE FENCES (inherited from #807 / #805, and load-bearing here).
//   - Advisory, never trust. A hint can be wrong (a turn the kernel expected to discard
//     keeps mattering). So the decision FAILS OPEN: a candidate with no positive
//     will-discard signal always returns AdmitProceed. An absent or stale hint is never
//     mistaken for a discard.
//   - Idempotent and fail-safe. A missed cancellation just wastes the slot it would have
//     freed (status quo), never corrupts; re-deciding an already-dropped candidate yields
//     the same verdict. This is a throughput/cost lever ONLY — a discarded result was going
//     to be ignored anyway, so there is NO correctness surface. The decision must never
//     gate correctness.
//   - We act ONLY on the KNOWN-discard signal, never on the weaker "may be discarded"
//     speculative hint. A speculative branch can still be needed; dropping it would lose
//     work, which is outside this kernel's fail-safe contract. IsSpeculative deprioritises
//     prefill (a batch-composition concern), it does not authorise a drop here.
//
// HONEST SCOPE — what this is NOT. This is the DECISION, not its live wiring. The real
// preemptive continuous-batching scheduler that would CALL it on every admit/step is the
// deferred sibling work (#36/#135/#550, self-hosted-fleet only); the shipped native
// scheduler (internal/modelengine/nativesched.go) is explicitly a no-preemption,
// no-admission-control shape proof. So this file closes the decidable, host-free core of
// #808 — the policy a scheduler invokes — the same way #807 shipped the TurnIntent type
// ahead of its first reader and batchsched.go shipped the composition policy ahead of the
// live serve path. It moves no KV and runs no model; it depends only on its inputs, so it
// is byte-deterministic across machines — the repo's house form for a scheduling policy
// that must not drift with hardware.

// AdmitPhase says where a sequence sits in the scheduler's lifecycle when the discard
// signal arrives — the discriminator between "never admit" and "preempt".
type AdmitPhase uint8

const (
	// PhaseQueued: the sequence is in the admit queue and holds no KV slot yet. A
	// known-discard verdict here is AdmitDrop — the prefill is never dispatched.
	PhaseQueued AdmitPhase = iota
	// PhaseRunning: the sequence has been admitted and holds a KV slot, in-flight. A
	// known-discard verdict here is AdmitPreempt — free the slot now.
	PhaseRunning
)

// String renders the phase for logs and the audit surface.
func (p AdmitPhase) String() string {
	switch p {
	case PhaseRunning:
		return "running"
	case PhaseQueued:
		return "queued"
	default:
		return "queued" // unknown phases are treated as not-yet-admitted, conservatively
	}
}

// AdmitVerdict is the typed outcome of the discard-aware admission decision.
type AdmitVerdict uint8

const (
	// AdmitProceed is the fail-open verdict: no known-discard opinion, so schedule per the
	// GPU-visible policy. It is the zero value, so a default-constructed decision proceeds.
	AdmitProceed AdmitVerdict = iota
	// AdmitDrop: a still-queued, known-discard sequence — never admit it.
	AdmitDrop
	// AdmitPreempt: an in-flight, known-discard sequence — preempt it and free its KV slot.
	AdmitPreempt
)

// String renders the verdict for logs and refusal/observability messages.
func (v AdmitVerdict) String() string {
	switch v {
	case AdmitDrop:
		return "drop"
	case AdmitPreempt:
		return "preempt"
	default:
		return "proceed"
	}
}

// DiscardCandidate is one sequence the scheduler is deciding about. It is the
// throughput-relevant projection of the sequence's state plus the kernel's known-discard
// signal — intentionally pure data so a scheduler can build it from what it already holds,
// without this tier-1 package importing the session/scheduler layers above it.
type DiscardCandidate struct {
	// Phase is where the sequence sits when the signal arrives (queued vs running).
	Phase AdmitPhase
	// WillDiscard is the kernel's KNOWN-discard signal for this sequence — projected from
	// session.TurnIntent.WillDiscard (#807): a cancelled tool, a superseded turn, or a
	// refuted sub-agent branch. False (the zero value) means "no opinion" -> fail open.
	WillDiscard bool
	// Reason is an optional human label for the audit log ("tool-cancelled",
	// "turn-superseded", "branch-refuted"). It is never parsed by policy code.
	Reason string
	// PrefillRows is the prompt token-rows this candidate would prefill (queued) — the
	// forward-pass compute a drop never dispatches. Non-negative; 0 when unknown.
	PrefillRows int
	// KVSlots is the KV slots this candidate holds while running — the slots a preempt
	// reclaims. Non-negative; 0 when queued or unknown.
	KVSlots int
}

// DecideDiscardAdmission returns the verdict for a single candidate. It FAILS OPEN: a
// candidate with no known-discard signal always proceeds, so wiring this into a scheduler
// never refuses a sequence that the historical path would have run. A known-discard
// candidate drops while queued and is preempted while running.
func DecideDiscardAdmission(c DiscardCandidate) AdmitVerdict {
	if !c.WillDiscard {
		return AdmitProceed
	}
	if c.Phase == PhaseRunning {
		return AdmitPreempt
	}
	// PhaseQueued and any unknown phase: treat as not-yet-admitted. Dropping a sequence
	// that is in fact running is a no-op on the admit queue (it holds no queue slot), so
	// the conservative default only ever wastes the reclaim it would have done — the
	// fail-safe posture, never a corruption.
	return AdmitDrop
}

// AdmitDecision is one candidate's verdict plus its original index, so a caller can map the
// plan back onto its own queue/lane slice without re-deriving order.
type AdmitDecision struct {
	Index   int
	Verdict AdmitVerdict
}

// DiscardAdmissionStats summarises a plan for observability and for the witness — the
// concrete throughput/cost win the decision realises.
type DiscardAdmissionStats struct {
	Candidates       int // total candidates decided
	Dropped          int // queued, known-discard: never admitted
	Preempted        int // running, known-discard: KV slot freed
	Proceeded        int // no known-discard opinion: scheduled normally
	SlotsFreed       int // Σ KVSlots over preempted candidates — the reclaimed KV slots
	PrefillRowsSaved int // Σ PrefillRows over dropped candidates — the forward-pass rows never dispatched
}

// PlanDiscardAdmission decides every candidate and rolls the verdicts into per-candidate
// decisions plus aggregate stats. The decisions are index-aligned and 1:1 with the input
// (every candidate is decided exactly once). It is a pure fold over DecideDiscardAdmission,
// so it is deterministic for a given input and carries the same fail-open / fail-safe
// contract. A nil/empty input yields a nil decision slice and a zero stats.
func PlanDiscardAdmission(cands []DiscardCandidate) ([]AdmitDecision, DiscardAdmissionStats) {
	if len(cands) == 0 {
		return nil, DiscardAdmissionStats{}
	}
	decisions := make([]AdmitDecision, len(cands))
	stats := DiscardAdmissionStats{Candidates: len(cands)}
	for i, c := range cands {
		v := DecideDiscardAdmission(c)
		decisions[i] = AdmitDecision{Index: i, Verdict: v}
		switch v {
		case AdmitDrop:
			stats.Dropped++
			if c.PrefillRows > 0 {
				stats.PrefillRowsSaved += c.PrefillRows
			}
		case AdmitPreempt:
			stats.Preempted++
			if c.KVSlots > 0 {
				stats.SlotsFreed += c.KVSlots
			}
		default:
			stats.Proceeded++
		}
	}
	return decisions, stats
}
