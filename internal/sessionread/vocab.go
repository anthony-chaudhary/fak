package sessionread

// vocab.go — the read/query/observe-op VOCABULARY SPINE of the session read plane
// (epic #4176, spine #4191): the one first-class, closed type every session READ op
// belongs to, and the uniform contract for what a read DISCLOSES, under WHOSE right,
// with WHAT evidence, and HOW it refuses when illegal. It is the outbound twin of
// internal/sessionctl/vocab.go, which does the same for the operator CONTROL (write)
// plane.
//
// Why this exists. The read surface shipped fragmented across a dozen seams — the
// managed-context report (fak_context_value / GET /v1/fak/ctxvalue), the dropped-span
// index and restore (fak_context_spans / fak_context_restore), the drive-state reads
// (GET /v1/fak/session/{id}, /v1/fak/sessions, /v1/fak/session/changes), the coherence
// feed (fak_changes / GET /v1/fak/changes), the durable audit journal (GET
// /v1/fak/events), the per-trace taint high-water (GET /v1/fak/trace/{trace_id}), the
// no-self-report run digest (dos_status), and the resume/heal self-observe
// (fak_resume_history). Each independently re-invented "who may read it", "what does it
// disclose", "is the answer a live observation or an attested artifact", and "how does
// it refuse when the read is illegal". This file NAMES each shipped read op with its
// four fixed properties, grounded in the real refusal the seam already makes, so the
// grammar is defined once. It does not re-implement any seam.
//
// The four fixed properties (per op):
//
//  1. Capability — the read-right required to submit the op: read-self (a caller
//     reading its OWN trace's state, the gateway loopback default) versus read-fleet
//     (a read that crosses the per-principal boundary — all sessions, any run-id, the
//     cross-agent bus). read-fleet is NOT yet enforced anywhere; the spine records the
//     requirement so the C1 scoping gate (#4192) has a name to adopt, exactly as
//     sessionctl records CapOperatorControl ahead of its gate.
//  2. Disclosure — the projection level the op returns: metadata (accounting/counts/
//     descriptors, no payload bytes), redacted (bounded, taint-screened excerpts), or
//     full (verbatim payload bytes cross the boundary). Exactly one op — context-restore
//     — is full; it is the seam where the taint screen is load-bearing.
//  3. Evidence — the OBSERVED-vs-WITNESSED qualifier on the answer: OBSERVED is a live
//     in-memory reading, true at read-time but not durably attested; WITNESSED is backed
//     by an artifact or ledger fak authored (the restored bytes fak dropped, the
//     hash-chained journal, the folded run ledger), non-forgeable. The outbound mirror of
//     the control plane's witness-of-applied: a read states which of the two it is so a
//     consumer never treats a live snapshot as attested.
//  4. RefusalReasons — the closed refusal token(s) the op surfaces when the read is
//     illegal: READ_UNKNOWN_TRACE (a miss on the named trace/session/run-id),
//     READ_SCOPE_DENIED (the principal may not read across this boundary — the C1 gate),
//     READ_TAINT_WITHHELD (a sealed/tombstoned span would leak, refused).
//
// The taint-safe-outbound invariant — no sealed/quarantined span bytes ever cross the
// boundary — is realized today by the full-disclosure op: context-restore honors the
// ctxplan seal/tombstone gate and refuses with READ_TAINT_WITHHELD. C1 (#4192)
// generalizes that screen across every op; the not-yet-shipped read seams (a durable
// content-addressed store, a live transcript query, a transcript-event subscribe, an
// MCP resource surface, the supervisor observe loop) are deliberately NOT registered
// here — a new op registers its row (and its tokens) when its child lands, exactly as
// sessionctl leaves the not-yet-shipped add-constraint op absent.

import "slices"

// ReadOp is the closed set of session read/query/observe ops at HEAD. It is a string
// enum so the wire/tool verb and the vocabulary key are the same token. A new op MUST
// add its constant here and its spec to vocabulary — the completeness test names any op
// missing either.
type ReadOp string

const (
	// OpContextValue reads the managed-context value report for one session — the
	// resident-window tokens, the turns-to-compaction forecast, the session lifecycle
	// rollup, and the closed step-advice verdict (fak_context_value, GET /v1/fak/ctxvalue).
	OpContextValue ReadOp = "context-value"
	// OpContextSpans enumerates the restorable content-address handles a trace holds —
	// one SAFE descriptor row per dropped span, never the full bytes (fak_context_spans).
	OpContextSpans ReadOp = "context-spans"
	// OpContextRestore pages one dropped span back in by its content-address handle,
	// returning the verbatim bytes fak authored the drop of (fak_context_restore).
	OpContextRestore ReadOp = "context-restore"
	// OpSessionState reads a run's drive state — run-state/budget/priority/pace — for one
	// session or all of them (GET /v1/fak/session/{id}, GET /v1/fak/sessions).
	OpSessionState ReadOp = "session-state"
	// OpSessionChanges tails the drive-state revision stream — a cursor-drained feed of
	// every session-table Rev bump (GET /v1/fak/session/changes).
	OpSessionChanges ReadOp = "session-changes"
	// OpCoherenceChanges tails the cross-agent coherence feed — the bounded, cursor-drained
	// "what changed" bus across agents (fak_changes, GET /v1/fak/changes).
	OpCoherenceChanges ReadOp = "coherence-changes"
	// OpAuditEvents tails the durable hash-chained audit journal — the attested record of
	// adjudicated syscalls (GET /v1/fak/events).
	OpAuditEvents ReadOp = "audit-events"
	// OpTraceTaint reads one trace's taint high-water mark — the outbound trust ceiling a
	// trace has reached (GET /v1/fak/trace/{trace_id}).
	OpTraceTaint ReadOp = "trace-taint"
	// OpRunDigest reads a run's folded status — liveness, ledger-VERIFIED progress, region,
	// resume — with NO self-report field by construction (dos_status).
	OpRunDigest ReadOp = "run-digest"
	// OpResumeHistory self-observes a session's resume/heal history — attempts, resume
	// state, retry-blocked, earned budget, operator-settled (fak_resume_history).
	OpResumeHistory ReadOp = "resume-history"
)

// Capability is the closed set of read-rights an op requires. The spine NAMES the
// requirement per op; it does NOT itself enforce. Today the gateway serves both rights
// unauthenticated, scoping a bare read to the caller's own trace via the default-trace
// resolution — the per-principal floor is exactly what C1 (#4192) wires. The rows record
// the right each seam SHOULD check when the gate lands.
type Capability string

const (
	// CapReadSelf is the read-right for a caller observing its OWN trace's state: the
	// gateway loopback default, where an omitted trace resolves to the wrapped session
	// itself (context-value, context-spans, context-restore, trace-taint, resume-history).
	CapReadSelf Capability = "read-self"
	// CapReadFleet is the read-right for a read that crosses the per-principal boundary —
	// every session's drive state, any run-id's digest, the cross-agent bus, the durable
	// journal (session-state, session-changes, coherence-changes, audit-events, run-digest).
	// It is NOT yet enforced: these seams serve any caller today, which is precisely the
	// worst-first exposure C1 (#4192) closes. The spine records the requirement so the
	// future gate has a name to adopt — the read-side twin of sessionctl.CapOperatorControl.
	CapReadFleet Capability = "read-fleet"
)

// Disclosure is the closed vocabulary for WHAT a read projects — the outbound analog of
// the control plane's boundary grammar. Every read op returns exactly one level.
type Disclosure string

const (
	// DisclosureMetadata — accounting, counts, verdicts, or descriptors only; no payload
	// bytes cross the boundary (context-value, context-spans, session-state,
	// session-changes, trace-taint, run-digest, resume-history).
	DisclosureMetadata Disclosure = "metadata"
	// DisclosureRedacted — bounded, taint-screened excerpts of content (the coherence
	// feed's change descriptors, the audit journal's adjudicated rows), never a raw span.
	DisclosureRedacted Disclosure = "redacted"
	// DisclosureFull — verbatim payload bytes cross the boundary (context-restore returns
	// the whole dropped turn). The one op where the taint screen is load-bearing: a sealed
	// or tombstoned span must refuse rather than disclose.
	DisclosureFull Disclosure = "full"
)

// Evidence is the closed OBSERVED-vs-WITNESSED qualifier on a read's answer — the
// taint/evidence grammar generalized across the plane. Each op declares which it carries
// so a consumer never treats a live snapshot as an attested artifact.
type Evidence string

const (
	// EvidenceObserved — a live in-memory reading, true at read-time but not durably
	// attested and stale the instant after (context-value's resident window, the live
	// drive-state and rev streams, the live coherence bus, the current taint high-water).
	EvidenceObserved Evidence = "OBSERVED"
	// EvidenceWitnessed — backed by an artifact or ledger fak authored: the verbatim bytes
	// fak dropped (context-restore), the index of those drops (context-spans), the
	// hash-chained audit journal (audit-events), the folded run ledger (run-digest), the
	// durable launch ledger (resume-history). Non-forgeable.
	EvidenceWitnessed Evidence = "WITNESSED"
)

// ReadSpec is one registered read op with its four fixed properties. It is pure data — a
// declarative row a consumer (a read route, an audit, an MCP tool surface, the C1 scoping
// gate) can adopt as the op's contract. The spine DECLARES the contract; no production
// caller consults it yet (wiring each consumer is per-op follow-on). Its fidelity is
// pinned by this package's completeness test.
type ReadSpec struct {
	// Op is the closed op token (also the wire/tool verb).
	Op ReadOp `json:"op"`
	// Capability is the read-right required to submit the op.
	Capability Capability `json:"capability"`
	// Disclosure is the projection level the op returns.
	Disclosure Disclosure `json:"disclosure"`
	// Evidence is the OBSERVED-vs-WITNESSED qualifier on the op's answer.
	Evidence Evidence `json:"evidence"`
	// RefusalReasons is the closed refusal token(s) the op surfaces for an
	// illegal-for-state read. Non-empty for every op.
	RefusalReasons []string `json:"refusal_reasons"`
	// Summary is the one-line human/audit description of the op's precise behavior.
	Summary string `json:"summary"`
}

// The closed read-refusal vocabulary. Unlike the control plane — whose tokens ground in
// session.ControlRefusalTokens and the abi send floor — the read plane had no closed
// refusal set before this spine, so it is DEFINED here (the read-side twin of
// internal/session/ctlrefuse.go). Every token is SCREAMING_SNAKE, disjoint from the
// tool-refusal (abi) and control-write (session) vocabularies, and used by at least one
// registered op — the grounded-tokens test pins all three invariants.
const (
	// ReasonReadUnknownTrace — the named trace, session id, run-id, or ledger is not
	// addressable: the "never had it" miss (ctxrestore's ErrRestoreMiss, an unknown session
	// id, a bad run-id, an unresolvable resume ledger). Distinct from a scope refusal.
	ReasonReadUnknownTrace = "READ_UNKNOWN_TRACE"
	// ReasonReadScopeDenied — the principal may not read across this boundary: the
	// cross-principal / fleet refusal the C1 gate (#4192) will emit for a read-fleet op
	// submitted without the fleet right. Registered ahead of the gate so the seam that
	// enforces it has a name to raise — not yet emitted by any live seam.
	ReasonReadScopeDenied = "READ_SCOPE_DENIED"
	// ReasonReadTaintWithheld — a sealed (trust-quarantined) or tombstoned (context-control)
	// span would leak through the read, refused. Live today on context-restore, which honors
	// the ctxplan seal/tombstone gate (ErrRestoreRefused wrapping ctxplan.ErrSealed /
	// ErrTombstoned); the taint-safe-outbound invariant C1 generalizes across the plane.
	ReasonReadTaintWithheld = "READ_TAINT_WITHHELD"
)

// vocabulary is the closed registry, in stable op order. Each op's refusal tokens are the
// closed read-refusal constants above — the miss, the scope denial, and the taint
// withhold — assigned to the ops that make (or, for the C1 gate, will make) them.
var vocabulary = []ReadSpec{
	{
		Op:             OpContextValue,
		Capability:     CapReadSelf,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceObserved,
		RefusalReasons: []string{ReasonReadUnknownTrace},
		Summary:        "managed-context value report for one session: resident tokens, turns-to-compaction forecast, lifecycle rollup, and the closed step-advice verdict; a live observation of the window",
	},
	{
		Op:             OpContextSpans,
		Capability:     CapReadSelf,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceWitnessed,
		RefusalReasons: []string{ReasonReadUnknownTrace},
		Summary:        "enumerates the restorable content-address handles a trace holds — one SAFE descriptor row per dropped span, never the bytes; an index of fak's own witnessed drops",
	},
	{
		Op:             OpContextRestore,
		Capability:     CapReadSelf,
		Disclosure:     DisclosureFull,
		Evidence:       EvidenceWitnessed,
		RefusalReasons: []string{ReasonReadTaintWithheld, ReasonReadUnknownTrace},
		Summary:        "pages one dropped span back in by its handle, returning the verbatim bytes fak authored the drop of; a sealed/tombstoned span refuses READ_TAINT_WITHHELD, an unknown id misses READ_UNKNOWN_TRACE",
	},
	{
		Op:             OpSessionState,
		Capability:     CapReadFleet,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceObserved,
		RefusalReasons: []string{ReasonReadScopeDenied, ReasonReadUnknownTrace},
		Summary:        "run drive-state — run-state/budget/priority/pace — for one session or all; a live table snapshot that crosses the per-principal boundary (the C1 scoping target)",
	},
	{
		Op:             OpSessionChanges,
		Capability:     CapReadFleet,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceObserved,
		RefusalReasons: []string{ReasonReadScopeDenied},
		Summary:        "drive-state revision stream: a cursor-drained tail of every session-table Rev bump; a live cross-session feed",
	},
	{
		Op:             OpCoherenceChanges,
		Capability:     CapReadFleet,
		Disclosure:     DisclosureRedacted,
		Evidence:       EvidenceObserved,
		RefusalReasons: []string{ReasonReadScopeDenied},
		Summary:        "cross-agent coherence feed: the bounded, cursor-drained 'what changed' bus across agents; live change descriptors, not raw spans",
	},
	{
		Op:             OpAuditEvents,
		Capability:     CapReadFleet,
		Disclosure:     DisclosureRedacted,
		Evidence:       EvidenceWitnessed,
		RefusalReasons: []string{ReasonReadScopeDenied},
		Summary:        "durable hash-chained audit journal: the attested tail of adjudicated syscalls; witnessed rows, not payload bytes",
	},
	{
		Op:             OpTraceTaint,
		Capability:     CapReadSelf,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceObserved,
		RefusalReasons: []string{ReasonReadUnknownTrace},
		Summary:        "one trace's taint high-water mark — the outbound trust ceiling it has reached; a live reading of the trace's accumulated taint",
	},
	{
		Op:             OpRunDigest,
		Capability:     CapReadFleet,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceWitnessed,
		RefusalReasons: []string{ReasonReadScopeDenied, ReasonReadUnknownTrace},
		Summary:        "a run's folded status — liveness, ledger-VERIFIED progress, region, resume — with NO self-report field by construction; folded from the run's witnessed ledger",
	},
	{
		Op:             OpResumeHistory,
		Capability:     CapReadSelf,
		Disclosure:     DisclosureMetadata,
		Evidence:       EvidenceWitnessed,
		RefusalReasons: []string{ReasonReadUnknownTrace},
		Summary:        "self-observes a session's resume/heal history — attempts, resume state, retry-blocked, earned budget, operator-settled; folded from the durable launch ledger",
	},
}

// specByOp is the lookup index, built once from vocabulary.
var specByOp = func() map[ReadOp]ReadSpec {
	m := make(map[ReadOp]ReadSpec, len(vocabulary))
	for _, s := range vocabulary {
		m[s.Op] = s
	}
	return m
}()

// readRefusalTokens is the closed read-refusal vocabulary in stable order, defined by
// this spine. Kept as an unexported source of truth so ReadRefusalTokens hands out copies.
var readRefusalTokens = []string{
	ReasonReadUnknownTrace,
	ReasonReadScopeDenied,
	ReasonReadTaintWithheld,
}

// Vocabulary returns the closed read-op registry in stable order — the declarative
// contract for the session read plane. The result is a deep copy (the nested
// RefusalReasons slices are cloned), so a caller cannot mutate the registry through it.
func Vocabulary() []ReadSpec {
	out := make([]ReadSpec, len(vocabulary))
	copy(out, vocabulary)
	for i := range out {
		out[i].RefusalReasons = slices.Clone(vocabulary[i].RefusalReasons)
	}
	return out
}

// Ops returns the closed read-op set in stable order.
func Ops() []ReadOp {
	out := make([]ReadOp, len(vocabulary))
	for i, s := range vocabulary {
		out[i] = s.Op
	}
	return out
}

// Spec returns the registered spec for op and whether it is a known read op. The returned
// spec is a deep copy (RefusalReasons cloned) so a caller cannot mutate the registry
// through the shared slice header.
func Spec(op ReadOp) (ReadSpec, bool) {
	s, ok := specByOp[op]
	s.RefusalReasons = slices.Clone(s.RefusalReasons)
	return s, ok
}

// ReadRefusalTokens returns the closed read-refusal vocabulary in stable order — the
// read-side twin of session.ControlRefusalTokens. The result is a copy so a caller cannot
// mutate the registry through it.
func ReadRefusalTokens() []string {
	return slices.Clone(readRefusalTokens)
}
