package sessionread

import "slices"

// ReadOp identifies a closed session read or observe operation.
type ReadOp string

const (
	// OpContextValue reads managed-context window metrics and advice.
	OpContextValue ReadOp = "context-value"
	// OpContextSpans lists content-addressed dropped span descriptors.
	OpContextSpans ReadOp = "context-spans"
	// OpContextRestore pages back dropped span payload bytes.
	OpContextRestore ReadOp = "context-restore"
	// OpSessionState reads operator drive state for sessions.
	OpSessionState ReadOp = "session-state"
	// OpSessionChanges streams session drive-state revision updates.
	OpSessionChanges ReadOp = "session-changes"
	// OpCoherenceChanges streams cross-agent change descriptors.
	OpCoherenceChanges ReadOp = "coherence-changes"
	// OpAuditEvents streams attested audit records of adjudicated calls.
	OpAuditEvents ReadOp = "audit-events"
	// OpTraceTaint reads the outbound taint ceiling reached by a trace.
	OpTraceTaint ReadOp = "trace-taint"
	// OpRunDigest reads folded liveness and verified progress for a run.
	OpRunDigest ReadOp = "run-digest"
	// OpResumeHistory reads session resume and heal lifecycle history.
	OpResumeHistory ReadOp = "resume-history"
)

// Capability represents the authorization scope required to submit a read operation.
type Capability string

const (
	// CapReadSelf permits reading caller-owned trace state.
	CapReadSelf Capability = "read-self"
	// CapReadFleet permits reading across session and principal boundaries.
	CapReadFleet Capability = "read-fleet"
)

// Disclosure defines the data projection depth returned by a read operation.
type Disclosure string

const (
	// DisclosureMetadata returns accounting and descriptor metrics without payloads.
	DisclosureMetadata Disclosure = "metadata"
	// DisclosureRedacted returns bounded, taint-screened content excerpts.
	DisclosureRedacted Disclosure = "redacted"
	// DisclosureFull returns verbatim payload bytes across the boundary.
	DisclosureFull Disclosure = "full"
)

// Evidence qualifies whether a read result is a live observation or durable attestation.
type Evidence string

const (
	// EvidenceObserved indicates a live in-memory reading.
	EvidenceObserved Evidence = "OBSERVED"
	// EvidenceWitnessed indicates output backed by a durable artifact or signed ledger.
	EvidenceWitnessed Evidence = "WITNESSED"
)

// ReadSpec declares the properties and refusal reasons for a registered read operation.
type ReadSpec struct {
	Op             ReadOp     `json:"op"`
	Capability     Capability `json:"capability"`
	Disclosure     Disclosure `json:"disclosure"`
	Evidence       Evidence   `json:"evidence"`
	RefusalReasons []string   `json:"refusal_reasons"`
	Summary        string     `json:"summary"`
}

// Closed read-refusal reason codes.
const (
	// ReasonReadUnknownTrace indicates the requested trace or session is not found.
	ReasonReadUnknownTrace = "READ_UNKNOWN_TRACE"
	// ReasonReadScopeDenied indicates the principal lacks permission for cross-principal reads.
	ReasonReadScopeDenied = "READ_SCOPE_DENIED"
	// ReasonReadTaintWithheld indicates data withheld due to trust quarantine or tombstone.
	ReasonReadTaintWithheld = "READ_TAINT_WITHHELD"
)

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
		RefusalReasons: []string{ReasonReadScopeDenied, ReasonReadUnknownTrace},
		Summary:        "enumerates the restorable content-address handles a trace holds — one SAFE descriptor row per dropped span, never the bytes; an index of fak's own witnessed drops; a cross-principal enumeration refuses READ_SCOPE_DENIED (the C1 scope floor)",
	},
	{
		Op:             OpContextRestore,
		Capability:     CapReadSelf,
		Disclosure:     DisclosureFull,
		Evidence:       EvidenceWitnessed,
		RefusalReasons: []string{ReasonReadScopeDenied, ReasonReadTaintWithheld, ReasonReadUnknownTrace},
		Summary:        "pages one dropped span back in by its handle, returning the verbatim bytes fak authored the drop of; a cross-principal read refuses READ_SCOPE_DENIED (the C1 scope floor), a sealed/tombstoned span refuses READ_TAINT_WITHHELD, an unknown id misses READ_UNKNOWN_TRACE",
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

var specByOp = func() map[ReadOp]ReadSpec {
	m := make(map[ReadOp]ReadSpec, len(vocabulary))
	for _, s := range vocabulary {
		m[s.Op] = s
	}
	return m
}()

var readRefusalTokens = []string{
	ReasonReadUnknownTrace,
	ReasonReadScopeDenied,
	ReasonReadTaintWithheld,
}

// Vocabulary returns all registered read operation specifications.
func Vocabulary() []ReadSpec {
	out := make([]ReadSpec, len(vocabulary))
	copy(out, vocabulary)
	for i := range out {
		out[i].RefusalReasons = slices.Clone(vocabulary[i].RefusalReasons)
	}
	return out
}

// Ops returns all registered read operations in stable order.
func Ops() []ReadOp {
	out := make([]ReadOp, len(vocabulary))
	for i, s := range vocabulary {
		out[i] = s.Op
	}
	return out
}

// Spec returns the registered specification for op, or false if unknown.
func Spec(op ReadOp) (ReadSpec, bool) {
	s, ok := specByOp[op]
	s.RefusalReasons = slices.Clone(s.RefusalReasons)
	return s, ok
}

// ReadRefusalTokens returns the closed list of read refusal reason codes.
func ReadRefusalTokens() []string {
	return slices.Clone(readRefusalTokens)
}
