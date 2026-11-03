// Package abi is the FROZEN wave-0 spine of fak: the one tree no worker may
// lease, imported by every fleet worker. It is ADDITIVE-ONLY — fields are only
// ever added with zero-value defaults, never removed or repurposed — and a golden
// conformance test (testdata/abi_v0.1.golden) fails any breaking change. No
// subsystem name ever appears here: every idea attaches as a registered driver
// in its own package via registry.go, never as an edit to this file.
package abi

import "context"

// ABIMajor never bumps in v0 (a bump is breaking, which the golden test forbids); ABIMinor increments only for additive surface.
const (
	ABIMajor = 0
	ABIMinor = 1
)

// Capability is a negotiated feature token (OPEN set); Kernel.Negotiate intersects advertised vs registered caps, so a pre-capability worker degrades to the conservative default.
type Capability string

// Ref is an ADDRESSABLE handle to bytes, never the bytes themselves. v0.1 backs
// it with a content-addressed blob store; a later in-address-space impl is a
// Resolver/RegionBackend swap behind Capability "zerocopy", so the wire shape
// never changes. Taint and Scope carry sharing provenance: a result is never
// shared more widely than its scope, and sharing a result shares its taint.
// Defaults (Tainted, ScopeAgent) mean "never shared, untrusted".
type Ref struct {
	Kind   RefKind    // Inline | Blob | Region
	Digest string     // content hash: stable identity for the vDSO + provenance
	Inline []byte     // populated iff Kind==RefInline (small payloads)
	Handle uint64     // backend-opaque (CAS key / arena offset); meaningless w/o Resolver
	Len    int64      // payload length
	Taint  TaintLabel // Trusted | Tainted | Quarantined (default Tainted)
	Scope  ShareScope // Agent | Fleet | Tenant (default ScopeAgent)
}

// RefKind is the CLOSED discriminator for where a Ref's bytes live: inline, blob-store, or addressable region.
type RefKind uint8

// RefKind picks how a result payload travels over the wire: RefInline embeds bytes in Inline; RefBlob keeps them in a content-addressed store (Handle = key); RefRegion in an addressable region (Handle = addr/offset).
const (
	RefInline RefKind = iota
	RefBlob
	RefRegion
)

// TaintLabel is a CLOSED, additive lattice (trusted < tainted < quarantined).
type TaintLabel uint8

// TaintLabel: TaintTainted is the default (untrusted unless proven, never shared); TaintTrusted is adjudicated and shareable per Scope; TaintQuarantined is held out of context by the MMU.
const (
	TaintTainted TaintLabel = iota
	TaintTrusted
	TaintQuarantined
)

// ShareScope is a CLOSED, additive isolation scope. Default ScopeAgent = private.
type ShareScope uint8

// ShareScope bounds how widely a result may be shared: ScopeAgent keeps it private to one agent (the fail-closed default); ScopeFleet shares across the fleet's trusted partition; ScopeTenant within a tenant boundary.
const (
	ScopeAgent ShareScope = iota
	ScopeFleet
	ScopeTenant
)

// Resolver is the OPEN backend behind every Ref; swapping it (copy-CAS -> zero-copy arena) is a backend change, not an ABI change.
type Resolver interface {
	Resolve(ctx context.Context, r Ref) ([]byte, error) // materialize on demand
	Put(ctx context.Context, b []byte) (Ref, error)
}

// SpeculationContext rides on every ToolCall; zero value means an ordinary committed call, and effects under a non-zero Epoch are PROVISIONAL until Promote/Rollback.
type SpeculationContext struct {
	Speculative bool
	Epoch       uint64 // speculation epoch id (0 = not speculative)
	ParentEpoch uint64 // the epoch this one branched from
}

// TxnID scopes two-phase context admission (KV checkpoint-rollback); 0 = auto-commit, and results admitted under a non-zero TxnID become visible only on Promote.
type TxnID uint64

// Outcome is the CLOSED resolution of a provisional call; default OutcomeCommitted makes ordinary calls "committed".
type Outcome uint8

const (
	// OutcomeCommitted is the default resolution: effects are durable, as for an ordinary committed call.
	OutcomeCommitted Outcome = iota
	// OutcomeSquashed discards a speculative branch; its provisional effects are retracted via Rollback.
	OutcomeSquashed
	// OutcomeRolledBack rolls back a transaction; its scratch state is discarded and admitted results stay hidden.
	OutcomeRolledBack
)

// ProvisionalSink is implemented by any driver that produces retractable effects
// (the context-MMU is canonical); the Kernel drives Promote/Rollback when a
// speculation epoch or transaction resolves, making "squash actually undoes the effect" a frozen cross-driver contract.
type ProvisionalSink interface {
	Promote(ctx context.Context, txn TxnID, epoch uint64) error
	Rollback(ctx context.Context, txn TxnID, epoch uint64) error
}

// ToolCall is the request envelope. Op selects an entry in the OPEN operation
// table; Tool is the logical tool name (also the lease lane and a training
// token). Meta/Ext are OPEN maps — unknown keys MUST be ignored (forward-compat); Ext carries typed per-subsystem sidecars keyed by a reserved ExtKey range.
type ToolCall struct {
	Op      OpCode
	Tool    string
	Engine  string // optional per-call engine route; empty => kernel default
	Args    Ref
	Caps    []Capability       // caller-advertised caps (negotiation)
	Spec    SpeculationContext // SEAM 4 (zero value = ordinary call)
	Txn     TxnID              // SEAM 4 (0 = auto-commit)
	SeqNo   uint64             // submission identity (async + speculative)
	TraceID string
	Meta    map[string]string // OPEN; unknown keys ignored
	Ext     map[ExtKey]Ref    // OPEN typed sidecar (spec id, async cursor, label rows...)
}

// Result is the response envelope: Payload is a handle, Status the immediate disposition, Outcome the provisional-lifecycle resolution (both CLOSED enums).
type Result struct {
	Call    *ToolCall
	Payload Ref
	Status  Status
	Outcome Outcome           // SEAM 4 (default OutcomeCommitted)
	Meta    map[string]string // OPEN; unknown keys ignored
	Ext     map[ExtKey]Ref    // OPEN typed sidecar
}

// Status is the CLOSED immediate disposition; StatusPending is the async seam — a non-blocking Submit returns Pending and the completion is Reap'd later.
type Status uint8

const (
	// StatusOK means the result is ready now, in Result.Payload.
	StatusOK Status = iota
	// StatusError means the call failed; Payload may carry an error blob for inspection.
	StatusError
	// StatusPending is the async seam: Submit returned without running the call; completion arrives later via Reap.
	StatusPending
)

// VerdictKind is the closed trainable adjudication vocabulary below
// VerdictReservedMax plus an OPEN registered range above it; an unaware worker
// resolves an unknown kind via its FallbackClass (deny by default) and never panics.
type VerdictKind uint16

const (
	// VerdictAllow admits the call unchanged and carries no payload; it opens the CLOSED trainable set.
	VerdictAllow VerdictKind = iota
	// VerdictDeny is a PROVABLE refusal (mirrors decide.go) and carries no payload.
	VerdictDeny
	// VerdictTransform rewrites the call's Args; its payload is TransformPayload.
	VerdictTransform
	// VerdictQuarantine holds the RESULT out of context via the MMU; its payload is QuarantinePayload.
	VerdictQuarantine
	// VerdictRequireWitness gates the call pending an independent read-back; its payload is WitnessPayload.
	VerdictRequireWitness
	// VerdictDefer means this link cannot adjudicate; pass the call to the next link, no payload.
	VerdictDefer
	// VerdictIndeterminate means the call was left undecided cheaply; the ladder climbs a rung or fails closed.
	VerdictIndeterminate VerdictKind = 6
	// 7..1023 reserved for additive CORE kinds. > VerdictReservedMax: registered.
	VerdictReservedMax VerdictKind = 1023
)

// Verdict is a DISCRIMINATED UNION: Payload is keyed by Kind, so a malformed combination
// (Kind==Deny with a transform payload) is UNREPRESENTABLE — every well-formed sample
// is type-valid by construction, which is what makes Verdict a clean training target.
type Verdict struct {
	Kind        VerdictKind
	Payload     VerdictPayload    // nil for Allow/Deny/Defer; typed for the rest
	Reason      ReasonCode        // CLOSED registered refusal vocabulary (trainable)
	By          string            // which adjudicator decided (forensics, not dispatch)
	Meta        map[string]string // OPEN; ignored if unknown
	Disposition string            // explicit loopback routing disposition; overrides Reason fallback
}

// VerdictPayload is a sealed sum type: each concrete payload is the typed body
// of exactly one Kind.
type VerdictPayload interface{ isVerdictPayload() }

// TransformPayload is the VerdictTransform body: NewArgs is the rewritten,
// adjudicator-approved Args the call proceeds with. NewTool is an OPTIONAL substitute
// tool name (e.g. MCP `git_push` -> Bash `fak sync push`); empty means same-tool
// arg-only rewrite, the back-compatible default. Appended after NewArgs to preserve the additive-only shape freeze.
type TransformPayload struct {
	NewArgs Ref
	NewTool string
}                                          // Kind==VerdictTransform
func (TransformPayload) isVerdictPayload() {}

// QuarantinePayload is the VerdictQuarantine body: PageOut asks the MMU to page the result out of context.
type QuarantinePayload struct{ PageOut bool }

func (QuarantinePayload) isVerdictPayload() {}

// WitnessPayload is the VerdictRequireWitness body: Claim names the statement to verify by an independent read-back.
type WitnessPayload struct{ Claim string }

func (WitnessPayload) isVerdictPayload() {}

// ReasonCode is the CLOSED registered refusal vocabulary — the model's label space; additive-only, so a model trained on vN degrades gracefully on vN+1.
type ReasonCode uint16

// FallbackClass tells an UNAWARE worker how to treat a verdict kind it doesn't
// know; the default (FallbackDeny) ensures an unknown kind can never silently widen authority.
type FallbackClass uint8

// FallbackClass tells an unaware worker how to treat a verdict kind it does not know: FallbackDeny (the default) denies it; FallbackAllow marks kinds explicitly safe to ignore; FallbackDefer passes them to the next link.
const (
	FallbackDeny FallbackClass = iota
	FallbackAllow
	FallbackDefer
)

// ExtKey and OpCode are OPEN registered uint32s with reserved per-subsystem ranges
// (registry.go); a well-known key's semantics are pinned by the typed sidecar that owns it.
type (
	ExtKey uint32
	OpCode uint32
)

// Adjudicator is one link in the LSM-style stackable PDP/PEP chain; the kernel folds
// the chain by registered Rank. The fold mirrors dos-preflake decide.go: a PROVABLE
// refusal returns VerdictDeny, an UNPROVABLE one VerdictDefer (advisory, fail-to-abstain).
// Semantic PDPs, the DOS lease PEP, preflight rungs, and future trust gates all stack here.
type Adjudicator interface {
	Adjudicate(ctx context.Context, c *ToolCall) Verdict
	Caps() []Capability
}

// FastPath is the vDSO seam: answer a call LOCALLY with no engine and no remote
// round-trip; a miss returns ok=false and the syscall proceeds normally. New tiers are new registrations.
type FastPath interface {
	Lookup(ctx context.Context, c *ToolCall) (r *Result, ok bool)
	Caps() []Capability
}

// Op is one entry in the OPEN operation table; a new operation is a registered Op,
// never a new core method. Invoke receives the Kernel so an Op can drive provisional Promote/Rollback.
type Op interface {
	Code() OpCode
	Invoke(ctx context.Context, k Kernel, c *ToolCall) (*Result, Verdict)
}

// Completion is the TYPED async completion Reap returns; the completion-to-submission
// binding is a frozen contract, not an opaque cursor, so two async drivers cannot collide on a shared key.
type Completion struct {
	Handle SubmissionHandle
	Result *Result
}

// SubmissionHandle is the typed identity of a non-blocking Submit.
type SubmissionHandle struct {
	Seq    uint64 // == ToolCall.SeqNo
	Queue  uint32 // which completion queue (multi-engine / multi-queue)
	Opaque uint64 // driver-private correlation token
}

// Emitter observes every lifecycle transition (KPIs, rung labels, the RSI signal).
// The label-bearing path is TYPED (Event.Label) so the model's supervised labels never drift into untyped mush.
type Emitter interface{ Emit(ev Event) }

// Event is one lifecycle transition handed to an Emitter: the call, its verdict and result, an optional typed Label, and OPEN telemetry Fields.
type Event struct {
	Kind    EventKind
	Call    *ToolCall
	Verdict *Verdict
	Result  *Result
	Label   *LabelRow      // TYPED training signal (nil unless a label-bearing kind)
	Fields  map[string]any // OPEN; unknown ignored (non-label telemetry only)
}

// EventKind is the OPEN registered discriminator naming which lifecycle transition an Event reports.
type EventKind uint32

// LabelRow is the frozen, typed self-labeling signal of the pre-flight ladder:
// "passed cheap rung k, failed expensive rung k+1" is a labeled hard negative.
// Freezing the shape keeps the training-data generator structured, so labels can't drift across drivers.
type LabelRow struct {
	CallHash    string
	RungPassed  int
	RungFailed  int // -1 if the call passed every rung
	Verdict     VerdictKind
	Reason      ReasonCode
	Speculative bool
}

// Kernel is the FROZEN chokepoint the whole fleet codes against — the ONLY core
// entrypoint. Sync Syscall is defined OVER async Submit/Reap, so adjudication always happens at Submit.
type Kernel interface {
	// Submit adjudicates (folds the Adjudicator chain) and enqueues the call; the returned Verdict is the admission decision (a denied call is never enqueued).
	Submit(ctx context.Context, c *ToolCall) (SubmissionHandle, Verdict)

	// Reap blocks for the completion of a specific submission.
	Reap(ctx context.Context, h SubmissionHandle) (*Result, error)

	// Syscall is the synchronous convenience: Submit then Reap.
	Syscall(ctx context.Context, c *ToolCall) (*Result, Verdict)

	// Resolver is the active Ref backend (copy-CAS in v0.1; zero-copy later).
	Resolver() Resolver

	// Negotiate intersects a caller's advertised caps with what's registered.
	Negotiate(advertised []Capability) []Capability
}
