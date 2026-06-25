// Package abi is the FROZEN wave-0 spine of fak (the Fused Agent Kernel).
//
// This is the ONE file every fleet worker imports and the ONE tree no worker may
// lease (dos-arbitrate denies a lease on internal/abi; dos-plan-price refuses any
// partition that collides with it). Once frozen it is ADDITIVE-ONLY: fields are
// only ever ADDED with zero-value defaults, never removed or repurposed. A golden
// conformance test (testdata/abi_v0.1.golden) fails any breaking change, turning
// the freeze into a machine-checked contract.
//
// DESIGN STANCE: a Linux-syscall-table-shaped stable spine. No subsystem name
// (AGT, DOS, vDSO, rung, steward, headroom, litellm) ever appears here. Every
// v0.1 subsystem AND every future idea attaches as a registered driver in its own
// package via the registries in registry.go — a new idea is a new package + one
// register() call + (optionally) one additive envelope field guarded by a
// Capability, NEVER an edit to this file.
//
// Three real precedents the openness is modeled on:
//   - Linux syscall table : stable numbered ops, never renumbered, append-only.
//   - io_uring opcode table: an unknown opcode returns -EINVAL, it does not crash.
//   - LSM stackable hooks  : new hooks register; the core walker is untouched.
//
// The FOUR seams below are the ones that, if missed at freeze time, force a
// fleet-wide recompile later. They are all present now, defaulted so v0.1 ignores
// them: (1) Verdict is an additive discriminated union, (2) payloads are
// addressable Refs not copied bytes, (3) sync Syscall is defined OVER async
// Submit/Reap, (4) a provisional speculation/txn lifecycle rides the envelope.
package abi

import "context"

// ---------------------------------------------------------------------------
// ABI VERSION + CAPABILITY NEGOTIATION (the ELF header)
// ---------------------------------------------------------------------------

// ABIMajor never bumps in v0 (a bump is a breaking change, which the golden test
// forbids). ABIMinor increments only for additive surface (a new reserved range,
// a new core-blessed capability).
const (
	ABIMajor = 0
	ABIMinor = 1
)

// Capability is a negotiated feature token (OPEN set). A driver advertises the
// caps it needs; Kernel.Negotiate intersects advertised-vs-registered, so a
// worker built before a capability existed never sees it and degrades to the
// conservative default. Examples: "async", "speculative", "zerocopy",
// "witness.v2", "trust.federated", "share.fleet".
type Capability string

// ---------------------------------------------------------------------------
// SEAM 2 — ADDRESSABLE PAYLOAD (the zero-copy + taint + share seam)
// ---------------------------------------------------------------------------

// Ref is an ADDRESSABLE handle to bytes, never the bytes themselves. v0.1 backs
// it with a content-addressed blob store (a copy); a later in-address-space impl
// (co-residing tool args/results with the KV cache — brainstorm 3.1a) is a
// Resolver/RegionBackend swap behind Capability "zerocopy". Callers only ever see
// Ref + Resolver, so the wire shape never changes.
//
// Ref also carries the provenance fields the cross-agent shared-result pool needs
// (Taint, Scope): a result is never shared more widely than its scope, and
// sharing a result shares its taint. Defaults (Tainted, ScopeAgent) mean "never
// shared, untrusted" — the fail-closed baseline.
type Ref struct {
	Kind   RefKind    // Inline | Blob | Region
	Digest string     // content hash: stable identity for the vDSO + provenance
	Inline []byte     // populated iff Kind==RefInline (small payloads)
	Handle uint64     // backend-opaque (CAS key / arena offset); meaningless w/o Resolver
	Len    int64      // payload length
	Taint  TaintLabel // Trusted | Tainted | Quarantined (default Tainted)
	Scope  ShareScope // Agent | Fleet | Tenant (default ScopeAgent)
}

// RefKind is the CLOSED discriminator for where a Ref's bytes live: inline,
// in a content-addressed blob store, or in an addressable region.
type RefKind uint8

const (
	RefInline RefKind = iota // bytes embedded in Inline
	RefBlob                  // bytes live in a content-addressed store (Handle = key)
	RefRegion                // bytes live in an addressable region (Handle = addr/offset)
)

// TaintLabel is a CLOSED, additive lattice (trusted < tainted < quarantined).
type TaintLabel uint8

const (
	TaintTainted     TaintLabel = iota // default: untrusted unless proven
	TaintTrusted                       // adjudicated trusted; shareable per Scope
	TaintQuarantined                   // held out of context by the MMU
)

// ShareScope is a CLOSED, additive isolation scope. Default ScopeAgent = private.
type ShareScope uint8

const (
	ScopeAgent  ShareScope = iota // private to one agent (fail-closed default)
	ScopeFleet                    // shareable across the fleet's trusted partition
	ScopeTenant                   // shareable within a tenant boundary
)

// Resolver is the OPEN backend behind every Ref. Swapping it (copy-CAS ->
// shared-arena zero-copy) is a backend change, not an ABI change. Registered via
// RegisterRegionBackend; the MMU's blob store is the v0.1 default.
type Resolver interface {
	Resolve(ctx context.Context, r Ref) ([]byte, error) // materialize on demand
	Put(ctx context.Context, b []byte) (Ref, error)
}

// ---------------------------------------------------------------------------
// SEAM 4 — PROVISIONAL LIFECYCLE (the speculative + transactional-context seam)
// ---------------------------------------------------------------------------

// SpeculationContext rides on every ToolCall. Zero value (Speculative=false,
// Epoch=0) means an ordinary committed call, so v0.1 ignores it entirely. A
// speculative call (brainstorm 2.6) sets Speculative=true and an Epoch; effects
// produced under a non-zero Epoch are PROVISIONAL until Promote/Rollback.
type SpeculationContext struct {
	Speculative bool
	Epoch       uint64 // speculation epoch id (0 = not speculative)
	ParentEpoch uint64 // the epoch this one branched from
}

// TxnID scopes a two-phase context admission (brainstorm 3.4 transactional
// context / KV checkpoint-rollback). 0 = auto-commit (no real transaction), so
// v0.1 ignores it. A non-zero TxnID means results are admitted to a scratch and
// only become visible on Promote(TxnID); Rollback(TxnID) discards them.
type TxnID uint64

// Outcome is the CLOSED resolution of a provisional call. Default OutcomeCommitted
// so a non-speculative, non-transactional call is always "committed".
type Outcome uint8

const (
	OutcomeCommitted  Outcome = iota // effects are durable (default)
	OutcomeSquashed                  // speculative branch discarded; effects retracted
	OutcomeRolledBack                // transaction rolled back; scratch discarded
)

// ProvisionalSink is implemented by any driver that produces retractable effects
// (the context-MMU is the canonical one). The Kernel drives Promote/Rollback when
// a speculation epoch or a transaction resolves. This makes "squash actually
// undoes the effect" a frozen cross-driver contract, not a discovered-at-
// integration gap. v0.1's MMU registers a sink whose Promote is a no-op append
// and whose Rollback drops the scratch.
type ProvisionalSink interface {
	Promote(ctx context.Context, txn TxnID, epoch uint64) error
	Rollback(ctx context.Context, txn TxnID, epoch uint64) error
}

// ---------------------------------------------------------------------------
// THE SYSCALL MESSAGE (frozen wire shape; additive-only)
// ---------------------------------------------------------------------------

// ToolCall is the request envelope. Op selects an entry in the OPEN operation
// table; Tool is the logical tool name (also the lease lane and a training
// token). Args is a handle, not bytes. Meta/Ext are OPEN maps — unknown keys MUST
// be ignored (forward-compat). Ext carries typed per-subsystem sidecar payloads
// keyed by a reserved ExtKey range.
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

// Result is the response envelope. Payload is a handle. Status is the immediate
// disposition; Outcome is the provisional-lifecycle resolution. Both are CLOSED
// tiny enums.
type Result struct {
	Call    *ToolCall
	Payload Ref
	Status  Status
	Outcome Outcome           // SEAM 4 (default OutcomeCommitted)
	Meta    map[string]string // OPEN; unknown keys ignored
	Ext     map[ExtKey]Ref    // OPEN typed sidecar
}

// Status is the CLOSED immediate disposition. StatusPending is the async seam:
// a non-blocking Submit returns Pending and the completion is Reap'd later.
type Status uint8

const (
	StatusOK      Status = iota // result is ready in Payload
	StatusError                 // call failed; Payload may carry an error blob
	StatusPending               // async: completion will arrive via Reap (SEAM 3)
)

// ---------------------------------------------------------------------------
// SEAM 1 — VERDICT: a CLOSED trainable union + an OPEN registered range
// ---------------------------------------------------------------------------

// VerdictKind is the closed/trainable adjudication vocabulary below
// VerdictReservedMax (the set the syscall-tuned small model emits and learns) and
// an OPEN registered range above it. A worker that doesn't know a registered kind
// resolves it via the kind's FallbackClass (fail-closed by default) and never
// panics.
type VerdictKind uint16

const (
	VerdictAllow          VerdictKind = iota // the CLOSED trainable set:
	VerdictDeny                              // a PROVABLE refusal (mirrors decide.go)
	VerdictTransform                         // rewrite Args (payload: TransformPayload)
	VerdictQuarantine                        // hold the RESULT out of context (MMU)
	VerdictRequireWitness                    // gate pending an independent read-back
	VerdictDefer                             // not adjudicable here; pass to next link
	VerdictIndeterminate  VerdictKind = 6    // undecided cheaply; climb or fail closed
	// 7..1023 reserved for additive CORE kinds. > VerdictReservedMax: registered.
	VerdictReservedMax VerdictKind = 1023
)

// Verdict is a DISCRIMINATED UNION, not a flat struct with "iff" comments: the
// Payload is keyed by Kind so a malformed combination (e.g. Kind==Deny with a
// transform payload) is UNREPRESENTABLE. This makes every well-formed sample
// type-valid by construction — the property that makes Verdict a clean training
// target for the syscall-tuned model.
type Verdict struct {
	Kind    VerdictKind
	Payload VerdictPayload    // nil for Allow/Deny/Defer; typed for the rest
	Reason  ReasonCode        // CLOSED registered refusal vocabulary (trainable)
	By      string            // which adjudicator decided (forensics, not dispatch)
	Meta    map[string]string // OPEN; ignored if unknown
}

// VerdictPayload is a sealed sum type. Each concrete payload is the typed body of
// exactly one Kind. New registered kinds supply their own payload type.
type VerdictPayload interface{ isVerdictPayload() }

// TransformPayload is the VerdictTransform body: NewArgs is the rewritten,
// adjudicator-approved Args the call proceeds with in place of the original.
type TransformPayload struct{ NewArgs Ref } // Kind==VerdictTransform
func (TransformPayload) isVerdictPayload()  {}

type QuarantinePayload struct{ PageOut bool } // Kind==VerdictQuarantine
func (QuarantinePayload) isVerdictPayload()   {}

type WitnessPayload struct{ Claim string } // Kind==VerdictRequireWitness
func (WitnessPayload) isVerdictPayload()   {}

// ReasonCode is the CLOSED, registered refusal vocabulary — the model's label
// space (mirrors DOS dos_refuse_reasons). Additive-only: a model trained on vN
// degrades gracefully on vN+1.
type ReasonCode uint16

// FallbackClass tells an UNAWARE worker how to treat a verdict kind it doesn't
// know. Fail-closed default (FallbackDeny) means an unknown kind can never
// silently widen authority.
type FallbackClass uint8

const (
	FallbackDeny  FallbackClass = iota // default: unknown kind => deny
	FallbackAllow                      // explicitly safe-to-ignore kinds
	FallbackDefer                      // pass to the next link
)

// ExtKey and OpCode are OPEN registered uint32s with reserved per-subsystem
// ranges (see registry.go). Reserved ranges keep KEYS disjoint at link time; the
// SEMANTICS of a well-known key are pinned by the typed sidecar that owns it.
type (
	ExtKey uint32
	OpCode uint32
)

// ---------------------------------------------------------------------------
// THE FROZEN HOOK INTERFACES (impls live in driver packages, never here)
// ---------------------------------------------------------------------------

// Adjudicator is one link in the LSM-style stackable PDP/PEP chain. It returns a
// Verdict; the kernel folds the chain by registered Rank (see RegisterVerdictKind
// / RegisterAdjudicator). The fold MIRRORS dos-preflake decide.go: a PROVABLE
// refusal returns VerdictDeny, an UNPROVABLE one returns VerdictDefer (advisory,
// fail-to-abstain). AGT (semantic PDP), the DOS lease PEP, every preflight rung,
// and future witness/federated-trust gates all stack here.
type Adjudicator interface {
	Adjudicate(ctx context.Context, c *ToolCall) Verdict
	Caps() []Capability
}

// FastPath is the vDSO seam: answer a call LOCALLY with no engine and no remote
// round-trip. Miss returns ok=false and the syscall proceeds normally. The 3 v0.1
// tiers (pure / content-addressed / static) are registered FastPaths; a new tier
// is a new registration.
type FastPath interface {
	Lookup(ctx context.Context, c *ToolCall) (r *Result, ok bool)
	Caps() []Capability
}

// Op is one entry in the OPEN operation table (the io_uring opcode analogue). A
// new operation (async submit/reap, speculative commit/squash) is a registered
// Op, never a new core method. Invoke receives the Kernel so an Op can drive
// provisional Promote/Rollback and consult resolver/registries.
type Op interface {
	Code() OpCode
	Invoke(ctx context.Context, k Kernel, c *ToolCall) (*Result, Verdict)
}

// Completion is the TYPED async completion (SEAM 3). Reap returns these — the
// completion-to-submission binding is a frozen contract, not an opaque cursor, so
// two independent async drivers cannot collide on the semantics of a shared key.
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

// Emitter observes every lifecycle transition (KPIs, rung labels, the RSI
// signal). Stewards, the meta-steward, KPI taps, and the self-labeling harvester
// all register here. The label-bearing path is TYPED (Event.Label) so the
// model's supervised labels never drift into untyped mush.
type Emitter interface{ Emit(ev Event) }

// Event is one lifecycle transition handed to an Emitter: the call, its verdict and
// result, an optional typed training Label, and OPEN non-label telemetry Fields.
type Event struct {
	Kind    EventKind
	Call    *ToolCall
	Verdict *Verdict
	Result  *Result
	Label   *LabelRow      // TYPED training signal (nil unless a label-bearing kind)
	Fields  map[string]any // OPEN; unknown ignored (non-label telemetry only)
}

// EventKind is the OPEN registered discriminator naming which lifecycle transition
// an Event reports (e.g. EvComplete).
type EventKind uint32

// LabelRow is the frozen, typed self-labeling signal of the pre-flight ladder
// (brainstorm 5.3): "passed cheap rung k, failed expensive rung k+1" is a labeled
// hard negative. Freezing the shape means the training-data generator is itself
// structured, so the syscall-tuned model's labels can't drift across drivers.
type LabelRow struct {
	CallHash    string
	RungPassed  int
	RungFailed  int // -1 if the call passed every rung
	Verdict     VerdictKind
	Reason      ReasonCode
	Speculative bool
}

// Kernel is the FROZEN chokepoint the whole fleet codes against — the ONLY core
// entrypoint. Sync Syscall is defined OVER async Submit/Reap (SEAM 3), so adding
// async later never splits the chokepoint: adjudication always happens at Submit.
type Kernel interface {
	// Submit adjudicates (folds the Adjudicator chain) and enqueues the call.
	// It returns immediately; a synchronous result arrives via Reap. The Verdict
	// is the admission decision (a denied call is never enqueued).
	Submit(ctx context.Context, c *ToolCall) (SubmissionHandle, Verdict)

	// Reap blocks for the completion of a specific submission.
	Reap(ctx context.Context, h SubmissionHandle) (*Result, error)

	// Syscall is the synchronous convenience: Submit then Reap. Every existing
	// caller uses this; async-aware callers use Submit/Reap directly.
	Syscall(ctx context.Context, c *ToolCall) (*Result, Verdict)

	// Resolver is the active Ref backend (copy-CAS in v0.1; zero-copy later).
	Resolver() Resolver

	// Negotiate intersects a caller's advertised caps with what's registered.
	Negotiate(advertised []Capability) []Capability
}
