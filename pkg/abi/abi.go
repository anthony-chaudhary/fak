// Package abi is the IMPORTABLE vendor surface of fak's frozen ABI.
//
// fak's spine lives in internal/abi (the "Linux-syscall-table-shaped stable
// spine"): a frozen, additive-only contract that every in-tree subsystem codes
// against. Go's internal/ rule seals that package — no module outside
// github.com/anthony-chaudhary/fak may import it — which is correct for the
// kernel's own consumers but blocks the ONE audience the freeze exists to serve:
// an OUT-OF-TREE driver author who wants to claim a vendor OpCode / VerdictKind
// and attach their own Adjudicator, ResultAdmitter, engine, or backend.
//
// This package is that surface, and ONLY that surface. Every name below is a Go
// TYPE ALIAS or a var/const re-export of a symbol in internal/abi, so a value or
// interface from pkg/abi is IDENTICAL (same underlying type) to its internal/abi
// counterpart: a driver implementing pkgabi.Adjudicator already satisfies
// internalabi.Adjudicator because they are the same type. The kernel walks its
// internal registries; a vendor registers through abi.Register* here and the
// effect is the same registration.
//
// Drivers import THIS package, never internal/abi. It re-exports only the
// vendor/driver-facing surface — the interfaces a driver implements, the
// Register* functions a driver calls, the value types it constructs, the closed
// constants it references, and the two observability helpers (ReasonName,
// FoldRank) it legitimately needs. The kernel-internal read side (Adjudicators(),
// ActiveResolver(), the snapshot machinery, ResetForTest, mask helpers) is
// deliberately NOT re-exported: it is the host's, not a driver's.
//
// The frozen single source of truth is internal/abi. This shim adds no behavior
// and holds no state; it is a stable, zero-cost re-export. See issue #454.
package abi

import internalabi "github.com/anthony-chaudhary/fak/internal/abi"

// ---------------------------------------------------------------------------
// Driver interfaces — a vendor IMPLEMENTS these and registers the impl.
// ---------------------------------------------------------------------------

type (
	// EngineDriver runs a tool call against an inference engine.
	EngineDriver = internalabi.EngineDriver
	// ResultAdmitter is the write-time context-MMU seam (post-tool result gate).
	ResultAdmitter = internalabi.ResultAdmitter
	// Adjudicator is one link in the stackable PDP/PEP chain (pre-call gate).
	Adjudicator = internalabi.Adjudicator
	// FastPath is the vDSO seam: answer a call locally with no engine.
	FastPath = internalabi.FastPath
	// RegionBackend provides the Resolver behind every Ref (zero-copy seam).
	RegionBackend = internalabi.RegionBackend
	// PageOutBackend is the context-MMU's swappable page-out codec.
	PageOutBackend = internalabi.PageOutBackend
	// KVBackend is the attention-cache the KV-MMU enforces a quarantine on.
	KVBackend = internalabi.KVBackend
	// KVBackendFactory adapts a session-like value into a KVBackend.
	KVBackendFactory = internalabi.KVBackendFactory
	// KVResidency is the typed result of a KVBackend residency transfer (off-box L3).
	KVResidency = internalabi.KVResidency
	// KVResidencyOutcome is the ok|MISS|FAULT trichotomy a residency transfer returns.
	KVResidencyOutcome = internalabi.KVResidencyOutcome
	// WitnessResolver backs the require-witness verdict.
	WitnessResolver = internalabi.WitnessResolver
	// SemanticScreen is the local-model-on-the-wire advisory seam.
	SemanticScreen = internalabi.SemanticScreen
	// Steward is one cheap single-invariant validator.
	Steward = internalabi.Steward
	// Emitter observes every lifecycle transition.
	Emitter = internalabi.Emitter
	// Resolver is the OPEN backend behind every Ref (returned by RegionBackend).
	Resolver = internalabi.Resolver
	// ProvisionalSink produces retractable effects (speculation / txn).
	ProvisionalSink = internalabi.ProvisionalSink
	// Kernel is the frozen chokepoint an Op's Invoke receives.
	Kernel = internalabi.Kernel

	// EventSubscriber is OPTIONAL: an Emitter may also implement it to scope itself
	// to specific EventKinds.
	EventSubscriber = internalabi.EventSubscriber
	// CallScope is OPTIONAL: a fold driver may declare the tool names it acts on.
	CallScope = internalabi.CallScope
	// CASPinner is OPTIONAL: a Resolver may implement it when its store is bounded.
	CASPinner = internalabi.CASPinner
)

// ---------------------------------------------------------------------------
// Value types — a vendor CONSTRUCTS these.
// ---------------------------------------------------------------------------

type (
	ToolCall           = internalabi.ToolCall
	Result             = internalabi.Result
	Ref                = internalabi.Ref
	Completion         = internalabi.Completion
	SubmissionHandle   = internalabi.SubmissionHandle
	Verdict            = internalabi.Verdict
	VerdictPayload     = internalabi.VerdictPayload
	TransformPayload   = internalabi.TransformPayload
	QuarantinePayload  = internalabi.QuarantinePayload
	WitnessPayload     = internalabi.WitnessPayload
	SpeculationContext = internalabi.SpeculationContext
	Event              = internalabi.Event
	LabelRow           = internalabi.LabelRow
	ScreenAdvice       = internalabi.ScreenAdvice
	Range              = internalabi.Range

	// Op is one entry in the OPEN operation table (vendor claims a code).
	Op = internalabi.Op

	// Scalar / enum types.
	OpCode            = internalabi.OpCode
	ExtKey            = internalabi.ExtKey
	VerdictKind       = internalabi.VerdictKind
	ReasonCode        = internalabi.ReasonCode
	Capability        = internalabi.Capability
	EventKind         = internalabi.EventKind
	ShareScope        = internalabi.ShareScope
	FallbackClass     = internalabi.FallbackClass
	Status            = internalabi.Status
	Outcome           = internalabi.Outcome
	RefKind           = internalabi.RefKind
	TaintLabel        = internalabi.TaintLabel
	TxnID             = internalabi.TxnID
	ScreenDisposition = internalabi.ScreenDisposition
	WitnessOutcome    = internalabi.WitnessOutcome
)

// ---------------------------------------------------------------------------
// Register* — the ONLY way a vendor extends the kernel (called from init()).
// ---------------------------------------------------------------------------

var (
	RegisterAdjudicator     = internalabi.RegisterAdjudicator
	RegisterResultAdmitter  = internalabi.RegisterResultAdmitter
	RegisterFastPath        = internalabi.RegisterFastPath
	RegisterOp              = internalabi.RegisterOp
	RegisterVerdictKind     = internalabi.RegisterVerdictKind
	RegisterReason          = internalabi.RegisterReason
	RegisterCapability      = internalabi.RegisterCapability
	RegisterEmitter         = internalabi.RegisterEmitter
	RegisterEngine          = internalabi.RegisterEngine
	RegisterRegionBackend   = internalabi.RegisterRegionBackend
	RegisterKVBackend       = internalabi.RegisterKVBackend
	RegisterPageOutBackend  = internalabi.RegisterPageOutBackend
	RegisterWitnessResolver = internalabi.RegisterWitnessResolver
	RegisterSteward         = internalabi.RegisterSteward
	RegisterProvisionalSink = internalabi.RegisterProvisionalSink
	RegisterSemanticScreen  = internalabi.RegisterSemanticScreen
)

// ---------------------------------------------------------------------------
// Observability helpers a driver legitimately needs.
// ---------------------------------------------------------------------------

var (
	// ReasonName resolves a ReasonCode to its stable name (core + registered).
	ReasonName = internalabi.ReasonName
	// FoldRank returns a VerdictKind's restrictiveness-lattice position.
	FoldRank = internalabi.FoldRank
)

// ---------------------------------------------------------------------------
// Constants — re-declared with the aliased type (so abi.VerdictAllow has type
// abi.VerdictKind == internalabi.VerdictKind).
// ---------------------------------------------------------------------------

// ABI version (the ELF-header negotiation handle).
const (
	ABIMajor = internalabi.ABIMajor
	ABIMinor = internalabi.ABIMinor
)

// VerdictKind closed core set (+ the reserved-max boundary).
const (
	VerdictAllow          = internalabi.VerdictAllow
	VerdictDeny           = internalabi.VerdictDeny
	VerdictTransform      = internalabi.VerdictTransform
	VerdictQuarantine     = internalabi.VerdictQuarantine
	VerdictRequireWitness = internalabi.VerdictRequireWitness
	VerdictDefer          = internalabi.VerdictDefer
	VerdictReservedMax    = internalabi.VerdictReservedMax
)

// ReasonCode closed core vocabulary (+ the core-max boundary).
const (
	ReasonNone           = internalabi.ReasonNone
	ReasonDefaultDeny    = internalabi.ReasonDefaultDeny
	ReasonPolicyBlock    = internalabi.ReasonPolicyBlock
	ReasonSelfModify     = internalabi.ReasonSelfModify
	ReasonLeaseHeld      = internalabi.ReasonLeaseHeld
	ReasonTrustViolation = internalabi.ReasonTrustViolation
	ReasonMalformed      = internalabi.ReasonMalformed
	ReasonMisroute       = internalabi.ReasonMisroute
	ReasonRateLimited    = internalabi.ReasonRateLimited
	ReasonSecretExfil    = internalabi.ReasonSecretExfil
	ReasonUnwitnessed    = internalabi.ReasonUnwitnessed
	ReasonOversize       = internalabi.ReasonOversize
	ReasonUnknownTool    = internalabi.ReasonUnknownTool
	ReasonCoreMax        = internalabi.ReasonCoreMax
)

// RefKind discriminator.
const (
	RefInline = internalabi.RefInline
	RefBlob   = internalabi.RefBlob
	RefRegion = internalabi.RefRegion
)

// KVResidencyOutcome trichotomy for an off-box KV residency transfer.
const (
	KVResidencyUnknown = internalabi.KVResidencyUnknown
	KVResidencyOK      = internalabi.KVResidencyOK
	KVResidencyMiss    = internalabi.KVResidencyMiss
	KVResidencyFault   = internalabi.KVResidencyFault
)

// TaintLabel lattice.
const (
	TaintTainted     = internalabi.TaintTainted
	TaintTrusted     = internalabi.TaintTrusted
	TaintQuarantined = internalabi.TaintQuarantined
)

// ShareScope isolation scopes.
const (
	ScopeAgent  = internalabi.ScopeAgent
	ScopeFleet  = internalabi.ScopeFleet
	ScopeTenant = internalabi.ScopeTenant
)

// Outcome provisional-lifecycle resolutions.
const (
	OutcomeCommitted  = internalabi.OutcomeCommitted
	OutcomeSquashed   = internalabi.OutcomeSquashed
	OutcomeRolledBack = internalabi.OutcomeRolledBack
)

// Status immediate dispositions.
const (
	StatusOK      = internalabi.StatusOK
	StatusError   = internalabi.StatusError
	StatusPending = internalabi.StatusPending
)

// FallbackClass — how an unaware worker treats an unknown verdict kind.
const (
	FallbackDeny  = internalabi.FallbackDeny
	FallbackAllow = internalabi.FallbackAllow
	FallbackDefer = internalabi.FallbackDefer
)

// EventKind closed core lifecycle vocabulary.
const (
	EvSubmit     = internalabi.EvSubmit
	EvDecide     = internalabi.EvDecide
	EvDeny       = internalabi.EvDeny
	EvDispatch   = internalabi.EvDispatch
	EvComplete   = internalabi.EvComplete
	EvQuarantine = internalabi.EvQuarantine
	EvVDSOHit    = internalabi.EvVDSOHit
	EvResultDeny = internalabi.EvResultDeny
	EvRungLabel  = internalabi.EvRungLabel
)

// ScreenDisposition advisory enum.
const (
	ScreenAllow      = internalabi.ScreenAllow
	ScreenQuarantine = internalabi.ScreenQuarantine
	ScreenDigest     = internalabi.ScreenDigest
)

// WitnessOutcome — a WitnessResolver's verdict on a claimed effect.
const (
	WitnessAbstain   = internalabi.WitnessAbstain
	WitnessConfirmed = internalabi.WitnessConfirmed
	WitnessRefuted   = internalabi.WitnessRefuted
)

// ---------------------------------------------------------------------------
// Reserved ranges (Range values, not constants) — a vendor draws its OpCode /
// ExtKey / VerdictKind / EventKind numbers from its *Vendor block so it can
// never collide with a core or another vendor's number.
// ---------------------------------------------------------------------------

var (
	// OpCode blocks.
	OpsCore   = internalabi.OpsCore
	OpsSpec   = internalabi.OpsSpec
	OpsAsync  = internalabi.OpsAsync
	OpsVendor = internalabi.OpsVendor

	// ExtKey blocks.
	ExtSpec   = internalabi.ExtSpec
	ExtAsync  = internalabi.ExtAsync
	ExtLabel  = internalabi.ExtLabel
	ExtTrust  = internalabi.ExtTrust
	ExtVendor = internalabi.ExtVendor

	// VerdictKind vendor block (above VerdictReservedMax).
	VerdictsVendor = internalabi.VerdictsVendor

	// EventKind blocks.
	EventsCore   = internalabi.EventsCore
	EventsKPI    = internalabi.EventsKPI
	EventsLabel  = internalabi.EventsLabel
	EventsVendor = internalabi.EventsVendor
)
