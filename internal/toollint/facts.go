package toollint

// facts.go — the linter's unit of analysis. A ToolFacts is the STATIC description
// of one registered tool, assembled (collect.go) from the live registries and,
// when present, a hint classifier. The rules in lint.go are pure functions over a
// []ToolFacts; nothing here imports the kernel except the value types, so a host
// can hand-build ToolFacts for a tool surface the kernel hasn't booted.

// Kind is HOW the kernel will serve a tool — which decides what static
// inconsistencies even matter for it. It is derived from the fast-path registries:
// a tool the vDSO registered as pure is KindPure, one with a canned static answer
// is KindStatic, anything else is engine-routed.
type Kind uint8

const (
	// KindUnknown: not on any fast path and no explicit route declared — the kernel
	// will route it to an engine by default. The hint/schema rules still apply.
	KindUnknown Kind = iota
	// KindPure: registered via vdso.RegisterPure (tier-1). Reachable ONLY when its
	// hints satisfy readOnly+idempotent+!destructive (TL002 checks that).
	KindPure
	// KindStatic: registered via vdso.RegisterStatic (tier-3). Served
	// UNCONDITIONALLY by Lookup — no hint or destructive gate (TL003 checks that).
	KindStatic
	// KindEngine: explicitly engine-routed; no fast path is expected.
	KindEngine
)

func (k Kind) String() string {
	switch k {
	case KindPure:
		return "pure"
	case KindStatic:
		return "static"
	case KindEngine:
		return "engine"
	default:
		return "unknown"
	}
}

// Hints is the declared annotation triple the kernel re-checks at call time. The
// Declared* flags record whether the hint was SET at all: an UNSET hint is not a
// lie, so a rule that fires on a contradictory or dead hint must require the hint
// to have been declared, never treat a zero value as a claim. A facts source that
// has no hint classifier (FromKernel) leaves every Declared* false, and the
// hint-shaped rules stay silent.
type Hints struct {
	ReadOnly    bool
	Idempotent  bool
	Destructive bool

	DeclaredReadOnly    bool
	DeclaredIdempotent  bool
	DeclaredDestructive bool
}

// HintsFromMeta parses the canonical Meta keys the kernel itself reads
// ("readOnlyHint", "idempotentHint", "destructive") into a Hints. A key that is
// absent leaves the matching Declared* false; a key present with any value other
// than "true"/"false" is treated as declared-but-false (the kernel's metaTrue only
// honors the exact string "true").
func HintsFromMeta(meta map[string]string) Hints {
	var h Hints
	if v, ok := meta["readOnlyHint"]; ok {
		h.DeclaredReadOnly = true
		h.ReadOnly = v == "true"
	}
	if v, ok := meta["idempotentHint"]; ok {
		h.DeclaredIdempotent = true
		h.Idempotent = v == "true"
	}
	if v, ok := meta["destructive"]; ok {
		h.DeclaredDestructive = true
		h.Destructive = v == "true"
	}
	return h
}

// ToolFacts is the static description of one registered tool. Fields a given facts
// source cannot populate stay zero-valued, and the rules treat a zero-valued input
// as "no claim" rather than a false claim — so a partial view never invents a
// finding it has no evidence for.
type ToolFacts struct {
	Name string
	Kind Kind

	// Hints is the declared annotation triple (empty Declared* => no classifier).
	Hints Hints

	// HasPreflightSchema is true iff the pre-flight ladder enforces a schema for
	// this tool (rung-1). SchemaTypes maps each REQUIRED field to its declared
	// pre-flight type string ("string"/"number"/...), carried as a string so this
	// package need not import preflight; TL006 checks each against the supported set.
	HasPreflightSchema bool
	SchemaTypes        map[string]string

	// HasGrammar is true iff a grammar (alias/repair) is registered for this tool —
	// an alternative in-kernel arg contract that exempts the tool from the
	// "advertised-but-unenforced" finding (TL004).
	HasGrammar bool

	// Advertised is true iff the tool is shown to the model with a parameter schema;
	// AdvertisedRequired lists the required params that schema declares. These come
	// from the model-facing tool catalog, the half of the contract TL004 compares
	// against what the kernel actually enforces.
	Advertised         bool
	AdvertisedRequired []string

	// PolicyDenied is true iff the adjudicator policy provably refuses this tool by
	// name (its Deny map). A denied tool that is ALSO on the vDSO fast path is the
	// TL008 hazard: kernel.Submit consults the fast path BEFORE folding the policy,
	// so the Deny never fires. Set only by a collector that can see the policy (the
	// agent/gateway layer); empty under the registries-only kernel view.
	PolicyDenied bool
}
