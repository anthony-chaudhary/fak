package toollint

// lint.go — the rule set and the fold. Each rule is a pure function of the facts
// that predicts ONE concrete kernel behavior; the Mechanism string on every finding
// names that behavior the way Verdict.By names the rung that decided — so a finding
// is auditable back to the exact code path it foresees. Rules that predict a
// fast-path decision borrow the kernel's own predicate (vdso.IsWriteShaped,
// vdso.ClassifyNamespace) so the lint can never disagree with the runtime.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// Severity orders how loudly a finding should be treated. SevError fails a
// `fak lint` run; SevWarn and SevInfo are reported but do not fail by default.
type Severity uint8

const (
	SevInfo Severity = iota
	SevWarn
	SevError
)

// String renders the severity as "error", "warn", or "info".
func (s Severity) String() string {
	switch s {
	case SevError:
		return "error"
	case SevWarn:
		return "warn"
	default:
		return "info"
	}
}

// Code is a stable lint identifier (TLNNN). Stable so a host can grep, suppress, or
// gate on a specific finding across releases.
type Code string

const (
	// DeadCacheHint: a write-shaped name carrying readOnly+idempotent. The vDSO's
	// destructive() override vetoes the fast path, so the cache hint is dead.
	DeadCacheHint Code = "TL001"
	// UnreachablePure: a tier-1 pure registration whose hints can never satisfy the
	// tier-1 gate, so the pure function is dead code.
	UnreachablePure Code = "TL002"
	// StaticWriteShape: a tier-3 static answer registered for a write-shaped name.
	// Tier-3 is served unconditionally, so the write would be silently swallowed.
	StaticWriteShape Code = "TL003"
	// AdvertisedUnenforced: a tool advertises required params to the model but the
	// kernel enforces no schema or grammar, so rung-1 cannot catch a malformed call.
	AdvertisedUnenforced Code = "TL004"
	// MultiNamespaceDegrade: a cacheable name that collides with >1 resource class,
	// so the finer eraser silently degrades it to a full flush.
	MultiNamespaceDegrade Code = "TL005"
	// SchemaTypeUnsupported: a pre-flight Required field whose declared type is
	// outside the supported subset, so typeOK silently treats it as TypeAny.
	SchemaTypeUnsupported Code = "TL006"
	// ContradictoryHints: a hint set that cannot all be true at once (e.g.
	// destructive AND idempotent), so the declaration is internally incoherent.
	ContradictoryHints Code = "TL007"
	// DenyFastPathBypass: a policy-DENIED tool also registered pure/static on the
	// vDSO. kernel.Submit consults the fast path BEFORE the adjudicator chain, so the
	// tool is served Allow and the provable Deny never fires.
	DenyFastPathBypass Code = "TL008"
)

// Finding is one lint result. Tool is the offending tool; Mechanism is the kernel
// behavior the finding predicts (the forensic link back to the code path).
type Finding struct {
	Code      Code
	Severity  Severity
	Tool      string
	Message   string
	Mechanism string
}

// Report is the fold of every rule over every tool, sorted deterministically.
type Report struct {
	Findings []Finding
}

// Errors / Warnings / Infos count by severity.
func (r Report) Errors() int   { return r.count(SevError) }
func (r Report) Warnings() int { return r.count(SevWarn) }
func (r Report) Infos() int    { return r.count(SevInfo) }

func (r Report) count(s Severity) int {
	n := 0
	for _, f := range r.Findings {
		if f.Severity == s {
			n++
		}
	}
	return n
}

// MaxSeverity is the highest severity present (SevInfo if the report is clean —
// callers should check len(Findings) to distinguish "clean" from "info only").
func (r Report) MaxSeverity() Severity {
	max := SevInfo
	for _, f := range r.Findings {
		if f.Severity > max {
			max = f.Severity
		}
	}
	return max
}

// Clean reports whether nothing was flagged at all.
func (r Report) Clean() bool { return len(r.Findings) == 0 }

// rule is one lint check over a single tool. A rule returns nil when it has nothing
// to say (the common case); the fold collects the rest.
type rule func(ToolFacts) *Finding

// rules is the registered rule set, run in Code order per tool.
var rules = []rule{
	ruleDeadCacheHint,
	ruleUnreachablePure,
	ruleStaticWriteShape,
	ruleAdvertisedUnenforced,
	ruleMultiNamespaceDegrade,
	ruleSchemaTypeUnsupported,
	ruleContradictoryHints,
	ruleDenyFastPathBypass,
}

// Lint folds every rule over every tool and returns a deterministically-ordered
// report. The order is (Tool, Code) so two runs over the same surface are
// byte-identical (a host can diff lint output across commits).
func Lint(facts []ToolFacts) Report {
	var out []Finding
	for _, f := range facts {
		for _, r := range rules {
			if fnd := r(f); fnd != nil {
				out = append(out, *fnd)
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Tool != out[j].Tool {
			return out[i].Tool < out[j].Tool
		}
		return out[i].Code < out[j].Code
	})
	return Report{Findings: out}
}

// claimsCacheable reports whether a tool's DECLARED hints claim it is fast-path
// eligible (readOnly AND idempotent, both declared true) WITHOUT declaring itself
// destructive. This is the annotation that says "serve me from the vDSO".
func claimsCacheable(h Hints) bool {
	return h.DeclaredReadOnly && h.ReadOnly &&
		h.DeclaredIdempotent && h.Idempotent &&
		!(h.DeclaredDestructive && h.Destructive)
}

// TL001 — a write-shaped name that nonetheless claims to be cacheable. The vDSO's
// destructive() re-check overrides the hint from the NAME alone, so tiers 1–2 will
// never serve this tool: the readOnly+idempotent annotation is dead weight that
// will mislead any human reading the definition. Borrow IsWriteShaped so the
// prediction is exactly the runtime veto.
func ruleDeadCacheHint(f ToolFacts) *Finding {
	if claimsCacheable(f.Hints) && vdso.IsWriteShaped(f.Name) {
		return &Finding{
			Code: DeadCacheHint, Severity: SevWarn, Tool: f.Name,
			Message:   "name is write-shaped but hints claim readOnly+idempotent; the cache hint is dead",
			Mechanism: "vdso.destructive() (unit 32) overrides the hint from the name; vDSO tiers 1-2 never serve this tool",
		}
	}
	return nil
}

// TL002 — a tier-1 pure registration that can never fire. Lookup gates tier-1 on
// readOnlyHint && idempotentHint && !destructive; if a pure tool's declared hints
// cannot satisfy that gate, the registered pure function is dead code (every call
// falls through to the engine). Only fires when hints were declared (a pure tool
// with no classifier view is not judged).
func ruleUnreachablePure(f ToolFacts) *Finding {
	if f.Kind != KindPure {
		return nil
	}
	// Name-only proof (unit 32): a write-shaped name makes vdso.destructive() true
	// unconditionally, so the tier-1 gate's !destructive term can NEVER hold whatever
	// the per-call hints are. This is provable from the registries-only view (no hint
	// classifier needed), exactly as TL003 fires for write-shaped static tools — so the
	// declared-hint guard below must not be allowed to suppress it.
	if vdso.IsWriteShaped(f.Name) {
		return &Finding{
			Code: UnreachablePure, Severity: SevWarn, Tool: f.Name,
			Message:   "registered as a tier-1 pure tool but its name is write-shaped; the tier-1 gate can never satisfy !destructive, so the pure function is dead",
			Mechanism: "vdso.destructive() returns true from the name (IsWriteShaped, unit 32); the vdso.Lookup tier-1 gate never holds for this tool",
		}
	}
	if !(f.Hints.DeclaredReadOnly || f.Hints.DeclaredIdempotent || f.Hints.DeclaredDestructive) {
		return nil // no classifier view of this tool's hints; cannot judge reachability
	}
	reachable := f.Hints.ReadOnly && f.Hints.Idempotent &&
		!(f.Hints.DeclaredDestructive && f.Hints.Destructive) &&
		!vdso.IsWriteShaped(f.Name)
	if !reachable {
		return &Finding{
			Code: UnreachablePure, Severity: SevWarn, Tool: f.Name,
			Message:   "registered as a tier-1 pure tool but its hints/name can never satisfy the tier-1 gate; the pure function is dead",
			Mechanism: "vdso.Lookup tier-1 gate (readOnlyHint && idempotentHint && !destructive) never holds for this tool",
		}
	}
	return nil
}

// TL003 — a canned static answer for a write-shaped tool. Lookup serves tier-3
// UNCONDITIONALLY: there is no readOnly or destructive gate on the static table, so
// a static answer registered for a "send"/"delete"/"book" tool would make the vDSO
// return a fixed blob and the write would NEVER execute. This is a soundness hazard,
// not a style nit — SevError.
func ruleStaticWriteShape(f ToolFacts) *Finding {
	if f.Kind == KindStatic && vdso.IsWriteShaped(f.Name) {
		return &Finding{
			Code: StaticWriteShape, Severity: SevError, Tool: f.Name,
			Message:   "a write-shaped tool has a canned static (tier-3) answer; the write would be silently swallowed",
			Mechanism: "vdso.Lookup serves tier-3 unconditionally (no destructive gate, unit 32); the engine call never runs",
		}
	}
	return nil
}

// TL004 — the model is shown a contract the kernel does not enforce. A tool that
// advertises required params to the planner but has neither a pre-flight schema nor
// a grammar means rung-1 cannot reject a malformed call: the model-facing schema and
// the kernel-enforced schema have drifted apart. Pure tools self-validate in their
// own function and are exempt; this is a coverage gap, so SevInfo.
func ruleAdvertisedUnenforced(f ToolFacts) *Finding {
	if f.Kind == KindPure {
		return nil
	}
	if f.Advertised && len(f.AdvertisedRequired) > 0 && !f.HasPreflightSchema && !f.HasGrammar {
		return &Finding{
			Code: AdvertisedUnenforced, Severity: SevInfo, Tool: f.Name,
			Message:   "advertises required params to the model but the kernel enforces no schema or grammar; rung-1 cannot catch a malformed call",
			Mechanism: "preflight.Adjudicate rung-1 (unit 48) defers when no schema is known for the tool; the model-facing contract is unenforced",
		}
	}
	return nil
}

// TL005 — a cacheable name the finer eraser CANNOT tag precisely. A name matching
// MORE THAN ONE resource class makes ClassifyNamespace degrade to the root. The
// effect is CONDITIONAL on the configured granularity: under the default Global
// eraser every read already binds only the root, so this is latent; under a
// non-Global (namespace/resource) eraser the collision strands the tool on every
// write to either class — it can never benefit from the finer scoping. The cache
// stays sound either way. The linter has no view of the (runtime-mutable)
// granularity, so it flags the latent definition-time hazard. SevInfo.
func ruleMultiNamespaceDegrade(f ToolFacts) *Finding {
	if !claimsCacheable(f.Hints) {
		return nil
	}
	if _, multi := vdso.ClassifyNamespace(f.Name); multi {
		return &Finding{
			Code: MultiNamespaceDegrade, Severity: SevInfo, Tool: f.Name,
			Message:   "name matches more than one resource class; under a finer (namespace/resource) eraser it can never be scoped and degrades to a full flush",
			Mechanism: "vdso.ClassifyNamespace returns multi-class, so the read binds the root; under Global this matches the default flush, under a finer eraser every write to either class strands it",
		}
	}
	return nil
}

// supportedSchemaTypes mirrors the JSON-Schema scalar subset preflight.typeOK
// honors. Carried here as strings so this package need not import preflight; a test
// (lint_sync_test) pins this set to preflight's so the two cannot drift. The empty
// string is preflight.TypeAny (an intentional "any type").
var supportedSchemaTypes = map[string]bool{
	"string": true, "number": true, "boolean": true, "object": true, "array": true, "": true,
}

// TL006 — a pre-flight Required field whose declared type is not in the supported
// subset. typeOK falls through to `return true` for an unknown type, so the field is
// accepted without ever being type-checked: a typo'd type ("str", "int", "bool")
// silently disables validation for that field. SevWarn.
func ruleSchemaTypeUnsupported(f ToolFacts) *Finding {
	if !f.HasPreflightSchema {
		return nil
	}
	// Deterministic order over the field map so the message is stable.
	fields := make([]string, 0, len(f.SchemaTypes))
	for k := range f.SchemaTypes {
		fields = append(fields, k)
	}
	sort.Strings(fields)
	for _, field := range fields {
		ty := f.SchemaTypes[field]
		if !supportedSchemaTypes[ty] {
			return &Finding{
				Code: SchemaTypeUnsupported, Severity: SevWarn, Tool: f.Name,
				Message:   "pre-flight schema field " + field + " declares unsupported type " + quote(ty) + "; it will never be type-checked",
				Mechanism: "preflight.typeOK (unit 48) returns true for an unknown type (treated as TypeAny); rung-1 validation for this field is a no-op",
			}
		}
	}
	return nil
}

// TL007 — an internally incoherent hint set. The kernel fail-closes (destructive
// wins), but a tool that declares BOTH destructive=true AND idempotent=true (or
// readOnly=true AND destructive=true) is making contradictory claims that will
// mislead any reader and any downstream tool that trusts one half. SevInfo.
func ruleContradictoryHints(f ToolFacts) *Finding {
	h := f.Hints
	destructive := h.DeclaredDestructive && h.Destructive
	if !destructive {
		return nil
	}
	if (h.DeclaredReadOnly && h.ReadOnly) || (h.DeclaredIdempotent && h.Idempotent) {
		return &Finding{
			Code: ContradictoryHints, Severity: SevInfo, Tool: f.Name,
			Message:   "hints declare destructive together with readOnly/idempotent; the claims contradict",
			Mechanism: "vdso.destructive() (unit 32) honors destructive (fail-closed), but the readOnly/idempotent claim is incoherent and misleading",
		}
	}
	return nil
}

// TL008 — a tool the operator's policy provably refuses (its Deny map) that is ALSO
// on the vDSO fast path. kernel.Submit consults vdso.Lookup BEFORE folding the
// adjudicator chain, so a tier-3 static answer (served unconditionally) or a
// tier-1-reachable pure result is returned Allow{By:"vdso"} and the policy Deny
// NEVER fires — a fast-path serve bypassing a provable refusal. Strictly
// higher-value than TL003: it catches a policy-denied name regardless of write-shape
// (a read-shaped denied name like "exfiltrate" is missed by TL003 but caught here).
// SevError — a soundness hole, not a style nit. Only a collector that can see the
// policy sets PolicyDenied, so this stays silent under the registries-only view.
func ruleDenyFastPathBypass(f ToolFacts) *Finding {
	if !f.PolicyDenied {
		return nil
	}
	if f.Kind == KindStatic {
		return &Finding{
			Code: DenyFastPathBypass, Severity: SevError, Tool: f.Name,
			Message:   "a policy-DENIED tool has a tier-3 static answer; the vDSO serves it as Allow before the policy Deny is folded",
			Mechanism: "kernel.Submit consults vdso.Lookup before the adjudicator chain; tier-3 is served unconditionally, so the policy Deny never fires",
		}
	}
	if f.Kind == KindPure {
		reachable := f.Hints.ReadOnly && f.Hints.Idempotent &&
			!(f.Hints.DeclaredDestructive && f.Hints.Destructive) &&
			!vdso.IsWriteShaped(f.Name)
		if reachable {
			return &Finding{
				Code: DenyFastPathBypass, Severity: SevError, Tool: f.Name,
				Message:   "a policy-DENIED tool is a tier-1 pure tool reachable under the read-only+idempotent gate; the vDSO serves it as Allow before the policy Deny is folded",
				Mechanism: "kernel.Submit consults vdso.Lookup before the adjudicator chain; the tier-1 gate holds, so the policy Deny never fires",
			}
		}
	}
	return nil
}

func quote(s string) string { return "\"" + s + "\"" }
