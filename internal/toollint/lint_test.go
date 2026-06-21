package toollint

import (
	"testing"
)

// findingsByCode indexes a report's findings by code for assertions.
func findingsByCode(r Report) map[Code]Finding {
	m := map[Code]Finding{}
	for _, f := range r.Findings {
		m[f.Code] = f
	}
	return m
}

// cacheableHints is the declared annotation that says "serve me from the vDSO".
func cacheableHints() Hints {
	return Hints{
		ReadOnly: true, Idempotent: true,
		DeclaredReadOnly: true, DeclaredIdempotent: true,
	}
}

// TL001 — a write-shaped name that still claims to be cacheable is a dead hint.
func TestTL001DeadCacheHint(t *testing.T) {
	// "send_message" is write-shaped ("send"); the cacheable hint can never serve.
	r := Lint([]ToolFacts{{Name: "send_message", Kind: KindEngine, Hints: cacheableHints()}})
	f, ok := findingsByCode(r)[DeadCacheHint]
	if !ok {
		t.Fatalf("TL001 expected for write-shaped cacheable tool; got %+v", r.Findings)
	}
	if f.Severity != SevWarn {
		t.Fatalf("TL001 severity: got %s want warn", f.Severity)
	}
	// A non-write-shaped cacheable read must NOT trip TL001.
	clean := Lint([]ToolFacts{{Name: "get_user_details", Kind: KindEngine, Hints: cacheableHints()}})
	if _, bad := findingsByCode(clean)[DeadCacheHint]; bad {
		t.Fatalf("TL001 false positive on read-shaped tool: %+v", clean.Findings)
	}
}

// TL002 — a pure registration whose hints can never satisfy the tier-1 gate is dead.
func TestTL002UnreachablePure(t *testing.T) {
	// Declared NOT idempotent => tier-1 gate (readOnly && idempotent) never holds.
	notIdem := Hints{ReadOnly: true, DeclaredReadOnly: true, DeclaredIdempotent: true}
	r := Lint([]ToolFacts{{Name: "calc", Kind: KindPure, Hints: notIdem}})
	if _, ok := findingsByCode(r)[UnreachablePure]; !ok {
		t.Fatalf("TL002 expected for unreachable pure tool; got %+v", r.Findings)
	}
	// A reachable pure tool (cacheable, non-write name) is clean.
	ok := Lint([]ToolFacts{{Name: "calculate", Kind: KindPure, Hints: cacheableHints()}})
	if _, bad := findingsByCode(ok)[UnreachablePure]; bad {
		t.Fatalf("TL002 false positive on reachable pure tool: %+v", ok.Findings)
	}
	// A pure tool with NO classifier view of its hints must not be judged — UNLESS its
	// name alone proves it unreachable (write-shaped => destructive => tier-1 can never
	// hold), which is provable from the registries-only view, like TL003.
	noHints := Lint([]ToolFacts{{Name: "calc", Kind: KindPure}})
	if _, bad := findingsByCode(noHints)[UnreachablePure]; bad {
		t.Fatalf("TL002 fired without any declared hints on a read-shaped name (cannot judge): %+v", noHints.Findings)
	}
	nameOnly := Lint([]ToolFacts{{Name: "run_report", Kind: KindPure}}) // "run" is write-shaped
	if _, ok := findingsByCode(nameOnly)[UnreachablePure]; !ok {
		t.Fatalf("TL002 must fire on a write-shaped pure tool from the name alone; got %+v", nameOnly.Findings)
	}
}

// TL008 — a policy-denied tool that is also on the vDSO fast path is served before
// the Deny folds.
func TestTL008DenyFastPathBypass(t *testing.T) {
	// Denied + static => the canned answer is served as Allow; the Deny never fires.
	stat := Lint([]ToolFacts{{Name: "exfiltrate", Kind: KindStatic, PolicyDenied: true}})
	f, ok := findingsByCode(stat)[DenyFastPathBypass]
	if !ok || f.Severity != SevError {
		t.Fatalf("TL008 expected (error) for denied static tool; got %+v", stat.Findings)
	}
	// Denied + reachable pure => same bypass.
	pure := Lint([]ToolFacts{{Name: "lookup_secret", Kind: KindPure, PolicyDenied: true, Hints: cacheableHints()}})
	if _, ok := findingsByCode(pure)[DenyFastPathBypass]; !ok {
		t.Fatalf("TL008 expected for denied reachable-pure tool; got %+v", pure.Findings)
	}
	// Denied but on NO fast path (KindEngine/Unknown) => no bypass (the Deny does fire).
	eng := Lint([]ToolFacts{{Name: "delete_account", Kind: KindEngine, PolicyDenied: true}})
	if _, bad := findingsByCode(eng)[DenyFastPathBypass]; bad {
		t.Fatalf("TL008 false positive on a denied tool that is NOT on the fast path: %+v", eng.Findings)
	}
	// A static tool that is NOT denied is not a bypass.
	allowed := Lint([]ToolFacts{{Name: "list_all_airports", Kind: KindStatic}})
	if _, bad := findingsByCode(allowed)[DenyFastPathBypass]; bad {
		t.Fatalf("TL008 false positive on a non-denied static tool: %+v", allowed.Findings)
	}
}

// TL003 — a canned static answer for a write-shaped tool is a soundness hazard.
func TestTL003StaticWriteShape(t *testing.T) {
	r := Lint([]ToolFacts{{Name: "delete_account", Kind: KindStatic}})
	f, ok := findingsByCode(r)[StaticWriteShape]
	if !ok {
		t.Fatalf("TL003 expected for write-shaped static tool; got %+v", r.Findings)
	}
	if f.Severity != SevError {
		t.Fatalf("TL003 severity: got %s want error", f.Severity)
	}
	// A read-shaped static tool (the seeded list_all_airports shape) is clean.
	clean := Lint([]ToolFacts{{Name: "list_all_airports", Kind: KindStatic}})
	if _, bad := findingsByCode(clean)[StaticWriteShape]; bad {
		t.Fatalf("TL003 false positive on read-shaped static tool: %+v", clean.Findings)
	}
}

// TL004 — an advertised contract the kernel does not enforce.
func TestTL004AdvertisedUnenforced(t *testing.T) {
	base := ToolFacts{Name: "search", Kind: KindEngine, Advertised: true, AdvertisedRequired: []string{"q"}}
	if _, ok := findingsByCode(Lint([]ToolFacts{base}))[AdvertisedUnenforced]; !ok {
		t.Fatalf("TL004 expected for advertised-unenforced tool")
	}
	// A pre-flight schema clears it.
	withSchema := base
	withSchema.HasPreflightSchema = true
	withSchema.SchemaTypes = map[string]string{"q": "string"}
	if _, bad := findingsByCode(Lint([]ToolFacts{withSchema}))[AdvertisedUnenforced]; bad {
		t.Fatalf("TL004 false positive when a pre-flight schema is present")
	}
	// A grammar clears it.
	withGrammar := base
	withGrammar.HasGrammar = true
	if _, bad := findingsByCode(Lint([]ToolFacts{withGrammar}))[AdvertisedUnenforced]; bad {
		t.Fatalf("TL004 false positive when a grammar is present")
	}
	// A pure tool self-validates and is exempt.
	pure := base
	pure.Kind = KindPure
	if _, bad := findingsByCode(Lint([]ToolFacts{pure}))[AdvertisedUnenforced]; bad {
		t.Fatalf("TL004 false positive on a pure tool")
	}
}

// TL005 — a cacheable name colliding with >1 resource class degrades to a flush.
func TestTL005MultiNamespaceDegrade(t *testing.T) {
	// "price_flight_in_currency" matches both "flight" (flights) and "currency" (fx).
	r := Lint([]ToolFacts{{Name: "price_flight_in_currency", Kind: KindEngine, Hints: cacheableHints()}})
	if _, ok := findingsByCode(r)[MultiNamespaceDegrade]; !ok {
		t.Fatalf("TL005 expected for multi-namespace cacheable tool; got %+v", r.Findings)
	}
	// A single-namespace cacheable read is clean.
	clean := Lint([]ToolFacts{{Name: "search_direct_flight", Kind: KindEngine, Hints: cacheableHints()}})
	if _, bad := findingsByCode(clean)[MultiNamespaceDegrade]; bad {
		t.Fatalf("TL005 false positive on single-namespace tool: %+v", clean.Findings)
	}
	// A multi-namespace tool that is NOT cacheable is not flagged (no precision to lose).
	noncache := Lint([]ToolFacts{{Name: "price_flight_in_currency", Kind: KindEngine}})
	if _, bad := findingsByCode(noncache)[MultiNamespaceDegrade]; bad {
		t.Fatalf("TL005 false positive on non-cacheable multi-namespace tool")
	}
}

// TL006 — a pre-flight required field with an unsupported type is never checked.
func TestTL006SchemaTypeUnsupported(t *testing.T) {
	bad := Lint([]ToolFacts{{Name: "t", Kind: KindEngine, HasPreflightSchema: true,
		SchemaTypes: map[string]string{"n": "int"}}}) // "int" is not in the supported subset
	f, ok := findingsByCode(bad)[SchemaTypeUnsupported]
	if !ok {
		t.Fatalf("TL006 expected for unsupported schema type; got %+v", bad.Findings)
	}
	if f.Severity != SevWarn {
		t.Fatalf("TL006 severity: got %s want warn", f.Severity)
	}
	// Supported types are clean.
	good := Lint([]ToolFacts{{Name: "t", Kind: KindEngine, HasPreflightSchema: true,
		SchemaTypes: map[string]string{"s": "string", "n": "number"}}})
	if _, b := findingsByCode(good)[SchemaTypeUnsupported]; b {
		t.Fatalf("TL006 false positive on supported types: %+v", good.Findings)
	}
}

// TL007 — destructive declared together with readOnly/idempotent is incoherent.
func TestTL007ContradictoryHints(t *testing.T) {
	h := Hints{Destructive: true, Idempotent: true, DeclaredDestructive: true, DeclaredIdempotent: true}
	if _, ok := findingsByCode(Lint([]ToolFacts{{Name: "t", Kind: KindEngine, Hints: h}}))[ContradictoryHints]; !ok {
		t.Fatalf("TL007 expected for destructive+idempotent hint set")
	}
	// Destructive alone is coherent.
	only := Hints{Destructive: true, DeclaredDestructive: true,
		DeclaredReadOnly: true, DeclaredIdempotent: true} // readOnly=false, idempotent=false
	if _, bad := findingsByCode(Lint([]ToolFacts{{Name: "t", Kind: KindEngine, Hints: only}}))[ContradictoryHints]; bad {
		t.Fatalf("TL007 false positive on a coherent destructive tool")
	}
}

// Report ordering is deterministic by (tool, code) so output diffs cleanly.
func TestLintDeterministicOrder(t *testing.T) {
	facts := []ToolFacts{
		{Name: "zzz_send", Kind: KindStatic},                                                     // TL003
		{Name: "aaa_get", Kind: KindEngine, Advertised: true, AdvertisedRequired: []string{"x"}}, // TL004
	}
	r1 := Lint(facts)
	r2 := Lint(facts)
	if len(r1.Findings) != len(r2.Findings) {
		t.Fatalf("nondeterministic finding count: %d vs %d", len(r1.Findings), len(r2.Findings))
	}
	for i := range r1.Findings {
		if r1.Findings[i] != r2.Findings[i] {
			t.Fatalf("nondeterministic order at %d: %+v vs %+v", i, r1.Findings[i], r2.Findings[i])
		}
	}
	// aaa_get sorts before zzz_send.
	if r1.Findings[0].Tool != "aaa_get" {
		t.Fatalf("order: first finding tool = %q, want aaa_get", r1.Findings[0].Tool)
	}
}

// Every finding carries a Mechanism (the forensic link to the predicted code path).
func TestEveryFindingHasMechanism(t *testing.T) {
	facts := []ToolFacts{
		{Name: "send_x", Kind: KindEngine, Hints: cacheableHints()},                                                    // TL001
		{Name: "calc", Kind: KindPure, Hints: Hints{ReadOnly: true, DeclaredReadOnly: true, DeclaredIdempotent: true}}, // TL002
		{Name: "delete_y", Kind: KindStatic},                                                                           // TL003
		{Name: "search", Kind: KindEngine, Advertised: true, AdvertisedRequired: []string{"q"}},                        // TL004
	}
	r := Lint(facts)
	if len(r.Findings) == 0 {
		t.Fatal("expected findings")
	}
	for _, f := range r.Findings {
		if f.Mechanism == "" {
			t.Fatalf("finding %s on %s has no Mechanism", f.Code, f.Tool)
		}
		if f.Message == "" {
			t.Fatalf("finding %s on %s has no Message", f.Code, f.Tool)
		}
	}
}
