package toolproc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// repeatreason_test.go — the witness for the one claim repeatreason.go makes that cannot
// defend itself: that reuseReasonCodes is TOTAL over ReuseReason. Totality is exactly the
// property that rots silently. A seventh ReuseReason added to repeatreuse.go compiles fine,
// passes every existing test, and simply returns ok=false forever — and a consumer that
// "MUST NOT fabricate" a verdict then has no verdict to cite for a decision the fold is
// still making. Nothing about that failure is visible at the call site.
//
// So the totality test does NOT restate the six tokens. Restating them is the same mistake
// the anchor-refusal vocabulary test was written to avoid: a mirror pinned against its own
// reflection agrees with itself no matter how far both have drifted. It DERIVES the
// vocabulary from the package's own source, so adding a reason without mapping it reds here
// on the next run rather than at some future call site.

// declaredReuseReasons parses this package's non-test sources and returns the string VALUE
// of every constant declared with type ReuseReason, keyed by its Go identifier.
//
// Deriving from source rather than from a hand-kept list is the whole point: the list is
// what would drift. Values, not identifiers, because reuseReasonCodes is keyed by a
// ReuseReason value.
//
// Worth being precise about what that does and does not catch, since the obvious guess is
// wrong: editing a constant's literal is NOT a drift, because reuseReasonCodes keys off the
// CONSTANT, so the map key moves with it (measured — a renamed literal leaves every test
// here green, correctly). What this catches is a reason declared with no row at all, and —
// in the reverse direction below — a map keyed by a restated string literal that has since
// gone stale. The second is only reachable if someone writes `"keyed_hit": ...` instead of
// `ReasonKeyedHit: ...`, which is exactly the shortcut worth failing on.
func declaredReuseReasons(t *testing.T) map[string]string {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, 0)
	if err != nil {
		t.Fatalf("parse package source: %v", err)
	}
	found := map[string]string{}
	for _, pkg := range pkgs {
		for name, f := range pkg.Files {
			if strings.HasSuffix(name, "_test.go") {
				continue
			}
			for _, d := range f.Decls {
				gd, ok := d.(*ast.GenDecl)
				if !ok || gd.Tok != token.CONST {
					continue
				}
				for _, s := range gd.Specs {
					vs, ok := s.(*ast.ValueSpec)
					if !ok {
						continue
					}
					id, ok := vs.Type.(*ast.Ident)
					if !ok || id.Name != "ReuseReason" {
						continue
					}
					for i, n := range vs.Names {
						if i >= len(vs.Values) {
							continue
						}
						lit, ok := vs.Values[i].(*ast.BasicLit)
						if !ok || lit.Kind != token.STRING {
							continue
						}
						v, err := strconv.Unquote(lit.Value)
						if err != nil {
							t.Fatalf("unquote %s = %s: %v", n.Name, lit.Value, err)
						}
						found[n.Name] = v
					}
				}
			}
		}
	}
	return found
}

// TestEveryDeclaredReuseReasonHasARegisteredCode is the totality assertion. It reds when a
// ReuseReason is added to repeatreuse.go without a row in reuseReasonCodes — the silent
// failure the doc comment promises cannot happen.
func TestEveryDeclaredReuseReasonHasARegisteredCode(t *testing.T) {
	declared := declaredReuseReasons(t)
	if len(declared) == 0 {
		t.Fatal("derived zero ReuseReason constants from source: the deriver is broken, " +
			"and a broken deriver would make this test pass vacuously forever")
	}
	for ident, val := range declared {
		if _, ok := ReuseReasonCode(ReuseReason(val)); !ok {
			t.Errorf("ReuseReason %s (%q) is declared but has no registered code: add it to "+
				"reuseReasonCodes and ReuseReasonPairs, or a consumer can never cite this verdict",
				ident, val)
		}
	}
	// The other direction: a mapped token that no longer exists in source is dead vocabulary,
	// and dead vocabulary is how a retired reason keeps a code alive that someone later reuses.
	values := map[string]bool{}
	for _, v := range declared {
		values[v] = true
	}
	for r := range reuseReasonCodes {
		if !values[string(r)] {
			t.Errorf("reuseReasonCodes maps %q, which is no longer a declared ReuseReason", r)
		}
	}
}

// TestReuseReasonPairsAgreesWithTheCodeMap pins the two exported views against each other.
// The file states reuseReasonCodes is "the single source of truth so ReuseReasonPairs and
// ReuseReasonCode cannot drift" — that is an assertion about two hand-maintained lists, so
// it needs a test rather than a comment.
func TestReuseReasonPairsAgreesWithTheCodeMap(t *testing.T) {
	pairs := ReuseReasonPairs()
	if len(pairs) != len(reuseReasonCodes) {
		t.Fatalf("ReuseReasonPairs has %d rows, reuseReasonCodes has %d: the two have drifted",
			len(pairs), len(reuseReasonCodes))
	}
	inPairs := map[abi.ReasonCode]string{}
	for _, p := range pairs {
		if p.Name == "" {
			t.Errorf("code %d registers an empty name: it would render as REASON_%d on the wire", p.Code, p.Code)
		}
		if _, dup := inPairs[p.Code]; dup {
			t.Errorf("code %d appears twice in ReuseReasonPairs", p.Code)
		}
		inPairs[p.Code] = p.Name
	}
	for r, c := range reuseReasonCodes {
		if _, ok := inPairs[c]; !ok {
			t.Errorf("ReuseReason %q maps to code %d, which ReuseReasonPairs never registers: "+
				"ReuseReasonCode would return a code that resolves to REASON_%d", r, c, c)
		}
	}
}

// TestReuseCodesAreDistinctAndOutOfTree pins the block allocation the file argues for: its
// own contiguous range, every code above ReasonCoreMax, and no collision with the
// process-table family ReasonPairs registers. A collision would not fail to compile — it
// would silently make one verdict render as the other's name, which is worse than a crash
// because the wire still looks well-formed.
func TestReuseCodesAreDistinctAndOutOfTree(t *testing.T) {
	seen := map[abi.ReasonCode]bool{}
	for _, p := range ReuseReasonPairs() {
		if p.Code <= abi.ReasonCoreMax {
			t.Errorf("%s = %d is at or below ReasonCoreMax (%d): out-of-tree codes must sit above it",
				p.Name, p.Code, abi.ReasonCoreMax)
		}
		if seen[p.Code] {
			t.Errorf("duplicate reuse code %d (%s)", p.Code, p.Name)
		}
		seen[p.Code] = true
	}
	for _, p := range ReasonPairs() {
		if seen[p.Code] {
			t.Errorf("reuse code %d collides with the process-table verdict %s: the two families "+
				"must not share a code", p.Code, p.Name)
		}
	}
	names := map[string]bool{}
	for _, p := range append(ReuseReasonPairs(), ReasonPairs()...) {
		if names[p.Name] {
			t.Errorf("verdict name %q is registered by two different codes", p.Name)
		}
		names[p.Name] = true
	}
}

// TestUnknownReuseReasonGetsNoCode pins the fail-closed half of ReuseReasonCode's contract.
// An empty Receipt.Reason is the realistic case — a zero-valued Receipt that never ran the
// fold — and returning a code for it would let a consumer cite a verdict for a decision that
// was never made.
func TestUnknownReuseReasonGetsNoCode(t *testing.T) {
	for _, r := range []ReuseReason{"", "keyed_miss", "KEYED_HIT", " keyed_hit", "reuse"} {
		if c, ok := ReuseReasonCode(r); ok {
			t.Errorf("ReuseReasonCode(%q) = (%d, true), want ok=false: a token outside the closed "+
				"set has no registered verdict and one must not be fabricated", r, c)
		}
	}
}

// TestReuseReasonNamesRoundTripOnceRegistered exercises the consumer's half of the bridge:
// after a consumer registers the pairs, abi.ReasonName resolves each code to its stable
// token rather than the REASON_<n> forward-compat fallback. That round-trip IS the reason
// this file exists, so it is asserted rather than assumed.
//
// This registers into the process-global abi registry, which has no unregister. That is safe
// here precisely because of TestReuseCodesAreDistinctAndOutOfTree: the block is this
// vocabulary's alone, so nothing else in the process can observe a different name for it.
func TestReuseReasonNamesRoundTripOnceRegistered(t *testing.T) {
	pairs := ReuseReasonPairs()
	for _, p := range pairs {
		if got := abi.ReasonName(p.Code); got != "REASON_"+strconv.Itoa(int(p.Code)) && got != p.Name {
			t.Fatalf("code %d already resolves to %q before registration — another vocabulary "+
				"has claimed this block", p.Code, got)
		}
	}
	for _, p := range pairs {
		abi.RegisterReason(p.Code, p.Name)
	}
	for _, p := range pairs {
		if got := abi.ReasonName(p.Code); got != p.Name {
			t.Errorf("abi.ReasonName(%d) = %q after registration, want %q", p.Code, got, p.Name)
		}
	}
	// And the mapping a live seam actually walks: receipt reason -> code -> stable name.
	for r := range reuseReasonCodes {
		c, ok := ReuseReasonCode(r)
		if !ok {
			t.Fatalf("ReuseReasonCode(%q) lost its mapping", r)
		}
		if n := abi.ReasonName(c); !strings.HasPrefix(n, "REUSE_") {
			t.Errorf("ReuseReason %q renders as %q, want a REUSE_-prefixed stable token", r, n)
		}
	}
}
