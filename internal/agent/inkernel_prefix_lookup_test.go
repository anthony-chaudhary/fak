package agent

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

var prefixLookupWitnessLookupFailure = errors.New("lookup failure")

type prefixLookupWitnessFakeSnapshot struct{ panicOnClose bool }

func (s *prefixLookupWitnessFakeSnapshot) Close() {
	if s.panicOnClose {
		panic("cleanup panic")
	}
}

type prefixLookupWitnessOutcome struct {
	success    bool
	err        error
	panicValue any
}

func prefixLookupWitnessFakeLookup(panicOnClose bool) (*prefixLookupWitnessFakeSnapshot, []float32, int, int, int, error) {
	return &prefixLookupWitnessFakeSnapshot{panicOnClose}, nil, 7, 2, 3, prefixLookupWitnessLookupFailure
}
func prefixLookupWitnessOldInline(panicOnClose bool) (out prefixLookupWitnessOutcome) {
	var err error
	defer func() { out.panicValue = recover() }()
	defer func() { out.success = err == nil; out.err = err }()
	var cachedLogits []float32
	var m, sourceScope, tier, lookupSnapshotM int
	var lookupSnapshotConsulted bool
	var matchedSnapshot *prefixLookupWitnessFakeSnapshot
	lookup := func() (*prefixLookupWitnessFakeSnapshot, []float32, int, int, int, error) {
		return prefixLookupWitnessFakeLookup(panicOnClose)
	}
	var snap *prefixLookupWitnessFakeSnapshot
	snap, cachedLogits, m, sourceScope, tier, err = lookup()
	lookupSnapshotConsulted, lookupSnapshotM = true, m
	if err == nil {
		matchedSnapshot = snap
	} else if snap != nil {
		snap.Close()
	}
	_, _, _, _, _, _, _ = cachedLogits, m, sourceScope, tier, lookupSnapshotM, lookupSnapshotConsulted, matchedSnapshot
	return
}
func prefixLookupWitnessFixedAlias(panicOnClose bool) (out prefixLookupWitnessOutcome) {
	var err error
	defer func() { out.panicValue = recover() }()
	defer func() { out.success = err == nil; out.err = err }()
	func(callerErr *error) {
		var cachedLogits []float32
		var m, sourceScope, tier, lookupSnapshotM int
		var lookupSnapshotConsulted bool
		var matchedSnapshot *prefixLookupWitnessFakeSnapshot
		lookup := func() (*prefixLookupWitnessFakeSnapshot, []float32, int, int, int, error) {
			return prefixLookupWitnessFakeLookup(panicOnClose)
		}
		var snap *prefixLookupWitnessFakeSnapshot
		snap, cachedLogits, m, sourceScope, tier, *callerErr = lookup()
		lookupSnapshotConsulted, lookupSnapshotM = true, m
		if *callerErr == nil {
			matchedSnapshot = snap
		} else if snap != nil {
			snap.Close()
		}
		_, _, _, _, _, _, _ = cachedLogits, m, sourceScope, tier, lookupSnapshotM, lookupSnapshotConsulted, matchedSnapshot
	}(&err)
	return
}
func prefixLookupWitnessBrokenNamedReturn(panicOnClose bool) (out prefixLookupWitnessOutcome) {
	var err error
	defer func() { out.panicValue = recover() }()
	defer func() { out.success = err == nil; out.err = err }()
	err = func() (err error) {
		var cachedLogits []float32
		var m, sourceScope, tier, lookupSnapshotM int
		var lookupSnapshotConsulted bool
		var matchedSnapshot *prefixLookupWitnessFakeSnapshot
		lookup := func() (*prefixLookupWitnessFakeSnapshot, []float32, int, int, int, error) {
			return prefixLookupWitnessFakeLookup(panicOnClose)
		}
		var snap *prefixLookupWitnessFakeSnapshot
		snap, cachedLogits, m, sourceScope, tier, err = lookup()
		lookupSnapshotConsulted, lookupSnapshotM = true, m
		if err == nil {
			matchedSnapshot = snap
		} else if snap != nil {
			snap.Close()
		}
		_, _, _, _, _, _, _ = cachedLogits, m, sourceScope, tier, lookupSnapshotM, lookupSnapshotConsulted, matchedSnapshot
		return
	}()
	return
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefixLookupCleanupPanicTelemetry(t *testing.T) {
	for _, panicOnClose := range []bool{false, true} {
		old, fixed := prefixLookupWitnessOldInline(panicOnClose), prefixLookupWitnessFixedAlias(panicOnClose)
		if old.success || fixed.success || !errors.Is(old.err, prefixLookupWitnessLookupFailure) || !errors.Is(fixed.err, prefixLookupWitnessLookupFailure) || old.panicValue != fixed.panicValue {
			t.Fatalf("panic=%v old=%+v fixed=%+v", panicOnClose, old, fixed)
		}
	}
	// Negative control: the rejected extraction must be distinguished.
	broken := prefixLookupWitnessBrokenNamedReturn(true)
	if !broken.success || broken.err != nil || broken.panicValue != "cleanup panic" {
		t.Fatalf("negative control no longer demonstrates regression: %+v", broken)
	}
}

// This is a source-derived guard and a stdlib cleanup-panic model, not a native
// backend fixture. Keep the model tied to the production error-bearing branch:
// assigning an error only through helper results loses it if Close panics.
// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefixLookupErrorSlotSourceGuard(t *testing.T) {
	helper, err := os.ReadFile("inkernel_prefix_lookup.go")
	if err != nil {
		t.Fatal(err)
	}
	caller, err := os.ReadFile("inkernel_decode.go")
	if err != nil {
		t.Fatal(err)
	}
	f, err := parser.ParseFile(token.NewFileSet(), "inkernel_prefix_lookup.go", helper, 0)
	if err != nil {
		t.Fatal(err)
	}
	var fn *ast.FuncDecl
	for _, decl := range f.Decls {
		if d, ok := decl.(*ast.FuncDecl); ok && d.Name.Name == "lookupOwnedNativePrefix" {
			fn = d
		}
	}
	if fn == nil {
		t.Fatal("production lookup helper missing")
	}
	aliasParameter := false
	for _, field := range fn.Type.Params.List {
		for _, name := range field.Names {
			if name.Name == "callerErr" {
				ptr, ok := field.Type.(*ast.StarExpr)
				if ok {
					id, ok := ptr.X.(*ast.Ident)
					aliasParameter = ok && id.Name == "error"
				}
			}
		}
	}
	if !aliasParameter {
		t.Fatal("lookup must alias the caller's error slot")
	}
	ast.Inspect(fn, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == "err" {
			t.Error("helper-local err can hide an error from caller telemetry during panic")
		}
		return true
	})
	normalize := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	branch := `snap, cachedLogits, m, sourceScope, tier, *callerErr = p.scopedTree.LookupSnapshotTieredContext(ctx, owner, ids)
 lookupSnapshotConsulted, lookupSnapshotM = true, m
 if *callerErr == nil {
 matchedSnapshot = snap
 } else if snap != nil {
 snap.Close()
 }`
	if !strings.Contains(normalize(string(helper)), normalize(branch)) {
		t.Fatal("production scoped cleanup branch changed; re-audit panic telemetry and update witness")
	}
	cf, err := parser.ParseFile(token.NewFileSet(), "inkernel_decode.go", caller, 0)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	ast.Inspect(cf, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "lookupOwnedNativePrefix" {
			return true
		}
		calls++
		if len(call.Args) != 4 {
			t.Error("unexpected lookup signature")
			return true
		}
		addr, ok := call.Args[3].(*ast.UnaryExpr)
		if !ok || addr.Op != token.AND {
			t.Error("caller must pass its error slot by address")
			return true
		}
		id, ok := addr.X.(*ast.Ident)
		if !ok || id.Name != "err" {
			t.Error("caller must pass its terminal telemetry error slot")
		}
		return true
	})
	if calls != 1 {
		t.Fatalf("lookup call count = %d, want 1", calls)
	}
}
