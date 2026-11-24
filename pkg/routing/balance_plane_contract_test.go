package routing

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// The tests in this file witness docs/ROUTER-FAILOVER-AUTHORITY.md §2.1
// ("Balance is not failover, and produces no severity") on the fabric side
// (ticket LB-07). The gateway side lives in
// platform/gateway/balance_plane_contract_test.go. Behaviour already proven by
// balance_select_test.go (TestBalanceSelectIsProactiveNotReactive and the
// distribution proofs) is not re-tested here.

// balancePlaneFiles are the fabric's balance-plane sources.
var balancePlaneFiles = []string{"balance.go", "balance_select.go"}

// failurePlaneIdent matches an identifier that names the failover/severity
// plane. A balance-plane symbol that mentions one has started to consume it.
var failurePlaneIdent = regexp.MustCompile(`Failure|Action|Ejected|Health|Status|Classify|RetryBudget|AttemptBudget`)

func parseBalancePlaneFile(t *testing.T, fset *token.FileSet, name string) *ast.File {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return f
}

func identsIn(n ast.Node) []string {
	var out []string
	if n == nil || reflect.ValueOf(n).IsNil() {
		return out
	}
	ast.Inspect(n, func(x ast.Node) bool {
		if id, ok := x.(*ast.Ident); ok {
			out = append(out, id.Name)
		}
		return true
	})
	return out
}

// TestFabricBalanceTypesCarryNoFailureOrHealthDependency witnesses §2.1:
// "`BalancePolicy.Select` takes a `BalanceSnapshot` (a *measurement*) and a key
// (*identity*). It is not a function of any `Failure`, any status code, or any
// `BackendHealth` ejection state."
//
// It reuses TestFabricIsPure's import-parsing pattern, narrowed to the balance
// plane: no balance file imports anything but the standard library (so neither
// platform/gateway nor platform/keyring), no balance file body names the
// in-package severity plane (Failure, Action, Classify, the budgets), and no
// exported balance symbol's signature or field mentions Failure, Action,
// Ejected, Health, or Status.
//
// fak-test:runtime fast
func TestFabricBalanceTypesCarryNoFailureOrHealthDependency(t *testing.T) {
	fset := token.NewFileSet()
	for _, name := range balancePlaneFiles {
		f := parseBalancePlaneFile(t, fset, name)

		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			first := strings.SplitN(path, "/", 2)[0]
			if strings.Contains(first, ".") {
				t.Errorf("%s imports %q; the balance plane may import only the standard library", name, path)
			}
		}

		for _, decl := range f.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				for _, id := range identsIn(d) {
					if failurePlaneIdent.MatchString(id) {
						t.Errorf("%s: func %s references %q, a failover/severity-plane name", name, d.Name.Name, id)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					var idents []string
					var label string
					exported := false
					switch s := spec.(type) {
					case *ast.TypeSpec:
						label, exported = s.Name.Name, s.Name.IsExported()
						idents = append(identsIn(s.Name), identsIn(s.Type)...)
					case *ast.ValueSpec:
						for _, n := range s.Names {
							label += n.Name + " "
							exported = exported || n.IsExported()
						}
						idents = append(idents, identsIn(s.Type)...)
						for _, n := range s.Names {
							idents = append(idents, n.Name)
						}
						for _, v := range s.Values {
							idents = append(idents, identsIn(v)...)
						}
					}
					if !exported {
						continue
					}
					for _, id := range idents {
						if failurePlaneIdent.MatchString(id) {
							t.Errorf("%s: exported %s mentions %q, a failover/severity-plane name", name, strings.TrimSpace(label), id)
						}
					}
				}
			}
		}
	}

	// The reflection half: Select's own signature is (BalanceGroup,
	// BalanceSnapshot, string) -> BalanceOrder, and no field reachable from
	// those types names the failover plane.
	sel, ok := reflect.TypeOf(BalancePolicy{}).MethodByName("Select")
	if !ok {
		t.Fatal("BalancePolicy has no Select method")
	}
	wantIn := []reflect.Type{reflect.TypeOf(BalancePolicy{}), reflect.TypeOf(BalanceGroup{}), reflect.TypeOf(BalanceSnapshot{}), reflect.TypeOf("")}
	if sel.Type.NumIn() != len(wantIn) || sel.Type.NumOut() != 1 || sel.Type.Out(0) != reflect.TypeOf(BalanceOrder{}) {
		t.Fatalf("Select signature = %v, want func(BalancePolicy, BalanceGroup, BalanceSnapshot, string) BalanceOrder", sel.Type)
	}
	for i, w := range wantIn {
		if sel.Type.In(i) != w {
			t.Fatalf("Select input %d = %v, want %v", i, sel.Type.In(i), w)
		}
	}
	seen := map[reflect.Type]bool{}
	var walk func(reflect.Type, string)
	walk = func(rt reflect.Type, path string) {
		for rt.Kind() == reflect.Pointer || rt.Kind() == reflect.Slice || rt.Kind() == reflect.Map {
			rt = rt.Elem()
		}
		if rt.Kind() != reflect.Struct || seen[rt] {
			return
		}
		seen[rt] = true
		if failurePlaneIdent.MatchString(rt.Name()) {
			t.Errorf("%s has type %s, a failover/severity-plane name", path, rt.Name())
		}
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			if failurePlaneIdent.MatchString(f.Name) || failurePlaneIdent.MatchString(f.Type.Name()) {
				t.Errorf("%s.%s (%s) names the failover/severity plane", path, f.Name, f.Type)
			}
			walk(f.Type, path+"."+f.Name)
		}
	}
	for _, rt := range []reflect.Type{reflect.TypeOf(BalanceGroup{}), reflect.TypeOf(BalanceSnapshot{}), reflect.TypeOf(BalanceOrder{})} {
		walk(rt, rt.Name())
	}
}

func cloneBalanceGroup(g BalanceGroup) BalanceGroup {
	out := g
	out.Members = append([]string(nil), g.Members...)
	if g.Weights != nil {
		out.Weights = make(map[string]int, len(g.Weights))
		for k, v := range g.Weights {
			out.Weights[k] = v
		}
	}
	return out
}

func cloneBalanceSnapshot(s BalanceSnapshot) BalanceSnapshot {
	out := s
	if s.Members != nil {
		out.Members = make(map[string]MemberScore, len(s.Members))
		for k, v := range s.Members {
			out.Members[k] = v
		}
	}
	return out
}

func contractStrategies() []BalanceStrategy {
	return []BalanceStrategy{
		BalanceStrategyPriority,
		BalanceStrategyWeightedHash,
		BalanceStrategyLeastLoad,
		BalanceStrategyLeastLatency,
		BalanceStrategyWeightedShare,
	}
}

// TestBalanceSelectIsIndependentOfAnyInputButSnapshotAndKey witnesses §2.1:
// "A balancer never produces a severity verdict, and never consumes one.
// `BalancePolicy.Select` takes a `BalanceSnapshot` (a *measurement*) and a key
// (*identity*)."
//
// Over the declared strategies and three snapshot shapes (fully measured,
// partially measured, empty), two Select calls on identical
// (group, snapshot, key) are reflect.DeepEqual, and no call mutates its
// BalanceGroup, BalanceSnapshot, or MemberScore arguments. With no hidden
// input there is nowhere for a failure or a health verdict to enter.
//
// fak-test:runtime fast
func TestBalanceSelectIsIndependentOfAnyInputButSnapshotAndKey(t *testing.T) {
	members := []string{"a", "b", "c", "d"}
	shapes := map[string]BalanceSnapshot{
		"measured": {Group: "g", Members: map[string]MemberScore{
			"a": {Load: 0.7, Latency: 0.2, Measured: true},
			"b": {Load: 0.1, Latency: 0.9, Measured: true},
			"c": {Load: 1.0, Latency: 0.5, Measured: true},
			"d": {Load: 0.3, Latency: 0.0, Measured: true},
		}},
		"partial": {Group: "g", Observed: true, Members: map[string]MemberScore{
			"a": {Load: 0.4, Latency: 0.4, Measured: true},
			"c": {Load: 7, Latency: -3, Measured: false},
		}},
		"empty": {Group: "g"},
	}
	shapeNames := make([]string, 0, len(shapes))
	for n := range shapes {
		shapeNames = append(shapeNames, n)
	}
	sort.Strings(shapeNames)

	var p BalancePolicy
	for _, strat := range contractStrategies() {
		for _, shape := range shapeNames {
			t.Run(string(strat)+"/"+shape, func(t *testing.T) {
				group := BalanceGroup{Name: "g", Members: append([]string(nil), members...), Strategy: strat, Weights: map[string]int{"a": 3, "b": 1}}
				snap := cloneBalanceSnapshot(shapes[shape])
				groupBefore, snapBefore := cloneBalanceGroup(group), cloneBalanceSnapshot(snap)
				for i := 0; i < 32; i++ {
					key := "req-" + itoa(i)
					first := p.Select(group, snap, key)
					second := p.Select(group, snap, key)
					if !reflect.DeepEqual(first, second) {
						t.Fatalf("key %s: Select is not a pure function of (group, snapshot, key):\n first=%+v\nsecond=%+v", key, first, second)
					}
					if !reflect.DeepEqual(group, groupBefore) {
						t.Fatalf("key %s: Select mutated its BalanceGroup: %+v, was %+v", key, group, groupBefore)
					}
					if !reflect.DeepEqual(snap, snapBefore) {
						t.Fatalf("key %s: Select mutated its BalanceSnapshot or a MemberScore: %+v, was %+v", key, snap, snapBefore)
					}
				}
			})
		}
	}
}

// TestBalanceOrderNeverShrinksTheDeclaredMembership witnesses §2.1:
// "Balance orders; failover substitutes. ... No balance change may add,
// remove, or re-derive a hop." and "A missing, stale, or unmeasured signal
// deprioritizes a member. It never removes one and never refuses service."
//
// For every strategy and every closed exclusion reason, BalanceOrder.Members
// is a permutation of group.Members: same length, same set, no duplicates, no
// additions. The test also proves every exclusion reason was reached, so the
// permutation claim is not vacuous for any of them.
//
// fak-test:runtime fast
func TestBalanceOrderNeverShrinksTheDeclaredMembership(t *testing.T) {
	members := []string{"a", "b", "c"}
	scenarios := []struct {
		name    string
		weights map[string]int
		snap    BalanceSnapshot
	}{
		{"saturated", nil, BalanceSnapshot{Group: "g", Members: map[string]MemberScore{
			"a": {Load: 1, Measured: true}, "b": {Load: 0.2, Measured: true}, "c": {Load: 0.5, Measured: true}}}},
		{"drained", map[string]int{"b": 0}, BalanceSnapshot{Group: "g", Members: map[string]MemberScore{
			"a": {Load: 0.1, Measured: true}, "b": {Load: 0, Measured: true}, "c": {Load: 0.1, Measured: true}}}},
		{"unmeasured", nil, BalanceSnapshot{Group: "g", Members: map[string]MemberScore{
			"a": {Load: 0.3, Measured: true}}}},
		{"no_snapshot", nil, BalanceSnapshot{Group: "g"}},
		{"observed_quiet", nil, BalanceSnapshot{Group: "g", Observed: true}},
		{"no_eligible_member", nil, BalanceSnapshot{Group: "g", Members: map[string]MemberScore{
			"a": {Load: 1, Measured: true}, "b": {Load: 1, Measured: true}, "c": {Load: 1, Measured: true}}}},
	}

	reached := map[BalanceExclusion]bool{}
	var p BalancePolicy
	for _, strat := range contractStrategies() {
		for _, sc := range scenarios {
			t.Run(string(strat)+"/"+sc.name, func(t *testing.T) {
				group := BalanceGroup{Name: "g", Members: append([]string(nil), members...), Strategy: strat, Weights: sc.weights}
				if err := group.Validate(); err != nil {
					t.Fatalf("scenario declares an invalid group: %v", err)
				}
				for i := 0; i < 64; i++ {
					o := p.Select(group, sc.snap, "k"+itoa(i))
					assertPermutation(t, group.Members, o.Members)
					if o.Plan.Reason != "" {
						reached[o.Plan.Reason] = true
					}
					for _, line := range o.Plan.Members {
						if line.Excluded != "" {
							reached[line.Excluded] = true
						}
					}
				}
			})
		}
	}
	for _, r := range BalanceExclusionReasons() {
		if !reached[BalanceExclusion(r)] {
			t.Errorf("exclusion reason %q was never reached; the permutation claim is untested for it", r)
		}
	}
}

func assertPermutation(t *testing.T, declared, got []string) {
	t.Helper()
	if len(got) != len(declared) {
		t.Fatalf("order %v has %d members, declared %v has %d", got, len(got), declared, len(declared))
	}
	want := map[string]bool{}
	for _, m := range declared {
		want[m] = true
	}
	seen := map[string]bool{}
	for _, m := range got {
		if !want[m] {
			t.Fatalf("order %v adds %q, which %v does not declare", got, m, declared)
		}
		if seen[m] {
			t.Fatalf("order %v repeats %q", got, m)
		}
		seen[m] = true
	}
}
