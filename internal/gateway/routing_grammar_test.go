package gateway

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// routing_grammar_test.go — the naming-grammar witness for the gateway/router
// taxonomy (docs/notes/GATEWAY-VS-ROUTER-2026-09-19.md, rules 1 and 3).
//
// The taxonomy fixes two nouns on two axes:
//
//	boundary (gateway): forwards or admits a LIVE request — name it gateway,
//	  proxy, front, boundary, or dispatch. NEVER router.
//	decision (router): pure data-in / decision-out, no I/O on the hot path —
//	  name it for the decision it makes (a policy), and do not put a bare
//	  `Router` type inside a package already named `gateway`.
//
// This test is the durable receipt that the violation cannot return. It parses
// the package's own declarations (not a grep over the tree) and asserts:
//
//  1. no FORWARDING BOUNDARY — a type implementing the agent.Planner dispatch
//     contract (Complete+Model) — is named `Router` or ends in `Router`
//     (rule 1: a type that forwards a live request is a boundary, never a
//     router); and
//  2. the two bare names this sweep retired (`Router`/`RouterConfig`/
//     `NewRouter` in the tier-selection decision, and `ReplicaRouter` on the
//     forwarding boundary) do not come back (rule 3).
//
// It also PINS the renamed survivors — `ReplicaDispatch` (the forwarding
// boundary) and `TierPolicy` (the decision) — so the invariant cannot be
// satisfied by deleting the types, and it fails loudly if it finds no
// forwarding boundary at all rather than passing vacuously.

// gatewayGrammarForbiddenName is the noun rule 1 forbids on a forwarding
// boundary inside `package gateway`: the exact re-derivation trap the taxonomy
// note exists to remove.
const gatewayGrammarForbiddenName = "Router"

// forwardingBoundaryMethods is the method set that makes a type a *forwarding*
// boundary rather than a pure decision: it dispatches a live turn downstream
// (agent.Planner.Complete) under an identity (agent.Planner.Model). A type
// carrying both is, by the taxonomy's own test, a boundary — so rule 1 governs
// its name. A pure placement/selection decision implements neither and is
// therefore out of this witness's scope.
var forwardingBoundaryMethods = []string{"Complete", "Model"}

// retiredGatewayNames are the bare names this sweep removed from the package
// API. Their return re-opens the taxonomy gap the sweep closed.
var retiredGatewayNames = []string{"Router", "RouterConfig", "NewRouter", "DefaultRouterConfig", "ReplicaRouter", "NewReplicaRouter"}

// TestGatewayGrammarNoForwardingBoundaryNamedRouter asserts the rule-1 and
// rule-3 invariants over package gateway's own declarations.
func TestGatewayGrammarNoForwardingBoundaryNamedRouter(t *testing.T) {
	dir := gatewayPackageDir(t)
	fset := token.NewFileSet()

	fileOf := map[string]string{} // type name -> declaring file
	methods := map[string]map[string]bool{}
	constructors := map[string]string{} // func name -> declaring file

	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool {
		// Production declarations only: a test helper named for a router is not
		// a forwarding boundary shipped in the package's own API.
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	for _, pkg := range pkgs {
		for fileName, file := range pkg.Files {
			base := filepath.Base(fileName)
			for _, decl := range file.Decls {
				switch d := decl.(type) {
				case *ast.GenDecl:
					if d.Tok != token.TYPE {
						continue
					}
					for _, spec := range d.Specs {
						ts, ok := spec.(*ast.TypeSpec)
						if !ok {
							continue
						}
						fileOf[ts.Name.Name] = base
					}
				case *ast.FuncDecl:
					if d.Recv == nil {
						constructors[d.Name.Name] = base
						continue
					}
					for _, field := range d.Recv.List {
						for _, name := range receiverTypeNames(field.Type) {
							if methods[name] == nil {
								methods[name] = map[string]bool{}
							}
							methods[name][d.Name.Name] = true
						}
					}
				}
			}
		}
	}
	if len(fileOf) == 0 {
		t.Fatalf("parsed no type declarations in %s — the witness would pass vacuously", dir)
	}

	// (1) Rule 1: no forwarding boundary carries the decision noun.
	forwarding := 0
	for name := range fileOf {
		if !implementsForwardingBoundary(methods[name]) {
			continue
		}
		forwarding++
		if name == gatewayGrammarForbiddenName || strings.HasSuffix(name, gatewayGrammarForbiddenName) {
			t.Errorf("%s declares forwarding boundary %s: it implements the agent.Planner dispatch contract (Complete+Model) yet is named with the decision noun `%s`. Rule 1: a type that forwards a live request is a boundary — name it proxy/front/boundary/dispatch, never router",
				fileOf[name], name, gatewayGrammarForbiddenName)
		}
	}
	if forwarding == 0 {
		t.Fatalf("found no forwarding boundary in package gateway — the rule-1 half of this witness would pass vacuously; expected at least ReplicaDispatch")
	}

	// (2) Rule 3: the retired bare names do not come back.
	for _, retired := range retiredGatewayNames {
		if _, ok := fileOf[retired]; ok {
			t.Errorf("%s re-declares type %s, a name this sweep retired: a bare `Router`-family type must not live in `package gateway` (rule 3) and a forwarding boundary must not be named `ReplicaRouter` (rule 1)",
				fileOf[retired], retired)
		}
		if _, ok := constructors[retired]; ok {
			t.Errorf("%s re-declares constructor %s, a name this sweep retired", constructors[retired], retired)
		}
	}

	// (3) The renamed survivors must exist and keep their axis-specific shape.
	if _, ok := fileOf["ReplicaDispatch"]; !ok {
		t.Errorf("type ReplicaDispatch is missing: rule 1 names the replica forwarding boundary for its dispatch role, and deleting it would make this witness vacuous rather than green")
	}
	if !implementsForwardingBoundary(methods["ReplicaDispatch"]) {
		t.Errorf("ReplicaDispatch does not implement the agent.Planner dispatch contract (Complete+Model) — it is no longer the forwarding boundary the taxonomy names")
	}
	if _, ok := fileOf["TierPolicy"]; !ok {
		t.Errorf("type TierPolicy is missing: rule 3 names the pure tier-selection decision for the decision it makes")
	}
	if implementsForwardingBoundary(methods["TierPolicy"]) {
		t.Errorf("TierPolicy implements the agent.Planner dispatch contract (Complete+Model): a type that forwards live turns is a boundary under rule 1, not a decision")
	}
}

// implementsForwardingBoundary reports whether a method set carries every
// method that makes a type a live-turn forwarding boundary.
func implementsForwardingBoundary(methodSet map[string]bool) bool {
	for _, m := range forwardingBoundaryMethods {
		if !methodSet[m] {
			return false
		}
	}
	return true
}

// receiverTypeNames unwraps a method receiver expression to the bare type names
// it can refer to: `*T`, `T`, `T[P]`, and the qualified `pkg.T`.
func receiverTypeNames(expr ast.Expr) []string {
	switch e := expr.(type) {
	case *ast.StarExpr:
		return receiverTypeNames(e.X)
	case *ast.IndexExpr:
		return receiverTypeNames(e.X)
	case *ast.IndexListExpr:
		return receiverTypeNames(e.X)
	case *ast.Ident:
		return []string{e.Name}
	case *ast.SelectorExpr:
		return []string{e.Sel.Name}
	}
	return nil
}

// gatewayPackageDir resolves the directory holding this test's package, so the
// witness reads the package the test is compiled into regardless of the test
// runner's working directory.
func gatewayPackageDir(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	return wd
}
