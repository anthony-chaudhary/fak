package main

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// TestCapabilitiesVerbIsDispatched is the reachability pin of #5558.
//
// runCapabilities compiled, was fully flag-parsed, and had a usage block plus
// three passing unit tests (capabilities_test.go) -- and its own
// guard_capabilities.go banner told the wrapped agent to run it -- yet
// answered `fak: unknown verb "capabilities"` for months, because the one
// thing it never had was a case in the cmd/fak/main.go dispatch switch. This
// is the same defect class #5546 closed for `fak idea-scout`
// (see TestIdeaScoutVerbIsDispatched in ideascout_test.go, the template this
// test follows).
//
// The existing capabilities_test.go cases cannot catch that class of bug:
// they call runCapabilities directly, so they stay green whether or not any
// user can reach it. This test asserts the two rungs a USER actually
// traverses instead:
//
//	rung 1  cmd/fak/main.go has a `case "capabilities":` whose body calls
//	        runCapabilities -- checked over the parsed AST, so a case that
//	        dispatched somewhere else (or an arm deleted by a future sweep)
//	        reds here, not in the field.
//	rung 2  devindex.DispatchVerbs -- the SAME parser the VERB_UNTIERED
//	        pre-push gate and the `fak help --all` / `fak dev` catalog read --
//	        sees the token, and it carries a tier. A verb the catalog cannot
//	        see is one no help surface lists.
//
// It scans the working tree (HEAD in CI) and asserts a property of
// correctly-wired code, not a snapshot, so it behaves identically in both.
func TestCapabilitiesVerbIsDispatched(t *testing.T) {
	const verb = "capabilities"
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Skipf("cmd/fak/main.go unreadable (%v); dispatch reachability is only checkable in-tree", err)
	}

	// rung 1: the case arm exists AND routes to runCapabilities.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "main.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse cmd/fak/main.go: %v", err)
	}
	var arm *ast.CaseClause
	ast.Inspect(file, func(n ast.Node) bool {
		cc, ok := n.(*ast.CaseClause)
		if !ok {
			return true
		}
		for _, e := range cc.List {
			lit, ok := e.(*ast.BasicLit)
			if ok && lit.Kind == token.STRING && strings.Trim(lit.Value, `"`) == verb {
				arm = cc
				return false
			}
		}
		return true
	})
	if arm == nil {
		t.Fatalf("cmd/fak/main.go has no `case %q:` -- `fak %s` answers \"unknown verb\" no matter what runCapabilities does. "+
			"Add the arm to the dispatch switch (see the `idea-scout` arm for the same fix, #5546).", verb, verb)
	}
	called := false
	ast.Inspect(arm, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == "runCapabilities" {
			called = true
		}
		return true
	})
	if !called {
		t.Errorf("the `case %q:` arm at %s does not call runCapabilities -- the verb dispatches somewhere else",
			verb, fset.Position(arm.Pos()))
	}

	// rung 2: the catalog/tier parser agrees, so help and the tier gate see it too.
	dispatched := false
	for _, tok := range devindex.DispatchVerbs(src) {
		if tok == verb {
			dispatched = true
			break
		}
	}
	if !dispatched {
		t.Errorf("devindex.DispatchVerbs does not see %q in cmd/fak/main.go -- `fak help --all`, `fak dev`, "+
			"and the VERB_UNTIERED gate all read that parser, so the verb would stay invisible to every help surface", verb)
	}
	if tier, ok := devindex.TierOf(verb); !ok {
		t.Errorf("%q has no tier in internal/devindex/tiers.go -- classify it in one tier block or the pre-push VERB_UNTIERED gate reds the tree", verb)
	} else if tier != devindex.TierDev {
		t.Errorf("%q is tiered %q; it is internal fleet tooling, not product surface (the frontdoor tier is ceiling-gated)", verb, tier)
	}
}

// TestCapabilitiesCarriesAHelpRow pins the other half of a reachable verb:
// `fak help capabilities` must answer. A dev-tier verb has no
// cmd/fak/help.go overviewGroups line by construction (the overview is
// frontdoor-ONLY, gated by TestOverviewIsExactlyFrontdoor), so its help row
// is the devindex catalog entry -- which is also what `fak help --all` and
// `fak dev` list. Without it, an agent that reads the guard-startup banner
// (guard_capabilities.go) or the field-borrow/study-repo skills and then
// reaches for help gets nothing.
func TestCapabilitiesCarriesAHelpRow(t *testing.T) {
	var out bytes.Buffer
	if !printVerbHelp(&out, "capabilities") {
		t.Fatal("`fak help capabilities` knows nothing about the verb -- add a verbManifest entry in internal/devindex/verbs.go")
	}
	got := out.String()
	// Dev-tier verbs are introduced by their canonical `fak dev <verb>` spelling.
	if !strings.Contains(got, "fak dev capabilities") {
		t.Errorf("help header does not name the canonical dev spelling:\n%s", got)
	}
}
