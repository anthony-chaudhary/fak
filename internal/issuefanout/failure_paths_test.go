package issuefanout

// failure_paths_test.go is the #2512 qa-failure-paths follow-on: every refusal
// or error Build can return is driven by exactly one table row asserting BOTH
// halves of the failure contract — the rejection is a deliberate *Refusal
// (never a bare error, never a partial plan) and its message names the
// recovery, not just the violation. TestEveryErrorReturnHasARecoveryTest pins
// the table exhaustive against the source: a new refusal site (or a bare
// errors.New/fmt.Errorf) cannot land in the package without a row here.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"strings"
	"testing"
)

// refusalContract has exactly one row per refusef call site in the package:
// the input that trips it, and the substrings proving the message states the
// violated requirement AND the recovery that clears it. Rows for refusals
// outside Build set drive instead of mutate.
var refusalContract = []struct {
	site   string
	mutate func(*Input)
	drive  func(t *testing.T) error
	want   []string
}{
	{
		site:   "title and leaf required",
		mutate: func(in *Input) { in.Title = " " },
		want:   []string{"title and leaf are required"},
	},
	{
		site:   "spine_ref required",
		mutate: func(in *Input) { in.SpineRef = "" },
		want:   []string{"spine_ref is required", "ship the minimal working spine first"},
	},
	{
		site:   "max below the floor",
		mutate: func(in *Input) { in.Max = MinFanout - 1 },
		want: []string{
			"below the fan-out floor",
			fmt.Sprintf("raise max to at least %d, or leave it 0 for the full taxonomy", MinFanout),
		},
	},
	{
		site:   "unknown area",
		mutate: func(in *Input) { in.Areas = []string{"bogus"} },
		want:   []string{`unknown area "bogus"`, "known: " + strings.Join(AreaNames(), ", ")},
	},
	{
		site:   "area filter below the floor",
		mutate: func(in *Input) { in.Areas = []string{"release"} },
		want:   []string{"below the fan-out floor", "widen the area filter (or drop it)"},
	},
	{
		site:   "candidate fails the issue contract",
		mutate: func(in *Input) { in.Leaf = "issue fanout" },
		want:   []string{"fails the issue contract", "fix the input field it names"},
	},
	{
		site: "paths without a Go package",
		drive: func(t *testing.T) error {
			in := spineInput()
			in.Paths = []string{"docs/integrations/openai-codex.md"}
			plan, err := Build(in)
			if !reflect.DeepEqual(plan, Plan{}) {
				t.Fatalf("a refused Build leaked a partial plan: %+v", plan)
			}
			return err
		},
		want: []string{"do not identify a Go package", "include one representative package path"},
	},
	{
		site: "paths with multiple Go packages",
		drive: func(t *testing.T) error {
			in := spineInput()
			in.Paths = []string{"cmd/fak/guard.go", "internal/issuefanout/issuefanout.go"}
			plan, err := Build(in)
			if !reflect.DeepEqual(plan, Plan{}) {
				t.Fatalf("a refused Build leaked a partial plan: %+v", plan)
			}
			return err
		},
		want: []string{"identify multiple Go packages", "pass one representative package"},
	},
	{
		site: "live filing without parent accounting",
		drive: func(t *testing.T) error {
			plan, err := Build(spineInput())
			if err != nil {
				t.Fatalf("Build(spineInput()): %v", err)
			}
			plan.Input.ParentIssue = 0
			plan.Input.ParentBaseline = 0
			_, err = FileLive(plan, nil, LiveOptions{Runner: func([]string) (string, string, bool) { return "", "", true }})
			return err
		},
		want: []string{"requires --parent-issue", "--parent-baseline-points"},
	}, {
		site: "live filing without a runner",
		drive: func(t *testing.T) error {
			plan, err := Build(spineInput())
			if err != nil {
				t.Fatalf("Build(spineInput()): %v", err)
			}
			res, err := FileLive(plan, nil, LiveOptions{})
			if !reflect.DeepEqual(res, LiveResult{}) {
				t.Fatalf("a refused FileLive leaked a partial result: %+v", res)
			}
			return err
		},
		want: []string{"needs a gh Runner", "set LiveOptions.Runner"},
	},
}

// TestBuildRefusalNamesRecovery drives each refusal site and asserts the full
// failure contract: a *Refusal outcome, no partial plan alongside it, and a
// message carrying both the violated requirement and the recovery.
func TestBuildRefusalNamesRecovery(t *testing.T) {
	for _, tc := range refusalContract {
		t.Run(tc.site, func(t *testing.T) {
			var err error
			if tc.drive != nil {
				err = tc.drive(t)
			} else {
				in := spineInput()
				tc.mutate(&in)
				var plan Plan
				plan, err = Build(in)
				if err != nil && !reflect.DeepEqual(plan, Plan{}) {
					t.Fatalf("a refused Build leaked a partial plan: %+v", plan)
				}
			}
			if err == nil {
				t.Fatalf("the %s input was accepted, want a refusal", tc.site)
			}
			if got := ClassifyOutcome(err); got != OutcomeRefused {
				t.Fatalf("refusal classified as %q, want %q (a contract rejection must be a *Refusal)", got, OutcomeRefused)
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("refusal does not name the recovery: got %q, want substring %q", err, want)
				}
			}
		})
	}
}

// TestEveryErrorReturnHasARecoveryTest pins refusalContract exhaustive against
// the source: the package constructs errors ONLY via refusef (so ClassifyOutcome
// buckets every rejection as a refusal), and every refusef call site has exactly
// one table row. A new error return cannot land without a test asserting its
// message names the recovery.
func TestEveryErrorReturnHasARecoveryTest(t *testing.T) {
	refusefs, bare := errorConstructorSites(t)
	if len(bare) != 0 {
		t.Fatalf("bare error constructor(s) in the package: %v — use refusef for contract refusals, or add a refusalContract row plus a classification test for a genuine error path", bare)
	}
	if refusefs != len(refusalContract) {
		t.Fatalf("%d refusef call site(s) in the package but %d refusalContract row(s): add one row per refusal asserting its message names the recovery", refusefs, len(refusalContract))
	}
}

// errorConstructorSites counts refusef call sites and bare error constructors
// (errors.New / fmt.Errorf) across the package's non-test files.
func errorConstructorSites(t *testing.T) (refusefs int, bare []string) {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	fset := token.NewFileSet()
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *ast.Ident:
				if fun.Name == "refusef" {
					refusefs++
				}
			case *ast.SelectorExpr:
				if pkg, ok := fun.X.(*ast.Ident); ok {
					if (pkg.Name == "errors" && fun.Sel.Name == "New") || (pkg.Name == "fmt" && fun.Sel.Name == "Errorf") {
						bare = append(bare, fmt.Sprintf("%s: %s.%s", fset.Position(call.Pos()), pkg.Name, fun.Sel.Name))
					}
				}
			}
			return true
		})
	}
	return refusefs, bare
}
