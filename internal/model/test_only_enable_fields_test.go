package model

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// test_only_enable_fields_test.go — TICKET-07 of the hot-path copy sweep (Ref #13876).
//
// The second failure class the sweep found is presence-vs-invokability: a fast path is built and
// tested, but only `_test.go` code ever enables it, so production runs the slow path forever.
// TICKET-01 (the resident V4.1 indexer keys) was one instance. This gate makes the class
// mechanical: for every struct field that both (a) names a budget/enable knob and (b) is read in a
// guard in non-test code, a non-test writer must exist, or the field must carry an allowlist entry
// naming why it is exempt.
//
// Scope and honesty. The scan reads the package's own non-test files (the working spine is
// `go test ./internal/model/ -run TestOnlyEnable`); a field written only by another package's
// struct literal is invisible here and is therefore allowlisted with that reason rather than
// mistaken for test-only. Per the ticket's gold-plating boundary this proves no arbitrary
// reachability — only that a guard-read knob is not left with zero in-package production writer.
//
// Residual: the sibling dtype-ceiling gate over the Vulkan upload path lives behind the `vulkan`
// build tag and a live device; it is a named residual, not covered by the default build.

// enableBudgetField reports whether a field name names an enable/budget knob.
func enableBudgetField(name string) bool {
	return strings.Contains(name, "Budget") || strings.Contains(name, "Enable")
}

// collectGuardNames records every selector/identifier name appearing in a guard expression (an
// if/for header). These are the fields that gate a template or fast path open or closed.
func collectGuardNames(node ast.Node, into map[string]bool) {
	if node == nil {
		return
	}
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.SelectorExpr:
			into[x.Sel.Name] = true
		case *ast.Ident:
			into[x.Name] = true
		}
		return true
	})
}

// scanTestOnlyEnableFields parses every non-test .go file in dir and returns the sorted field names
// that name an enable/budget knob, are read in a guard, and have no non-test writer (assignment,
// increment, or keyed struct-literal field) in that directory.
func scanTestOnlyEnableFields(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	candidates := map[string]bool{}
	written := map[string]bool{}
	guardRead := map[string]bool{}
	fset := token.NewFileSet()
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, 0)
		if err != nil {
			return nil, err
		}
		ast.Inspect(file, func(n ast.Node) bool {
			ts, ok := n.(*ast.TypeSpec)
			if !ok {
				return true
			}
			st, ok := ts.Type.(*ast.StructType)
			if !ok || st.Fields == nil {
				return true
			}
			for _, fld := range st.Fields.List {
				for _, nm := range fld.Names {
					if enableBudgetField(nm.Name) {
						candidates[nm.Name] = true
					}
				}
			}
			return true
		})
		ast.Inspect(file, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.AssignStmt:
				for _, lhs := range x.Lhs {
					if sel, ok := lhs.(*ast.SelectorExpr); ok {
						written[sel.Sel.Name] = true
					}
				}
			case *ast.IncDecStmt:
				if sel, ok := x.X.(*ast.SelectorExpr); ok {
					written[sel.Sel.Name] = true
				}
			case *ast.KeyValueExpr:
				if id, ok := x.Key.(*ast.Ident); ok {
					written[id.Name] = true
				}
			}
			return true
		})
		ast.Inspect(file, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.IfStmt:
				collectGuardNames(x.Init, guardRead)
				collectGuardNames(x.Cond, guardRead)
			case *ast.ForStmt:
				collectGuardNames(x.Init, guardRead)
				collectGuardNames(x.Cond, guardRead)
				collectGuardNames(x.Post, guardRead)
			case *ast.RangeStmt:
				collectGuardNames(x.X, guardRead)
			}
			return true
		})
	}
	var flagged []string
	for name := range candidates {
		if guardRead[name] && !written[name] && enableBudgetField(name) {
			flagged = append(flagged, name)
		}
	}
	sort.Strings(flagged)
	return flagged, nil
}

// testOnlyEnableAllowlist exempts guard-read knobs whose production writer is legitimately not an
// in-package assignment. Every entry names why. An empty reason is a test failure.
var testOnlyEnableAllowlist = map[string]string{
	"EnableResidualHook": "public model.Config seam: production writer is the embedding caller, which materializes the field off-package; consumed by rawdecode/executor.go and the model residual seam",
	"HostCPUEnabled":     "caller-supplied MoEBandwidthProfile measurement field: the production writer is the profile probe outside this package's AST",
	"IndexerBudget":      "Qwen4ExpFlashNextConfig JSON field: production writer is internal/metalgemm (qwen4exp_plan.go), a cross-package struct literal invisible to this scan",
	// Documented residual of the sweep class this gate exists for. indexerKeyBudgetBytes was the
	// live instance until TICKET-01 landed its writer (v41IndexerKeyAdmissionBytes wiring,
	// 8e664ed4d); the entry is kept to name that history and to re-exempt it if the writer regresses.
	"indexerKeyBudgetBytes": "Ref #12640/TICKET-01: production writer added by 8e664ed4d; kept to document the swept class and re-exempt on regression",
}

// TestOnlyEnableFieldsHaveProductionWriters fails when a guard-read budget/enable field has no
// in-package production writer and is not allowlisted. A new test-only knob on a fast path turns it
// red; adding the allowlist entry with a reason is the deliberate escape.
// fak-test:runtime fast est=1s
func TestOnlyEnableFieldsHaveProductionWriters(t *testing.T) {
	flagged, err := scanTestOnlyEnableFields(".")
	if err != nil {
		t.Fatalf("scan failed: %v", err)
	}
	for _, name := range flagged {
		if reason, ok := testOnlyEnableAllowlist[name]; !ok || strings.TrimSpace(reason) == "" {
			t.Errorf("%s is read in a guard but has no non-test writer: production never enables it. "+
				"Either wire a production writer or add it to testOnlyEnableAllowlist with a reason naming the ticket", name)
		}
	}
}

// TestOnlyEnableDetectorFlagsSeededTestOnlyField proves the detector rejects the class it exists to
// catch: a guard-read knob written only by test code (no production writer) is reported, while a
// sibling knob with a production writer is not.
// fak-test:runtime fast est=1s
func TestOnlyEnableDetectorFlagsSeededTestOnlyField(t *testing.T) {
	dir := t.TempDir()
	src := "package seed\n\n" +
		"type seeded struct {\n" +
		"\tseededOnlyBudget int\n" +
		"\tseededProdBudget int\n" +
		"\tplainField       int\n" +
		"}\n\n" +
		"func readIt(x *seeded) {\n" +
		"\tif x.seededOnlyBudget > 1 {\n\t\treturn\n\t}\n" +
		"\tif x.seededProdBudget > 1 {\n\t\treturn\n\t}\n" +
		"}\n\n" +
		"func writeIt(x *seeded) { x.seededProdBudget = 3 }\n"
	if err := os.WriteFile(filepath.Join(dir, "seed.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	flagged, err := scanTestOnlyEnableFields(dir)
	if err != nil {
		t.Fatalf("scan failed: %v", err)
	}
	if len(flagged) != 1 || flagged[0] != "seededOnlyBudget" {
		t.Fatalf("seeded scan = %v, want [seededOnlyBudget]: a test-only knob must be flagged and a "+
			"production-written sibling must not", flagged)
	}
}
