package architest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"
)

func bodyCallsFuncOrSelector(t *testing.T, dir, fnName, callee string) bool {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, dir,
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	found := false
	for _, p := range parsed {
		for _, f := range p.Files {
			for _, decl := range f.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Name.Name != fnName || fn.Body == nil {
					continue
				}
				ast.Inspect(fn.Body, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					switch target := call.Fun.(type) {
					case *ast.Ident:
						if target.Name == callee {
							found = true
							return false
						}
					case *ast.SelectorExpr:
						if target.Sel.Name == callee {
							found = true
							return false
						}
					}
					return true
				})
			}
		}
	}
	return found
}

// TestIslandsAudit is an alias for TestFeatureIslandsWiredAudit.
func TestIslandsAudit(t *testing.T) {
	TestFeatureIslandsWiredAudit(t)
}
// All 13 feature islands from September 1–4 must be actively wired into production
// runtime entry points and must never regress into unwired ghost implementations.
func TestFeatureIslandsWiredAudit(t *testing.T) {
	internal := internalDir(t)
	adjDir := filepath.Join(internal, "adjudicator")
	agentDir := filepath.Join(internal, "agent")
	policyDir := filepath.Join(internal, "policy")
	ctxmmuDir := filepath.Join(internal, "ctxmmu")
	gwDir := filepath.Join(internal, "gateway")

	checks := []struct {
		islandNum int
		name      string
		dir       string
		fnName    string
		callee    string
	}{
		{1, "adjudicator/lifecycle_fsm", adjDir, "AdjudicateWithFSM", "NewLifecycleFSM"},
		{2, "agent/goal_prefix_stabilizer", agentDir, "prepareUpstream", "StabilizePromptPrefix"},
		{3, "agent/goal_anchor", agentDir, "runArm", "NewGoalAnchor"},
		{4, "adjudicator/recovery_audit", adjDir, "New", "NewRecoveryAuditLedger"},
		{5, "adjudicator/heredoc_target", adjDir, "commandWriteTargetsWithSpecs", "ExtractHeredocTargets"},
		{6, "adjudicator/shell_edit_nudge", adjDir, "Adjudicate", "CheckShellEditNudge"},
		{7, "adjudicator/shell_ast_parser", adjDir, "commandWriteTargetsWithSpecs", "ParseBashWriteTargets"},
		{8, "policy/decouple_tiers", policyDir, "NewTieredEvaluator", "NewTieredEvaluator"},
		{9, "ctxmmu/compactor", ctxmmuDir, "Compactor", "NewCompactorWithMMU"},
		{10, "gateway/refusal_notes", gwDir, "CompactDenySummary", "CompressRefusalNote"},
		{11, "ctxmmu/positive_compactor", ctxmmuDir, "CompactPositive", "CompactPositiveState"},
		{12, "kv/kv.go+loopback.go", gwDir, "New", "DefaultStore"},
		{13, "gateway/negative_*", gwDir, "VerifyNominalHostCallbacks", "VerifyNominalHostCallbackRetention"},
	}

	for _, tc := range checks {
		t.Run(tc.name, func(t *testing.T) {
			if !bodyCallsFuncOrSelector(t, tc.dir, tc.fnName, tc.callee) {
				t.Errorf("Island #%d (%s): production function %s() does not call %s() — ghost implementation island detected!",
					tc.islandNum, tc.name, tc.fnName, tc.callee)
			}
		})
	}
}
