package architest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// policy_convergence_test.go — enforce policy convergence and posture awareness on all
// registered security gates (#11384).
//
// Every security gate registered via abi.RegisterAdjudicator or abi.RegisterResultAdmitter
// must adhere to the centralized reference monitor contract:
// 1. Must not couple to mutable global singletons (adjudicator.Default, ifc.DefaultSinkGate, etc.)
// 2. Must not read raw environment variables (os.Getenv) on hot adjudication/admission paths
// 3. Must be posture-aware (respecting PolicyContext)
//
// A strictly bounded ratchet baseline (policyConvergenceBaseline) tracks existing legacy
// subsystems undergoing phased refactoring under #11357. Any newly added gate outside the baseline
// must strictly converge on the PolicyContext contract.

var policyConvergenceBaseline = map[string]string{
	"internal/engine":     "Issue #11357: needs PolicyContext migration for engine routing",
	"internal/ifc":        "Issue #11357: needs PostureDefaultOpen softening support",
	"internal/normgate":   "Issue #11357: couples to adjudicator.Default.SecretPolicy()",
	"internal/secretgate": "Issue #11357: couples to adjudicator.Default.SecretPolicy() in Admit",
	"internal/a2achan":    "Issue #11357: hardcoded fail-closed without PolicyContext",
	"internal/wirescreen": "Issue #11357: env-driven without PolicyContext",
}

var forbiddenGateSingletons = []string{
	"adjudicator.Default",
	"ifc.DefaultSinkGate",
	"ifc.DefaultStampGate",
	"normgate.Default",
}

type registeredGate struct {
	pkgPath  string
	file     string
	gateType string
	line     int
	isResult bool
}

func findRepoRootForConvergence(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repo root with dos.toml")
		}
		dir = parent
	}
}

func discoverRegisteredGates(t *testing.T, root string) []registeredGate {
	t.Helper()
	var gates []registeredGate
	internalDir := filepath.Join(root, "internal")

	fset := token.NewFileSet()
	err := filepath.Walk(internalDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}

		node, parseErr := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if parseErr != nil {
			return nil
		}

		// Check if file imports internal/abi
		hasABI := false
		for _, imp := range node.Imports {
			if strings.Contains(imp.Path.Value, "internal/abi") {
				hasABI = true
				break
			}
		}
		if !hasABI {
			return nil
		}

		// Full parse to inspect calls
		fullNode, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}

		relPath, _ := filepath.Rel(root, path)
		pkgDir := filepath.ToSlash(filepath.Dir(relPath))

		ast.Inspect(fullNode, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkgIdent, ok := sel.X.(*ast.Ident)
			if !ok || pkgIdent.Name != "abi" {
				return true
			}

			fnName := sel.Sel.Name
			if fnName == "RegisterAdjudicator" || fnName == "RegisterResultAdmitter" {
				gateType := "unknown"
				if len(call.Args) >= 2 {
					gateType = typesExprString(call.Args[1])
				}
				gates = append(gates, registeredGate{
					pkgPath:  pkgDir,
					file:     relPath,
					gateType: gateType,
					line:     fset.Position(call.Pos()).Line,
					isResult: fnName == "RegisterResultAdmitter",
				})
			}
			return true
		})

		return nil
	})

	if err != nil {
		t.Fatalf("discover gates: %v", err)
	}
	return gates
}

func typesExprString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		return typesExprString(e.X) + "." + e.Sel.Name
	case *ast.CompositeLit:
		return typesExprString(e.Type) + "{}"
	case *ast.CallExpr:
		return typesExprString(e.Fun) + "()"
	case *ast.UnaryExpr:
		return "&" + typesExprString(e.X)
	default:
		return "expr"
	}
}

func TestArchitestPolicyConvergence(t *testing.T) {
	root := findRepoRootForConvergence(t)
	gates := discoverRegisteredGates(t, root)

	if len(gates) == 0 {
		t.Fatal("no registered gates discovered via abi.RegisterAdjudicator / abi.RegisterResultAdmitter")
	}

	fset := token.NewFileSet()
	for _, gate := range gates {
		reason, inBaseline := policyConvergenceBaseline[gate.pkgPath]
		if inBaseline {
			t.Logf("[BASELINE RATCHET] %s (%s line %d) grandfathered: %s", gate.gateType, gate.file, gate.line, reason)
			continue
		}

		// For gates not in baseline, assert no direct coupling to forbidden singletons
		// and no raw os.Getenv calls inside decision methods.
		pkgDir := filepath.Join(root, filepath.FromSlash(gate.pkgPath))
		pkgs, err := parser.ParseDir(fset, pkgDir, func(fi os.FileInfo) bool {
			return !strings.HasSuffix(fi.Name(), "_test.go")
		}, 0)
		if err != nil {
			t.Fatalf("failed to parse package %s: %v", gate.pkgPath, err)
		}

		for _, pkg := range pkgs {
			for _, file := range pkg.Files {
				ast.Inspect(file, func(n ast.Node) bool {
					fn, ok := n.(*ast.FuncDecl)
					if !ok || fn.Body == nil {
						return true
					}
					name := fn.Name.Name
					if name == "Adjudicate" || name == "Admit" {
						ast.Inspect(fn.Body, func(inner ast.Node) bool {
							call, isCall := inner.(*ast.CallExpr)
							if isCall {
								if sel, isSel := call.Fun.(*ast.SelectorExpr); isSel {
									if id, isId := sel.X.(*ast.Ident); isId && id.Name == "os" && sel.Sel.Name == "Getenv" {
										t.Errorf("%s: gate method %s directly reads os.Getenv; must read from PolicyContext", fset.Position(call.Pos()), name)
									}
								}
							}
							sel, isSel := inner.(*ast.SelectorExpr)
							if isSel {
								exprStr := typesExprString(sel)
								for _, bad := range forbiddenGateSingletons {
									if exprStr == bad {
										t.Errorf("%s: gate method %s couples directly to mutable global %s", fset.Position(sel.Pos()), name, bad)
									}
								}
							}
							return true
						})
					}
					return true
				})
			}
		}
	}
}

func TestArchitestPolicyConvergenceRatchetIntegrity(t *testing.T) {
	root := findRepoRootForConvergence(t)
	// Assert all grandfathered baseline paths actually exist
	for pkgPath, rationale := range policyConvergenceBaseline {
		fullPath := filepath.Join(root, filepath.FromSlash(pkgPath))
		if _, err := os.Stat(fullPath); err != nil {
			t.Errorf("policyConvergenceBaseline entry %s does not exist on disk (rationale: %s); clean up grandfathered ratchet", pkgPath, rationale)
		}
	}
}
