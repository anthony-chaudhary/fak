package hooks

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestTierDeclared_LiveTreeClean: once the four leaves (dispatchorder/dojo/looprecover/
// nightrun) are declared, the gate must find ZERO undeclared internal packages on the
// real tracked tree — the same end-to-end witness internal/architest's
// TestEveryPackageDeclaresTier provides, one boundary earlier.
func TestTierDeclared_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateTierDeclaredTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	// Match authoritative hygiene agreement (HygieneGates TIER_DECLARED PushScoped: true):
	// findings outside the active push delta are demoted to advisory so peer WIP does not wedge.
	findings = ScopeTierDeclaredFindings(findings, []string{"internal/hooks/"}, true)
	var blocking []Finding
	for _, f := range findings {
		if !f.Advisory {
			blocking = append(blocking, f)
		}
	}
	if len(blocking) != 0 {
		t.Fatalf("undeclared internal leaf on the tracked tree: %+v", blocking)
	}
}

// TestTierDeclared_FiresOnUndeclaredLeaf: a synthetic tree with a tier table that does
// NOT list internal/synthundeclared must produce exactly one TIER_DECLARED finding;
// adding the row clears it.
func TestTierDeclared_FiresOnUndeclaredLeaf(t *testing.T) {
	tierBody := func(declareSynth bool) string {
		rows := `	"abi": 0,
	"hooks": 1,
`
		if declareSynth {
			rows += "\t\"synthundeclared\": 1,\n"
		}
		return "package architest\n\nvar tier = map[string]int{\n" + rows + "}\n"
	}

	build := func(declareSynth bool) *TrackedTree {
		root := t.TempDir()
		write := func(rel, body string) {
			p := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write(tierTableFile, tierBody(declareSynth))
		write("internal/abi/x.go", "package abi\n")
		write("internal/hooks/x.go", "package hooks\n")
		write("internal/synthundeclared/x.go", "package synthundeclared\n")
		// an _test.go-only dir must NOT count as a leaf needing a tier
		write("internal/onlytests/x_test.go", "package onlytests\n")
		return &TrackedTree{
			Root: root,
			Paths: []string{
				tierTableFile,
				"internal/abi/x.go",
				"internal/hooks/x.go",
				"internal/synthundeclared/x.go",
				"internal/onlytests/x_test.go",
			},
			fileCache: map[string]fileEntry{},
		}
	}

	// Undeclared -> exactly one finding, naming the leaf.
	findings, err := gateTierDeclaredTree(build(false))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 TIER_DECLARED finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != "TIER_DECLARED" || findings[0].File != "internal/synthundeclared/" {
		t.Fatalf("finding wrong: %+v", findings[0])
	}

	// Declared -> clean.
	findings, err = gateTierDeclaredTree(build(true))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("declared leaf should be clean, got %+v", findings)
	}
}

// TestTierDeclared_FiresOnStaleTierRow covers the other #1145 recurrence: a package
// was removed, but its tier row stayed behind. That must fail at the same hygiene gate
// as a missing row, without waiting for someone to run internal/architest by hand.
func TestTierDeclared_FiresOnStaleTierRow(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(tierTableFile, `package architest

var tier = map[string]int{
	"abi": 0,
	"hooks": 1,
	"stalegone": 1,
}
`)
	write("internal/abi/x.go", "package abi\n")
	write("internal/hooks/x.go", "package hooks\n")
	tree := &TrackedTree{
		Root: root,
		Paths: []string{
			tierTableFile,
			"internal/abi/x.go",
			"internal/hooks/x.go",
		},
		fileCache: map[string]fileEntry{},
	}

	findings, err := gateTierDeclaredTree(tree)
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 stale TIER_DECLARED finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != "TIER_DECLARED" || findings[0].File != "internal/stalegone/" {
		t.Fatalf("finding wrong: %+v", findings[0])
	}
}

// TestTierDeclared_FailsOpenOnUnreadableTable: with no tier table on the tree, the gate
// returns ErrCouldNotRun (fail open) rather than flagging every package.
func TestTierDeclared_FailsOpenOnUnreadableTable(t *testing.T) {
	tree := &TrackedTree{
		Root:      t.TempDir(),
		Paths:     []string{"internal/foo/x.go"},
		fileCache: map[string]fileEntry{},
	}
	if _, err := gateTierDeclaredTree(tree); err != ErrCouldNotRun {
		t.Fatalf("want ErrCouldNotRun on a missing tier table, got %v", err)
	}
}

func TestScopeTierDeclaredFindingsBlocksOnlyPushOwnedLeaf(t *testing.T) {
	in := []Finding{
		{Gate: "TIER_DECLARED", File: "internal/owned/", Detail: "owned missing"},
		{Gate: "TIER_DECLARED", File: "internal/peer/", Detail: "peer missing"},
		{Gate: "BROKEN_LINK", File: "internal/peer/doc.md", Detail: "unrelated gate"},
	}
	got := ScopeTierDeclaredFindings(in, []string{"internal/owned/new.go"}, true)
	if got[0].Advisory {
		t.Fatalf("push-owned leaf was demoted: %+v", got[0])
	}
	if !got[1].Advisory || !strings.Contains(got[1].Detail, "does not touch") {
		t.Fatalf("peer leaf not advisory: %+v", got[1])
	}
	if got[2].Advisory {
		t.Fatalf("unrelated gate was demoted: %+v", got[2])
	}
	fallback := ScopeTierDeclaredFindings(in[:1], nil, false)
	if fallback[0].Advisory {
		t.Fatalf("no-trunk fallback demoted finding: %+v", fallback[0])
	}
}

// TestLiveTreeGates_FailOnBlockingFindings is the structural policy test for issue #12378:
// any test named *LiveTreeClean* must enforce a clean gate by failing (e.g. t.Fatalf/t.Errorf)
// on blocking findings, and must not silently log. Tests that are intentionally advisory audits
// must be named *LiveTreeAudit* instead.
func TestLiveTreeGates_FailOnBlockingFindings(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi os.FileInfo) bool {
		return strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parser.ParseDir: %v", err)
	}

	cleanTestsFound := 0
	for _, pkg := range pkgs {
		for fileName, file := range pkg.Files {
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok {
					continue
				}
				if !strings.HasPrefix(fn.Name.Name, "Test") || !strings.Contains(fn.Name.Name, "LiveTreeClean") {
					continue
				}
				cleanTestsFound++

				hasFailureCall := false
				hasSilentLogOnlyOnFindings := false

				ast.Inspect(fn.Body, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					sel, ok := call.Fun.(*ast.SelectorExpr)
					if !ok {
						return true
					}
					switch sel.Sel.Name {
					case "Fatalf", "Fatal", "Errorf", "Error", "FailNow", "Fail":
						hasFailureCall = true
					}
					return true
				})

				ast.Inspect(fn.Body, func(n ast.Node) bool {
					ifStmt, ok := n.(*ast.IfStmt)
					if !ok {
						return true
					}
					condStr := fmt.Sprintf("%v", ifStmt.Cond)
					checksFindings := strings.Contains(condStr, "findings") || strings.Contains(condStr, "blocking")
					if !checksFindings {
						return true
					}
					bodyCallsLog := false
					bodyCallsFail := false
					ast.Inspect(ifStmt.Body, func(inner ast.Node) bool {
						c, ok := inner.(*ast.CallExpr)
						if !ok {
							return true
						}
						s, ok := c.Fun.(*ast.SelectorExpr)
						if !ok {
							return true
						}
						switch s.Sel.Name {
						case "Log", "Logf":
							bodyCallsLog = true
						case "Fatalf", "Fatal", "Errorf", "Error", "FailNow", "Fail":
							bodyCallsFail = true
						}
						return true
					})
					if bodyCallsLog && !bodyCallsFail {
						hasSilentLogOnlyOnFindings = true
					}
					return true
				})

				if !hasFailureCall {
					t.Errorf("%s in %s is named LiveTreeClean but contains no test failure assertion (t.Fatalf/t.Errorf) — rename to LiveTreeAudit if advisory",
						fn.Name.Name, filepath.Base(fileName))
				}
				if hasSilentLogOnlyOnFindings {
					t.Errorf("%s in %s checks findings but only logs (t.Logf) without failing — silent logging on nonempty findings is forbidden in clean gates",
						fn.Name.Name, filepath.Base(fileName))
				}
			}
		}
	}

	if cleanTestsFound < 2 {
		t.Fatalf("expected to scan at least 2 LiveTreeClean tests, found %d", cleanTestsFound)
	}
}
