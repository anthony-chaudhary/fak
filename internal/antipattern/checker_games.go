package antipattern

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// CheckerGames is solution/grader gaming: an artifact makes its own witness
// green instead of satisfying the behavior the witness is meant to measure.
const CheckerGames Class = "SOLUTION_GAMES_CHECKER"

var (
	artifactNameRE  = regexp.MustCompile(`(?i)(solution|submission|candidate|artifact)`)
	hardcodedPassRE = regexp.MustCompile(`(?i)\b(print|println|printf|echo)\s*\(?\s*["'](?:pass|passed|success|ok)["']`)
)

// detectCheckerGames scans source artifacts for two deliberately narrow,
// high-signal static signatures. It is not a proof of cheating; every finding
// names the exact review target.
func detectCheckerGames(root string) []Finding {
	var findings []Finding
	_ = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry == nil {
			return nil
		}
		if entry.IsDir() {
			if path != root && (entry.Name() == ".git" || entry.Name() == "vendor") {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		if artifactNameRE.MatchString(filepath.Base(path)) {
			findings = append(findings, hardcodedPassFindings(path, filepath.ToSlash(rel))...)
		}
		if strings.HasSuffix(path, "_test.go") {
			findings = append(findings, shortCircuitTestFindings(path, filepath.ToSlash(rel))...)
		}
		return nil
	})
	return findings
}

func hardcodedPassFindings(path, rel string) []Finding {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var findings []Finding
	s := bufio.NewScanner(f)
	for line := 1; s.Scan(); line++ {
		if hardcodedPassRE.MatchString(s.Text()) {
			findings = append(findings, Finding{Class: CheckerGames, Ref: rel + ":" + strconv.Itoa(line), Weight: 1, Detail: "artifact prints a hardcoded passing result"})
		}
	}
	return findings
}

func shortCircuitTestFindings(path, rel string) []Finding {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil
	}
	var findings []Finding
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil || !strings.HasPrefix(fn.Name.Name, "Test") {
			continue
		}
		assertAt := firstAssertionIndex(fn.Body.List)
		if assertAt < 1 {
			continue
		}
		for _, stmt := range fn.Body.List[:assertAt] {
			if isUnconditionalShortCircuit(stmt) {
				findings = append(findings, Finding{
					Class:  CheckerGames,
					Ref:    rel + ":" + strconv.Itoa(fset.Position(stmt.Pos()).Line),
					Weight: 1,
					Detail: "test short-circuits before its assertion",
				})
				break
			}
		}
	}
	return findings
}

func firstAssertionIndex(stmts []ast.Stmt) int {
	for i, stmt := range stmts {
		found := false
		ast.Inspect(stmt, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			switch sel.Sel.Name {
			case "Error", "Errorf", "Fatal", "Fatalf", "Fail", "FailNow", "Equal", "NoError", "True", "False":
				found = true
			}
			return !found
		})
		if found {
			return i
		}
	}
	return -1
}

func isUnconditionalShortCircuit(stmt ast.Stmt) bool {
	if _, ok := stmt.(*ast.ReturnStmt); ok {
		return true
	}
	expr, ok := stmt.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := expr.X.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return sel.Sel.Name == "Exit" || sel.Sel.Name == "Skip" || sel.Sel.Name == "SkipNow"
}
