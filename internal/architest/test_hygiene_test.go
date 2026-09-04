package architest

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Issue #11307: tests must not bind hardcoded static localhost ports (e.g. :8080, :50051, :9090)
// which cause port collision flaky failures on concurrent runs, and must not use arbitrary
// brittle time.Sleep synchronization in test suites where channels, sync.WaitGroup, or bounded
// polling loops should be used instead.

type testHygieneViolation struct {
	File    string
	Line    int
	Kind    string // "hardcoded-port" or "brittle-sleep"
	Message string
}

func (v testHygieneViolation) String() string {
	return fmt.Sprintf("%s:%d [%s]: %s", v.File, v.Line, v.Kind, v.Message)
}

var hardcodedPortRe = regexp.MustCompile(`:(8080|50051|9090)\b`)

// scanTestHygiene inspects Go test AST for hardcoded port bindings and brittle synchronization sleeps.
func scanTestHygiene(path string, src []byte, checkSleeps bool) ([]testHygieneViolation, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var violations []testHygieneViolation

	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		// 1. Check for hardcoded port bindings in net.Listen, http.ListenAndServe, etc.
		if isBindingCall(call) {
			for _, arg := range call.Args {
				if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
					val := strings.Trim(lit.Value, "`\"")
					if hardcodedPortRe.MatchString(val) {
						pos := fset.Position(call.Pos())
						violations = append(violations, testHygieneViolation{
							File: path,
							Line: pos.Line,
							Kind: "hardcoded-port",
							Message: fmt.Sprintf("hardcoded port binding %q in %s; use dynamic port allocation (127.0.0.1:0)",
								val, callName(call)),
						})
					}
				}
			}
		}

		// 2. Check for time.Sleep calls if checkSleeps is requested for this package.
		if checkSleeps && isTimeSleep(call) {
			pos := fset.Position(call.Pos())
			violations = append(violations, testHygieneViolation{
				File:    path,
				Line:    pos.Line,
				Kind:    "brittle-sleep",
				Message: "brittle time.Sleep synchronization in test; use channels, sync.WaitGroup, or bounded retry/polling loops",
			})
		}

		return true
	})

	return violations, nil
}

func isBindingCall(call *ast.CallExpr) bool {
	name := callName(call)
	switch name {
	case "net.Listen", "net.ListenTCP", "http.ListenAndServe", "http.ListenAndServeTLS", "tls.Listen":
		return true
	default:
		return false
	}
}

func isTimeSleep(call *ast.CallExpr) bool {
	return callName(call) == "time.Sleep"
}

func callName(call *ast.CallExpr) string {
	switch fn := call.Fun.(type) {
	case *ast.SelectorExpr:
		if ident, ok := fn.X.(*ast.Ident); ok {
			return ident.Name + "." + fn.Sel.Name
		}
	case *ast.Ident:
		return fn.Name
	}
	return ""
}

// TestNoHardcodedPortBindingsInTests verifies test files across the repo do not bind hardcoded ports.
func TestNoHardcodedPortBindingsInTests(t *testing.T) {
	root := repoRoot(t)
	var violations []testHygieneViolation

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			name := info.Name()
			if name == ".git" || name == "vendor" || name == "_scratch" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(info.Name(), "_test.go") {
			return nil
		}

		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		v, err := scanTestHygiene(path, src, false)
		if err != nil {
			return nil // skip unparseable test fixtures
		}
		violations = append(violations, v...)
		return nil
	})

	if err != nil {
		t.Fatalf("walk repo: %v", err)
	}

	for _, v := range violations {
		t.Errorf("violation: %s", v)
	}
}

// TestNoBrittleSleepsInAuditedPackages verifies test files in packages audited by #11307
// have zero arbitrary time.Sleep calls for synchronization.
func TestNoBrittleSleepsInAuditedPackages(t *testing.T) {
	root := repoRoot(t)
	auditedPackages := []string{
		"internal/childprocess",
		"internal/vdso",
		"internal/l3kv",
		"internal/servingsupervision",
		"internal/tb4bench",
		"internal/agentopt",
		"internal/metalgemm",
		"internal/power",
		"internal/gpulease",
		"internal/breathgate",
		"internal/cache",
		"internal/dataslot",
		"internal/gcpgpu",
		"internal/openviking",
		"internal/dropin",
		"internal/fleetbus",
		"internal/configguide",
	}

	var violations []testHygieneViolation

	for _, pkg := range auditedPackages {
		pkgDir := filepath.Join(root, filepath.FromSlash(pkg))
		if _, err := os.Stat(pkgDir); os.IsNotExist(err) {
			continue
		}

		err := filepath.Walk(pkgDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() || !strings.HasSuffix(info.Name(), "_test.go") {
				return nil
			}

			src, err := os.ReadFile(path)
			if err != nil {
				return err
			}

			v, err := scanTestHygiene(path, src, true)
			if err != nil {
				return err
			}
			violations = append(violations, v...)
			return nil
		})
		if err != nil {
			t.Fatalf("scan %s: %v", pkg, err)
		}
	}

	for _, v := range violations {
		t.Errorf("brittle synchronization sleep in audited test suite (#11307): %s", v)
	}
}

// TestCheckTestHygieneFlagsHardcodedPortBinding proves the scanner flags hardcoded port bindings.
func TestCheckTestHygieneFlagsHardcodedPortBinding(t *testing.T) {
	snippet := `package foo_test

import (
	"net"
	"testing"
)

func TestBadListen(t *testing.T) {
	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
}
`
	violations, err := scanTestHygiene("bad_test.go", []byte(snippet), false)
	if err != nil {
		t.Fatalf("scanTestHygiene: %v", err)
	}
	if len(violations) != 1 || violations[0].Kind != "hardcoded-port" {
		t.Fatalf("expected 1 hardcoded-port violation, got: %+v", violations)
	}
}

// TestCheckTestHygieneFlagsBrittleSleep proves the scanner flags arbitrary time.Sleep calls.
func TestCheckTestHygieneFlagsBrittleSleep(t *testing.T) {
	snippet := `package foo_test

import (
	"testing"
	"time"
)

func TestBrittle(t *testing.T) {
	go doBackgroundWork()
	time.Sleep(50 * time.Millisecond)
}
`
	violations, err := scanTestHygiene("bad_test.go", []byte(snippet), true)
	if err != nil {
		t.Fatalf("scanTestHygiene: %v", err)
	}
	if len(violations) != 1 || violations[0].Kind != "brittle-sleep" {
		t.Fatalf("expected 1 brittle-sleep violation, got: %+v", violations)
	}
}
