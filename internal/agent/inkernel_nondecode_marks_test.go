package agent

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// nonDecodeDeviceCalls are the planner calls that run prefill, restore or
// admission device work. The kernel profile is process-wide, so each must run
// under a nativeperf non-decode mark or a concurrent decode window books it.
var nonDecodeDeviceCalls = map[string]bool{
	"Prefill": true, "Restore": true, "PrefixSnapshot": true,
	"prefillDivergentSuffix": true, "admitPrefixSnapshot": true, "admitPrefillCheckpoint": true,
}

type plannerFunc struct {
	calls  map[string]bool
	events []string
}

func calleeName(call *ast.CallExpr) string {
	switch fn := call.Fun.(type) {
	case *ast.SelectorExpr:
		return fn.Sel.Name
	case *ast.Ident:
		return fn.Name
	}
	return ""
}

func parsePlannerFuncs(t *testing.T) map[string]*plannerFunc {
	t.Helper()
	files, err := filepath.Glob("inkernel_*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	out := map[string]*plannerFunc{}
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			pf := &plannerFunc{calls: map[string]bool{}}
			// spanVars tracks the locals bound to nativeperf.StartDecodeSpan so
			// only the host decode span's Finish counts as the window-close
			// event; unrelated Finish methods (e.g. a verification round's)
			// must not break the mark-around-window contract.
			spanVars := map[string]bool{}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				if as, ok := n.(*ast.AssignStmt); ok && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
					if id, ok := as.Lhs[0].(*ast.Ident); ok {
						if rhs, ok := as.Rhs[0].(*ast.CallExpr); ok {
							if sel, ok := rhs.Fun.(*ast.SelectorExpr); ok {
								if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "nativeperf" && sel.Sel.Name == "StartDecodeSpan" {
									spanVars[id.Name] = true
								}
							}
						}
					}
				}
				if d, ok := n.(*ast.DeferStmt); ok {
					if sel, ok := d.Call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "End" {
						if id, ok := sel.X.(*ast.Ident); ok && id.Name == "nonDecode" {
							pf.events = append(pf.events, "DeferEnd")
							return false
						}
					}
				}
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				callee := calleeName(call)
				pf.calls[callee] = true
				switch callee {
				case "BeginNonDecodeWork", "StartDecodeSpan":
					pf.events = append(pf.events, callee)
				case "Finish":
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if id, ok := sel.X.(*ast.Ident); ok && spanVars[id.Name] {
							pf.events = append(pf.events, "Finish")
						}
					}
				case "End":
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if id, ok := sel.X.(*ast.Ident); ok && id.Name == "nonDecode" {
							pf.events = append(pf.events, "End")
						}
					}
				}
				return true
			})
			out[fd.Name.Name] = pf
		}
	}
	return out
}

// fak-test:runtime fast est=2s lane=default
func TestInKernelNonDecodeDeviceWorkIsMarked(t *testing.T) {
	funcs := parsePlannerFuncs(t)
	marked := func(name string) bool { return funcs[name] != nil && funcs[name].calls["BeginNonDecodeWork"] }
	for _, name := range []string{
		"generateReusedContextWithBias", "generateReusedSpeculative", "generateReusedMetalMTP",
		"WarmPrefix", "withSegmentBridge", "refLogitsExact",
	} {
		if !marked(name) {
			t.Errorf("%s runs non-decode device work without BeginNonDecodeWork", name)
		}
	}
	var covered func(name string, depth int) bool
	covered = func(name string, depth int) bool {
		if marked(name) {
			return true
		}
		if depth == 0 {
			return false
		}
		callers := 0
		for caller, pf := range funcs {
			if caller == name || !pf.calls[name] {
				continue
			}
			callers++
			if !covered(caller, depth-1) {
				return false
			}
		}
		return callers > 0
	}
	var unmarked []string
	devices := 0
	for name, pf := range funcs {
		for call := range nonDecodeDeviceCalls {
			if pf.calls[call] && call != name {
				devices++
				if !covered(name, 3) {
					unmarked = append(unmarked, name+"->"+call)
				}
			}
		}
	}
	sort.Strings(unmarked)
	if devices < 8 {
		t.Fatalf("found %d non-decode device call sites, want the planner's full set", devices)
	}
	if len(unmarked) > 0 {
		t.Errorf("non-decode device work reachable outside a mark: %v", unmarked)
	}
	spans := 0
	for name, pf := range funcs {
		if !pf.calls["StartDecodeSpan"] {
			continue
		}
		spans++
		got := strings.Join(pf.events, ",")
		if got != "BeginNonDecodeWork,DeferEnd,End,StartDecodeSpan,Finish,BeginNonDecodeWork" {
			t.Errorf("%s decode window events = %s, want the mark closed around exactly the window", name, got)
		}
	}
	if spans != 3 {
		t.Errorf("decode windows = %d, want 3", spans)
	}
}
