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

// interpreterEvalFloorSlice is the identifier name of the adjudicator's inline-eval write
// floor — the `<interpreter> <inline-program-flag>` prefix table (`python -c`, `node -e`, …)
// that commandWrites ranges over to treat an inline interpreter program as write-shaped.
// If decide.go renames the slice, update this constant (a visible, reviewable edit) — the
// gate witnesses the floor by the identifier the code actually reads.
const interpreterEvalFloorSlice = "interpreterEvalFlags"

// commandWritesFn is the predicate that decides whether a shell/exec command STRING is
// write-shaped. It is the one body that must consult BOTH the shell-verb floor
// (shellWriteVerbs, gated by TestShellSelfModifyGuardWiredInDecide via commandSelfModify)
// AND the interpreter inline-eval floor — both feed the same commandSelfModify guard.
const commandWritesFn = "commandWrites"

// bodyReferencesIdent reports whether the named top-level function (matched by name across
// every non-test file of the package at dir) references the identifier ident anywhere in its
// body. It is the identifier sibling of bodyCallsFunc: bodyCallsFunc finds a CALL of a
// callee, but a floor SLICE is ranged over (`for _, ev := range interpreterEvalFlags`), not
// called, so the witness must be an *ast.Ident use, not an *ast.CallExpr. Like bodyCallsFunc
// it reads SOURCE, not a built value: a runtime-built or reflected floor (which a self-editing
// loop could make say anything) cannot satisfy it — only a visible identifier reference does.
// Returns false if the function is absent (the caller turns that into a failure naming which
// function went missing).
func bodyReferencesIdent(t *testing.T, dir, fnName, ident string) bool {
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
					id, ok := n.(*ast.Ident)
					if ok && id.Name == ident {
						found = true
						return false
					}
					return true
				})
			}
		}
	}
	return found
}

// TestInlineEvalFloorWiredInCommandWrites is the cross-package half of the #172 Hole 1
// RESIDUAL self-modify floor — the inline-interpreter twin of TestShellSelfModifyGuardWiredInDecide.
//
// Hole 1's shell-verb floor (sed -i, > redirect, …) is gated two ways: the in-package behavioral
// twin TestSelfModifyGuardsShellWritePath, and the out-of-tree wiring gate
// TestShellSelfModifyGuardWiredInDecide. The RESIDUAL (commit 6f56ff4) closes the sibling laundering
// vector: a general-purpose interpreter invoked with an inline-program flag — `python3 -c
// "open('internal/abi/x.go','w').write(...)"`, `node -e "fs.writeFileSync('internal/adjudicator/decide.go', …)"`
// — opens and rewrites a guarded file from INSIDE the program string, carrying NO shellWriteVerbs
// leading token and NO `>` redirect, so without this floor it laundered a self-edit past every rung
// above. decide.go closes it by having commandWrites ALSO range over interpreterEvalFlags (the
// interpreter analogue of shellWriteVerbs) and report the command write-shaped, which routes it
// through the SAME commandSelfModify guard the shell floor uses.
//
// WHY A GATE (and why it BITES): nothing in the type system keeps that floor in commandWrites. A
// refactor — or a self-editing loop that does so deliberately — could delete the
// `for _, ev := range interpreterEvalFlags` loop (or the slice) from decide.go and the kernel would
// compile and pass every other architest test while the inline-eval write path silently regained the
// ability to launder a `python -c` self-edit into internal/abi/. The residual added an in-package
// behavioral floor (the python-c/node-e cases in TestCommandWrites), but that test lives INSIDE a
// witness tree (internal/adjudicator), so the same self-edit that drops the floor could drop the test
// that guards it — the exact circularity the witness-tree gates were written to break. This gate is the
// belt to that test's suspenders: it lives OUTSIDE adjudicator, re-derives the wiring from adjudicator's
// source, and fails if commandWrites stops consulting the inline-eval floor.
//
// It is NOT a duplicate of TestShellSelfModifyGuardWiredInDecide: that gate proves Adjudicate calls
// commandSelfModify (the guard is ON the path); THIS gate proves commandWrites — the write-shape
// predicate that guard consults — still contains the interpreter floor. Deleting commandSelfModify's
// CALL trips the shell gate; deleting the interpreterEvalFlags loop INSIDE commandWrites trips only
// this one. Both rungs are needed for the residual to actually bite a python/node self-edit.
//
// Seeded GREEN (2026-06-20): decide.go's commandWrites ranges over interpreterEvalFlags (decide.go:464).
// Deleting that loop or the slice — re-opening the Hole 1 residual — turns this gate RED from outside
// every guarded tree.
func TestInlineEvalFloorWiredInCommandWrites(t *testing.T) {
	internal := internalDir(t)
	adjDir := filepath.Join(internal, "adjudicator")

	if !bodyReferencesIdent(t, adjDir, commandWritesFn, interpreterEvalFloorSlice) {
		t.Errorf("adjudicator.%s does not reference %s — the #172 Hole 1 RESIDUAL inline-eval "+
			"self-modify floor is NOT consulted by the write-shape predicate. An interpreter inline "+
			"program (`python3 -c \"open('internal/abi/x.go','w').write(...)\"`, `node -e "+
			"\"fs.writeFileSync('internal/adjudicator/decide.go', …)\"`) writes a guarded file from "+
			"inside an opaque program string, carrying no shellWriteVerbs token and no '>' redirect, so "+
			"%s must ALSO range over %s and treat such a command as write-shaped — only then does it "+
			"route through commandSelfModify and deny the self-edit. Without that reference a "+
			"self-improving loop can launder a python/node self-edit past the floor. Re-add the loop in "+
			"decide.go (the in-package python-c/node-e cases in TestCommandWrites are its behavioral "+
			"twin), or, if %s was renamed, update interpreterEvalFloorSlice/commandWritesFn in this gate.",
			commandWritesFn, interpreterEvalFloorSlice, commandWritesFn, interpreterEvalFloorSlice,
			interpreterEvalFloorSlice)
	}
}
