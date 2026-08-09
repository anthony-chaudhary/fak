package main

// guard_parked_exit_witness_test.go — the durable half of #5862.
//
// #5862 measured 50 of 513 guarded sessions (9.7%) journaling ZERO rows: the
// file was created, the child booted, and the journal stayed zero-byte, so the
// session was evidence-free by construction and its cause unrecoverable.
//
// The terminal row that would have explained them already existed —
// appendGuardChildExitWitness -> journal.AppendChildExit writes a
// CHILD_EXIT/CHILD_CRASH row. It was simply SKIPPED on the goal-parked teardown
// branches, which is the shape most of those empties take (a provider 429 at
// turn 0 parks the goal with reason=LONG_RETRY_AFTER before the child ever
// proposes a tool call). The parked branches are the only teardowns in the
// supervision loop that reach an exit without a witness, so an empty journal was
// the guaranteed outcome of the very failure mode operators most needed to see.
//
// The asymmetry was invisible to review because the three parked branches sit in
// two different loops and only ONE of them was right:
//
//	guardChildRestart + parked   stop, witness, finish   correct
//	guardChildCompleted + parked banner, finish          no witness — and this is
//	                                                     the PRODUCTION path, since
//	                                                     dispatch always passes
//	                                                     --max-duration (1740s), and
//	                                                     maxDurationLimit > 0 routes
//	                                                     to the supervised loop
//	unsupervised loop + parked   banner, bare break      no witness AND no teardown
//	                                                     at all: no journal Close,
//	                                                     no refusal sidecar, no report
//
// A test that pinned today's three call sites by line would rot on the next edit
// and would not have caught the original divergence either, so this file
// DISCOVERS the parked branches by parsing the supervision source and asserts
// the invariant as a property over whatever it finds:
//
//	every `if rec, parked := guardGoalParked(); parked { … }` teardown must
//	call appendGuardChildExitWitness, must call finishGuardChildAndReport, and
//	must call them IN THAT ORDER.
//
// The ordering rung is load-bearing rather than stylistic:
// finishGuardChildAndReport ends by flushing and closing the audit journal
// (`auditJournal.Close()`), so a witness appended after it would be written to a
// closed journal and lost — the same zero-byte outcome by a different route.
//
// LOCAL-WORKTREE vs CI-HEAD SEAM: like guard_tempdir_convention_test.go, a
// source-scanning test reads the working tree locally and HEAD in CI, and
// cmd/fak is a shared tree. The assertion is a PROPERTY of correctly-written
// code rather than a snapshot, so a peer's new parked branch passes in both
// environments when written correctly and fails in both when it is not, and
// every failure names the file, the line, and the exact cure.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// guardParkedTeardownSource is the file that owns every goal-parked teardown.
// guardGoalParked is declared there and called nowhere else in the package, so
// scanning this one file is exhaustive over the population the property governs.
const guardParkedTeardownSource = "guard_child_supervision.go"

// guardParkedBranch is one discovered `if _, parked := guardGoalParked(); parked`
// teardown, with the positions of the two calls its exit must make.
type guardParkedBranch struct {
	line        int
	witnessLine int // 0 when appendGuardChildExitWitness is never called
	finishLine  int // 0 when finishGuardChildAndReport is never called
	witnessPos  token.Pos
	finishPos   token.Pos
}

func (b guardParkedBranch) where() string {
	return guardParkedTeardownSource + ":" + strconv.Itoa(b.line)
}

// guardScanParkedBranches parses the supervision source and returns every
// goal-parked branch it finds, recording which of the two teardown calls each
// body makes. It matches on the guardGoalParked() call in the if-statement's
// initializer rather than on a variable name, so renaming rec/parked does not
// silently drop a branch out of the scan.
func guardScanParkedBranches(t *testing.T, dir string) []guardParkedBranch {
	t.Helper()
	fset := token.NewFileSet()
	path := filepath.Join(dir, guardParkedTeardownSource)
	f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	var found []guardParkedBranch
	ast.Inspect(f, func(n ast.Node) bool {
		ifStmt, ok := n.(*ast.IfStmt)
		if !ok || ifStmt.Init == nil || !guardCallsIdent(ifStmt.Init, "guardGoalParked") {
			return true
		}
		branch := guardParkedBranch{line: fset.Position(ifStmt.Pos()).Line}
		ast.Inspect(ifStmt.Body, func(inner ast.Node) bool {
			call, ok := inner.(*ast.CallExpr)
			if !ok {
				return true
			}
			ident, ok := call.Fun.(*ast.Ident)
			if !ok {
				return true
			}
			switch ident.Name {
			case "appendGuardChildExitWitness":
				if branch.witnessLine == 0 {
					branch.witnessLine = fset.Position(call.Pos()).Line
					branch.witnessPos = call.Pos()
				}
			case "finishGuardChildAndReport":
				if branch.finishLine == 0 {
					branch.finishLine = fset.Position(call.Pos()).Line
					branch.finishPos = call.Pos()
				}
			}
			return true
		})
		found = append(found, branch)
		return true
	})
	return found
}

// guardCallsIdent reports whether node contains a call to the named package-level
// function.
func guardCallsIdent(node ast.Node, name string) bool {
	hit := false
	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == name {
			hit = true
			return false
		}
		return true
	})
	return hit
}

// TestGuardParkedChildTeardownWritesExitWitness is the discovering half: it walks
// every goal-parked teardown in the supervision loop as it exists on disk and
// holds each to the same three rungs. A new parked branch is covered the day it
// is written, with no list here to update.
func TestGuardParkedChildTeardownWritesExitWitness(t *testing.T) {
	branches := guardScanParkedBranches(t, ".")
	// A scan that finds nothing would pass vacuously, which is exactly how a
	// convention test rots into decoration. The supervision loop has carried a
	// parked branch in both its unsupervised and supervised halves since the park
	// was introduced; fewer than two means the scan, not the code, is broken.
	if len(branches) < 2 {
		t.Fatalf("scan of %s found %d goal-parked branches; expected at least 2 (one per supervision loop) — the scan is vacuous",
			guardParkedTeardownSource, len(branches))
	}
	for _, b := range branches {
		if b.witnessLine == 0 {
			t.Errorf("%s: a goal-parked teardown exits without appendGuardChildExitWitness, so this session's journal stays ZERO-ROW and its cause is unrecoverable (#5862).\n"+
				"\tcure: call appendGuardChildExitWitness(auditJournal, agentName, guardTraceID, nil, child.ProcessState, childStarted) before finishGuardChildAndReport, mirroring the guardChildRestart+parked branch.", b.where())
		}
		if b.finishLine == 0 {
			t.Errorf("%s: a goal-parked branch leaves the loop without finishGuardChildAndReport, so the gateway is never torn down, the journal is never flushed/closed, and the .refusals.json carry-forward sidecar is never written.\n"+
				"\tcure: end the branch with finishGuardChildAndReport(nil, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler) and return.", b.where())
		}
		if b.witnessLine != 0 && b.finishLine != 0 && b.witnessPos > b.finishPos {
			t.Errorf("%s: appendGuardChildExitWitness (line %d) runs AFTER finishGuardChildAndReport (line %d), which closes the audit journal — the witness row is written to a closed journal and lost.\n"+
				"\tcure: append the witness first.", b.where(), b.witnessLine, b.finishLine)
		}
	}
}

// TestGuardParkedChildWritesTerminalExitWitnessRow is the behavioral half: it
// drives appendGuardChildExitWitness with exactly the arguments a parked
// teardown passes — a nil runErr, because a parked child is stopped by the guard
// rather than crashed — and asserts the row that lands is the terminal row
// #5862's done-condition asks for, on a journal that would otherwise be empty.
func TestGuardParkedChildWritesTerminalExitWitnessRow(t *testing.T) {
	j := journal.OpenMemory()
	before := len(j.Recent(64))
	row := appendGuardChildExitWitness(j, "claude", "guard-5862", nil, nil, time.Now().Add(-2*time.Second))
	if row.Kind != journal.KindChildExit {
		t.Fatalf("parked teardown row kind = %q, want %q", row.Kind, journal.KindChildExit)
	}
	if row.Reason != journal.CrashCleanExit {
		t.Errorf("parked teardown row reason = %q, want %q — a guard-stopped park is not a crash", row.Reason, journal.CrashCleanExit)
	}
	if row.TraceID != "guard-5862" {
		t.Errorf("row TraceID = %q, want %q: without the session id the row cannot be joined back to the empty session", row.TraceID, "guard-5862")
	}
	if row.ChildExit == nil || row.ChildExit.WallTimeMS < 1000 {
		t.Errorf("row carries no child-exit wall time: %+v", row.ChildExit)
	}
	// The whole point of #5862: the session's journal must not be zero-row.
	if after := len(j.Recent(64)); after != before+1 {
		t.Fatalf("journal rows %d -> %d; a parked session must leave exactly one terminal row behind, not an empty journal", before, after)
	}
}
