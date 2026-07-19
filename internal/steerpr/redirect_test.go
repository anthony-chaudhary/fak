package steerpr

// Tests for `fak steer redirect` (#5030): the record/ledger contract (an
// attributable, anchored, append-only, COUNTABLE steering event) and the
// ticket's acceptance gate — the structural proof that NO code path from
// redirect reaches a git mutation. A redirect that can touch the trunk is a
// failed implementation of the affordance regardless of how useful it seems.

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

var redirectNow = time.Date(2026, 7, 18, 4, 0, 0, 0, time.UTC)

func validRedirect(t *testing.T) Redirect {
	t.Helper()
	r, err := NewRedirect("gateway", "op-jane", "aim at the read path, not the write path",
		[]string{"bbb2", "aaa1", "bbb2"}, BandResidual, "#4321", redirectNow)
	if err != nil {
		t.Fatalf("NewRedirect(valid) = %v", err)
	}
	return r
}

// A redirect missing any leg — the unit, the steerer, the NOTE, or the anchor
// SHA set — is refused rather than defaulted: a row missing a leg steers
// nothing. A full row carries the schema, UTC time, and the sorted deduped
// anchor set.
func TestNewRedirectRefusesIncompleteRows(t *testing.T) {
	shas := []string{"aaa1"}
	cases := []struct {
		name           string
		leaf, by, note string
		shas           []string
	}{
		{"no unit", "", "op", "note", shas},
		{"unattributable", "gateway", "", "note", shas},
		{"empty note", "gateway", "op", "   ", shas},
		{"empty anchor set", "gateway", "op", "note", nil},
	}
	for _, tc := range cases {
		if _, err := NewRedirect(tc.leaf, tc.by, tc.note, tc.shas, BandResidual, "", redirectNow); err == nil {
			t.Errorf("NewRedirect(%s) = nil error, want refusal", tc.name)
		}
	}

	r := validRedirect(t)
	if r.Schema != RedirectSchema {
		t.Fatalf("schema = %q, want %q", r.Schema, RedirectSchema)
	}
	if got := []string{"aaa1", "bbb2"}; len(r.SHAs) != 2 || r.SHAs[0] != got[0] || r.SHAs[1] != got[1] {
		t.Fatalf("anchor SHAs = %v, want sorted deduped %v", r.SHAs, got)
	}
	if r.At != "2026-07-18T04:00:00Z" {
		t.Fatalf("At = %q, want RFC3339 UTC", r.At)
	}
	if r.Band != BandResidual || r.Issue != "#4321" {
		t.Fatalf("anchor band/issue = %q/%q, want RESIDUAL/#4321", r.Band, r.Issue)
	}
}

// The ledger is append-only, best-effort on load, refuses incomplete rows,
// and makes the redirect COUNTABLE per unit (the #5030 "first-class,
// countable steering event" leg). It also lives under .fak — runtime state
// beside the other ledgers, never inside .git.
func TestRedirectLedgerAppendOnlyAndCountable(t *testing.T) {
	root := t.TempDir()
	path := RedirectLedgerPath(root)
	if !strings.Contains(path, ".fak") || strings.Contains(path, ".git") {
		t.Fatalf("ledger path %q must live under .fak, never .git", path)
	}
	if rows := LoadRedirects(path); rows != nil {
		t.Fatalf("missing ledger loaded %d row(s), want empty", len(rows))
	}

	first := validRedirect(t)
	first.FollowUp = "#4321"
	if err := AppendRedirect(path, first); err != nil {
		t.Fatalf("append first: %v", err)
	}
	raw1, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	second, err := NewRedirect("gateway", "op-alex", "second re-aim: fold into the dispatcher",
		[]string{"ccc3"}, BandCleared, "", redirectNow.Add(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if err := AppendRedirect(path, second); err != nil {
		t.Fatalf("append second: %v", err)
	}
	other, err := NewRedirect("dojo", "op-jane", "different unit", []string{"ddd4"}, BandUnverifiable, "", redirectNow)
	if err != nil {
		t.Fatal(err)
	}
	if err := AppendRedirect(path, other); err != nil {
		t.Fatalf("append other-unit row: %v", err)
	}

	raw2, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(raw2), string(raw1)) {
		t.Fatal("append-only violated: the earlier ledger row was rewritten")
	}

	rows := LoadRedirects(path)
	if len(rows) != 3 {
		t.Fatalf("loaded %d row(s), want 3", len(rows))
	}
	if got := RedirectsFor(rows, "gateway"); len(got) != 2 || got[0].By != "op-jane" || got[1].By != "op-alex" {
		t.Fatalf("RedirectsFor(gateway) = %d row(s) %v, want the 2 gateway steers oldest-first", len(got), got)
	}
	if got := RedirectsFor(rows, "dojo"); len(got) != 1 {
		t.Fatalf("RedirectsFor(dojo) = %d row(s), want 1", len(got))
	}
	var back Redirect
	if err := json.Unmarshal([]byte(strings.Split(strings.TrimSpace(string(raw2)), "\n")[0]), &back); err != nil {
		t.Fatalf("ledger row does not round-trip: %v", err)
	}
	if back.FollowUp != "#4321" || back.Note != first.Note {
		t.Fatalf("round-tripped row = %#v, want follow-up + note preserved", back)
	}

	// An incomplete row is refused and writes nothing.
	if err := AppendRedirect(path, Redirect{Leaf: "gateway", By: "op", SHAs: []string{"a"}}); err == nil {
		t.Fatal("AppendRedirect(no note) = nil, want refusal")
	}
	if after := LoadRedirects(path); len(after) != 3 {
		t.Fatalf("a refused append changed the ledger: %d row(s)", len(after))
	}
}

// The filed follow-up carries the full anchor: the note, the unit, the exact
// member SHA set, the band at redirect time — and says out loud that the
// landed history stays put.
func TestRedirectFollowUpCarriesNoteSHAsAndBand(t *testing.T) {
	r := validRedirect(t)
	body := r.FollowUpBody()
	for _, want := range []string{
		r.Note, "`gateway`", "`aaa1`", "`bbb2`", string(BandResidual), "#4321", "op-jane", r.At,
		"never reverts, rewrites",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("follow-up body missing %q:\n%s", want, body)
		}
	}
	if title := r.FollowUpTitle(); !strings.Contains(title, "gateway") || !strings.Contains(title, "redirect") {
		t.Errorf("follow-up title %q should name the unit and the act", title)
	}
}

// ---- The #5030 acceptance gate: no code path from redirect reaches git ----

// redirectReadOnlyGitVerbs is the closed allowlist of git subcommands the
// redirect path (and the steer verb file that hosts it) may reach through the
// git seam: reads only. A mutating verb (commit, push, reset, revert, merge,
// rebase, tag, checkout, restore, apply, stash, clean, update-ref, ...) must
// never be added.
var redirectReadOnlyGitVerbs = map[string]bool{
	"log": true, "rev-parse": true, "show": true, "diff": true,
	"ls-files": true, "status": true, "for-each-ref": true,
	"merge-base": true, "cat-file": true, "describe": true,
	// `config user.name` — the attribution read. Still a read.
	"config": true,
}

func redirectStringLit(e ast.Expr) (string, bool) {
	lit, ok := e.(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", false
	}
	s, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return s, true
}

// redirectScanVerbShell reports every way the scanned file could mutate git:
// a subprocess whose tool/verb is not a provably read-only literal
// (`git <read-verb>` / `dos commit-audit`), a git-seam call with a
// non-literal or mutating verb, or a ghexec seam call outside the `gh issue`
// family. Fail closed: anything unprovable is a violation.
func redirectScanVerbShell(fset *token.FileSet, f *ast.File) []string {
	gitSeams := map[string]bool{"releasePRPlanGit": true, "releaseStatusGitOutput": true}
	ghSeamVerbIndex := map[string]int{"ghexec.Command": 1, "ghexec.CommandTimeout": 2}
	var violations []string
	report := func(pos token.Pos, msg string) {
		violations = append(violations, fset.Position(pos).String()+": "+msg)
	}
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		var callee string
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			callee = fun.Name
		case *ast.SelectorExpr:
			if x, ok := fun.X.(*ast.Ident); ok {
				callee = x.Name + "." + fun.Sel.Name
			}
		}
		switch {
		case callee == "exec.Command" || callee == "exec.CommandContext":
			args := call.Args
			if callee == "exec.CommandContext" {
				if len(args) < 1 {
					return true
				}
				args = args[1:]
			}
			if len(args) == 0 {
				return true
			}
			tool, ok := redirectStringLit(args[0])
			if !ok {
				report(call.Pos(), "subprocess with a NON-LITERAL command name — unprovable, refused")
				return true
			}
			switch tool {
			case "git":
				if len(args) < 2 {
					report(call.Pos(), "bare `git` with no subcommand — unprovable, refused")
					return true
				}
				if verb, ok := redirectStringLit(args[1]); !ok || !redirectReadOnlyGitVerbs[verb] {
					report(call.Pos(), "`git "+verb+"` — not a provably read-only verb; the redirect path must NEVER mutate git")
				}
			case "dos":
				if len(args) < 2 {
					report(call.Pos(), "bare `dos` — unprovable, refused")
					return true
				}
				if verb, ok := redirectStringLit(args[1]); !ok || verb != "commit-audit" {
					report(call.Pos(), "`dos` outside the read-only commit-audit query")
				}
			default:
				report(call.Pos(), "unexpected tool `"+tool+"` — only read-only git/dos queries and the ghexec seam are allowed")
			}
		case gitSeams[callee]:
			if len(call.Args) < 2 {
				report(call.Pos(), "git seam "+callee+" with no subcommand — unprovable, refused")
				return true
			}
			if verb, ok := redirectStringLit(call.Args[1]); !ok || !redirectReadOnlyGitVerbs[verb] {
				report(call.Pos(), "git seam "+callee+" with `"+verb+"` — not a provably read-only verb; the redirect path must NEVER mutate git")
			}
		case ghSeamVerbIndex[callee] > 0:
			idx := ghSeamVerbIndex[callee]
			if len(call.Args) <= idx {
				report(call.Pos(), "gh seam "+callee+" with no verb — unprovable, refused")
				return true
			}
			if verb, ok := redirectStringLit(call.Args[idx]); !ok || verb != "issue" {
				report(call.Pos(), "gh seam "+callee+" outside the `gh issue` family — the redirect's only outward act is the issue follow-up")
			}
		}
		return true
	})
	return violations
}

// TestRedirectNeverReachesGitMutation is #5030's acceptance gate, made
// structural at both ends of the redirect path:
//
//   - The leaf (redirect.go) can never reach git AT ALL: no subprocess, no
//     network, no internal import — the record and the ledger are pure state.
//   - The verb shell (cmd/fak/steer_prs.go, which hosts runSteerRedirect and
//     the gh seam) launches only provably read-only subprocesses: literal
//     `git <read-verb>` / `dos commit-audit` / the deadlined `gh issue` seam.
//     A `git commit`, `git push`, a laundered seam mutation, or any
//     unprovable call shape reds this test.
func TestRedirectNeverReachesGitMutation(t *testing.T) {
	fset := token.NewFileSet()

	// The leaf.
	leaf, err := parser.ParseFile(fset, "redirect.go", nil, 0)
	if err != nil {
		t.Fatalf("parse redirect.go: %v", err)
	}
	forbidden := map[string]bool{"os/exec": true, "syscall": true, "net": true, "net/http": true, "os/signal": true}
	for _, imp := range leaf.Imports {
		path, _ := strconv.Unquote(imp.Path.Value)
		if forbidden[path] {
			t.Errorf("redirect.go imports %q — the redirect leaf must have no subprocess/network vector to git", path)
		}
		if strings.Contains(path, "github.com/anthony-chaudhary/fak/") {
			t.Errorf("redirect.go imports internal package %q — the leaf stays pure so nothing impure can hide behind it", path)
		}
	}

	// The verb shell that hosts runSteerRedirect.
	shellPath := filepath.Join("..", "..", "cmd", "fak", "steer_prs.go")
	shell, err := parser.ParseFile(fset, shellPath, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", shellPath, err)
	}
	hosts := map[string]bool{}
	for _, d := range shell.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok {
			hosts[fn.Name.Name] = true
		}
	}
	// Non-vacuous: the scan covers the file that actually implements redirect.
	for _, want := range []string{"runSteerRedirect", "ghSteerRedirectFollowUp"} {
		if !hosts[want] {
			t.Fatalf("%s does not declare %s — the no-mutation scan would be scanning the wrong surface", shellPath, want)
		}
	}
	for _, v := range redirectScanVerbShell(fset, shell) {
		t.Error(v)
	}
}

// TestRedirectGitMutationDetectorBites is the guard's own witness: a
// structural test that has never been seen to fail proves nothing. The
// detector is fed the exact shapes the fence exists to refuse — a direct
// `git commit` exec, a `git push` laundered through the seam, a gh seam call
// outside the issue family — and MUST red on each; and a read-only fixture on
// which it MUST stay silent.
func TestRedirectGitMutationDetectorBites(t *testing.T) {
	const blocking = `package main

import "os/exec"

func redirectGoneWrong() {
	_ = exec.Command("git", "commit", "--amend", "--no-edit")
	_ = releasePRPlanGit(".", "push", "--force")
	_, _ = ghexec.CommandTimeout(nil, 0, "repo", "delete")
}
`
	const readOnly = `package main

import "os/exec"

func redirectDoneRight() {
	_ = exec.Command("dos", "commit-audit", "a..b", "--json")
	_ = releasePRPlanGit(".", "log", "--no-merges")
	_ = releasePRPlanGit(".", "config", "user.name")
	_, _ = ghexec.CommandTimeout(nil, 0, "issue", "create")
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "blocking_fixture.go", blocking, 0)
	if err != nil {
		t.Fatalf("parse blocking fixture: %v", err)
	}
	got := redirectScanVerbShell(fset, f)
	if len(got) != 3 {
		t.Fatalf("detector must red on all 3 wired mutations (git commit, seam push, off-family gh); got %d: %v", len(got), got)
	}
	joined := strings.Join(got, "\n")
	for _, needle := range []string{"git commit", "push", "issue"} {
		if !strings.Contains(joined, needle) {
			t.Errorf("detector violations do not name the wired mutation %q:\n%s", needle, joined)
		}
	}

	f, err = parser.ParseFile(fset, "readonly_fixture.go", readOnly, 0)
	if err != nil {
		t.Fatalf("parse read-only fixture: %v", err)
	}
	if got := redirectScanVerbShell(fset, f); len(got) != 0 {
		t.Errorf("detector reds on provably read-only calls — it would fence the redirect's own job: %v", got)
	}
}
