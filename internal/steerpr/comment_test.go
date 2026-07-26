package steerpr

// Tests for `fak steer comment` (#5029): the record/ledger contract (an
// attributable, anchored, append-only record of operator attention), the
// ticket's two acceptance gates — an UNBOUND unit refuses rather than posting
// somewhere plausible, and the posted body is anchored to the exact member SHA
// SET rather than to a unit name — and the structural proof that the annotate
// rung stays annotate-only: no code path from comment reaches a git mutation.

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

var commentNow = time.Date(2026, 7, 26, 9, 30, 0, 0, time.UTC)

func validComment(t *testing.T) Comment {
	t.Helper()
	c, err := NewComment("gateway", "op-jane", "the retry budget here looks like it double-counts a shed tick",
		[]string{"bbb2", "aaa1", "bbb2"}, BandResidual, "#4321", commentNow)
	if err != nil {
		t.Fatalf("NewComment(valid) = %v", err)
	}
	return c
}

// A comment missing any leg — the unit, the annotator, the NOTE, or the anchor
// SHA set — is refused rather than defaulted. A full row carries the schema,
// UTC time, and the sorted deduped anchor set.
func TestNewCommentRefusesIncompleteRows(t *testing.T) {
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
		if _, err := NewComment(tc.leaf, tc.by, tc.note, tc.shas, BandResidual, "#7", commentNow); err == nil {
			t.Errorf("NewComment(%s) = nil error, want refusal", tc.name)
		}
	}

	c := validComment(t)
	if c.Schema != CommentSchema {
		t.Fatalf("schema = %q, want %q", c.Schema, CommentSchema)
	}
	if want := []string{"aaa1", "bbb2"}; len(c.SHAs) != 2 || c.SHAs[0] != want[0] || c.SHAs[1] != want[1] {
		t.Fatalf("anchor SHAs = %v, want sorted deduped %v", c.SHAs, want)
	}
	if c.At != "2026-07-26T09:30:00Z" {
		t.Fatalf("At = %q, want RFC3339 UTC", c.At)
	}
	if c.Band != BandResidual || c.Issue != "#4321" || c.IssueNumber() != 4321 {
		t.Fatalf("anchor band/issue = %q/%q (n=%d), want RESIDUAL/#4321/4321", c.Band, c.Issue, c.IssueNumber())
	}
}

// #5029's first acceptance gate: a unit with NO closure-grade binding refuses
// rather than posting somewhere plausible. A unit's Mentions are not a binding,
// so anything that is not a real "#N" — empty, a bare number, a URL, a
// non-positive ref — leaves the annotation no honest place to live. The refusal
// must SAY so rather than defaulting to an issue nobody chose.
func TestNewCommentRefusesAnUnboundUnit(t *testing.T) {
	for _, unbound := range []string{"", "   ", "4321", "#", "#0", "#-3", "#abc", "https://github.com/o/r/issues/4321"} {
		c, err := NewComment("gateway", "op-jane", "a real note", []string{"aaa1"}, BandResidual, unbound, commentNow)
		if err == nil {
			t.Errorf("NewComment(issue=%q) = nil error, want refusal — an unbound unit has no honest place to post (row posted to %q)", unbound, c.Issue)
			continue
		}
		if !strings.Contains(err.Error(), "bound") {
			t.Errorf("NewComment(issue=%q) error = %q, want it to name the missing binding", unbound, err)
		}
	}

	// The ledger refuses the same shape independently: a row that reached the
	// ledger unbound would be operator reasoning stored against no intent.
	unbound := validComment(t)
	unbound.Issue = ""
	if err := AppendComment(CommentLedgerPath(t.TempDir()), unbound); err == nil {
		t.Error("AppendComment(unbound row) = nil error, want refusal")
	}
}

// The ledger is append-only, best-effort on load, refuses incomplete rows, and
// makes operator attention COUNTABLE per unit (the "brief/loop can see that a
// unit received operator attention" leg). It lives under .fak — runtime state
// beside the other ledgers, never inside .git.
func TestCommentLedgerAppendOnlyAndCountable(t *testing.T) {
	root := t.TempDir()
	path := CommentLedgerPath(root)
	if !strings.Contains(path, ".fak") || strings.Contains(path, ".git") {
		t.Fatalf("ledger path %q must live under .fak, never .git", path)
	}
	if rows := LoadComments(path); rows != nil {
		t.Fatalf("missing ledger loaded %d row(s), want empty", len(rows))
	}

	first := validComment(t)
	first.Posted = "https://github.com/o/r/issues/4321#issuecomment-1"
	if err := AppendComment(path, first); err != nil {
		t.Fatalf("append first: %v", err)
	}
	raw1, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	second, err := NewComment("gateway", "op-alex", "agreed, and the shed path needs the same read",
		[]string{"ccc3"}, BandCleared, "#4321", commentNow.Add(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if err := AppendComment(path, second); err != nil {
		t.Fatalf("append second: %v", err)
	}
	other, err := NewComment("dojo", "op-jane", "different unit", []string{"ddd4"}, BandUnverifiable, "#99", commentNow)
	if err != nil {
		t.Fatal(err)
	}
	if err := AppendComment(path, other); err != nil {
		t.Fatalf("append other-unit row: %v", err)
	}

	// Append-only: the earlier bytes are still a prefix, never rewritten.
	raw2, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(raw2), string(raw1)) {
		t.Fatal("appending rewrote earlier ledger bytes; the ledger must be append-only")
	}

	rows := LoadComments(path)
	if len(rows) != 3 {
		t.Fatalf("loaded %d row(s), want 3", len(rows))
	}
	if got := CommentsFor(rows, "gateway"); len(got) != 2 {
		t.Fatalf("CommentsFor(gateway) = %d, want 2 — annotations must be countable per unit", len(got))
	} else if got[0].By != "op-jane" || got[1].By != "op-alex" {
		t.Fatalf("CommentsFor(gateway) order = %q,%q, want oldest first", got[0].By, got[1].By)
	}
	if got := CommentsFor(rows, "dojo"); len(got) != 1 {
		t.Fatalf("CommentsFor(dojo) = %d, want 1", len(got))
	}
	if first.Posted != rows[0].Posted {
		t.Fatalf("Posted = %q, want the round-tripped %q", rows[0].Posted, first.Posted)
	}

	// A torn or foreign line is skipped rather than poisoning its neighbours,
	// and a failed read never invents operator attention.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("{not json\n" + `{"schema":"fak.steerpr.redirect.v1","leaf":"gateway"}` + "\n"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if rows := LoadComments(path); len(rows) != 3 {
		t.Fatalf("after a torn + foreign line, loaded %d row(s), want the same 3", len(rows))
	}
}

// #5029's second acceptance gate: the posted comment names the exact member SHA
// SET the operator was looking at, not just a unit name. A unit name means
// different commits tomorrow; the SHA set is what was actually read. The body
// also states in-band that it changes nothing (annotate-only), and rendering it
// is pure — producing the text must have no side effect on the record.
func TestCommentBodyAnchorsToTheSHASetNotJustTheUnitName(t *testing.T) {
	c := validComment(t)
	body := c.Body()
	for _, want := range []string{c.Note, "gateway", "aaa1", "bbb2", string(BandResidual), "op-jane", c.At} {
		if !strings.Contains(body, want) {
			t.Errorf("comment body is missing %q — the annotation must be anchored to what was read:\n%s", want, body)
		}
	}
	// Non-vacuous: a body that merely named the unit would still contain
	// "gateway". Pin that every anchor SHA is present INDIVIDUALLY.
	for _, sha := range c.SHAs {
		if strings.Count(body, sha) == 0 {
			t.Errorf("anchor SHA %q absent from the posted body", sha)
		}
	}
	if !strings.Contains(strings.ToUpper(body), "ANNOTATION") {
		t.Errorf("comment body must say it is an annotation (it changes no band and nothing that landed):\n%s", body)
	}

	before, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	_ = c.Body()
	after, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("rendering the body mutated the record; the render must be pure")
	}
}

// The annotate rung stays ANNOTATE-ONLY, made structural at both ends of the
// comment path:
//
//   - The leaf (comment.go) can never reach git AT ALL: no subprocess, no
//     network, no internal import — the record and the ledger are pure state.
//   - The verb shell (cmd/fak/steer_comment.go) launches only provably
//     read-only subprocesses: literal `git <read-verb>` and the deadlined
//     `gh issue` seam. A `git commit`, a `git push`, a laundered seam mutation,
//     or any unprovable call shape reds this test.
//
// It also pins that the leaf exposes no band/ack mutator: a comment that could
// move the machine band is a failed implementation of the affordance regardless
// of how useful it seems.
func TestCommentNeverReachesGitMutation(t *testing.T) {
	fset := token.NewFileSet()

	// The leaf.
	leaf, err := parser.ParseFile(fset, "comment.go", nil, 0)
	if err != nil {
		t.Fatalf("parse comment.go: %v", err)
	}
	forbidden := map[string]bool{"os/exec": true, "syscall": true, "net": true, "net/http": true, "os/signal": true}
	for _, imp := range leaf.Imports {
		path, _ := strconv.Unquote(imp.Path.Value)
		if forbidden[path] {
			t.Errorf("comment.go imports %q — the comment leaf must have no subprocess/network vector to git", path)
		}
		if strings.Contains(path, "github.com/anthony-chaudhary/fak/") {
			t.Errorf("comment.go imports internal package %q — the leaf stays pure so nothing impure can hide behind it", path)
		}
	}

	// The verb shell that hosts runSteerComment, scanned by the same detector
	// the redirect rung is fenced with.
	shellPath := filepath.Join("..", "..", "cmd", "fak", "steer_comment.go")
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
	// Non-vacuous: the scan covers the file that actually implements comment.
	for _, want := range []string{"runSteerComment", "ghSteerCommentPost"} {
		if !hosts[want] {
			t.Fatalf("%s does not declare %s — the no-mutation scan would be scanning the wrong surface", shellPath, want)
		}
	}
	for _, v := range redirectScanVerbShell(fset, shell) {
		t.Error(v)
	}
}
