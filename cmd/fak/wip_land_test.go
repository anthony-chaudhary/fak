package main

import (
	"context"
	"os"
	"regexp"
	"strings"
	"testing"
)

// landTestRepo builds a throwaway repo (via wipTestRepo) whose HEAD is on `main`, the
// trunk safecommit requires — otherwise safecommit would refuse OFF_TRUNK.
func landTestRepo(t *testing.T) (string, string) {
	t.Helper()
	dir, file := wipTestRepo(t)
	if _, err := gitWipOut(context.Background(), dir, nil, "branch", "-M", "main"); err != nil {
		t.Fatalf("branch -M main: %v", err)
	}
	return dir, file
}

// TestWipLandScope pins the scope selection: dominant top-level segment, deterministic
// lexical tie-break, the "cmd" fallback for a root-only delta, and the never-"wip" rule.
func TestWipLandScope(t *testing.T) {
	cases := []struct {
		name  string
		files []string
		want  string
	}{
		{"dominant dir wins", []string{"cmd/fak/a.go", "cmd/fak/b.go", "internal/x.go"}, "cmd"},
		{"lexical tie-break", []string{"b/y.go", "a/x.go"}, "a"},
		{"root-level file is its own scope", []string{"note.txt"}, "note.txt"},
		{"empty falls back to cmd", nil, "cmd"},
		{"wip top segment is never the scope", []string{"wip/x.go"}, "cmd"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := wipLandScope(tc.files); got != tc.want {
				t.Fatalf("wipLandScope(%v) = %q, want %q", tc.files, got, tc.want)
			}
		})
	}
}

// noClaimMarkers mirrors the closed dos commit-audit no-claim vocabulary (whole word);
// any of these forces the audit to ABSTAIN even over a real source diff.
var noClaimMarkers = regexp.MustCompile(`(?i)\b(wip|misc|cleanup|chore|bump|version|release|merge|revert|format|lint|whitespace|style|nit|nits)\b`)

// codeVerbs is a subset of the audit's _CODE_VERBS that leads code_effect subjects.
var codeVerbs = map[string]bool{"land": true, "fix": true, "add": true, "implement": true, "refactor": true, "restore": true}

// TestWipLandSubjectGradesOKShape witnesses the load-bearing design property of the
// default subject WITHOUT importing the kernel: it must (a) carry no whole-word no-claim
// marker, and (b) lead the post-scope description with a recognized code verb — the two
// conditions that make a source-touching commit grade code_effect / diff-witnessed / OK.
func TestWipLandSubjectGradesOKShape(t *testing.T) {
	subjects := []string{
		wipLandSubject("sess1", []string{"cmd/fak/wip.go"}),
		wipLandSubject("recover-abc", []string{"internal/wipref/wipref.go", "internal/wipref/wipref_test.go"}),
		wipLandSubject("s", []string{"note.txt"}),
	}
	for _, s := range subjects {
		if noClaimMarkers.MatchString(s) {
			t.Fatalf("subject carries a no-claim marker (would ABSTAIN): %q", s)
		}
		lead, ok := postScopeLeadWord(s)
		if !ok {
			t.Fatalf("subject has no `scope: <verb>` shape: %q", s)
		}
		if !codeVerbs[lead] {
			t.Fatalf("subject does not lead with a code verb (would not classify code_effect): lead=%q in %q", lead, s)
		}
	}
}

// postScopeLeadWord extracts the first word after the conventional-commit "): " — the
// token the audit classifies against when the type itself is not a code verb.
func postScopeLeadWord(subject string) (string, bool) {
	i := strings.Index(subject, "): ")
	if i < 0 {
		return "", false
	}
	rest := strings.TrimSpace(subject[i+3:])
	if rest == "" {
		return "", false
	}
	return strings.ToLower(strings.Fields(rest)[0]), true
}

// TestWipLandRecoversAndCommits is the #3876 done-condition end to end: a crashed
// session's checkpoint (delta present only in the ref, wiped from the tree) is
// materialized and committed by explicit pathspec, and the delta lands in HEAD.
func TestWipLandRecoversAndCommits(t *testing.T) {
	ctx := context.Background()
	dir, file := landTestRepo(t)

	dirty := "base line\nan uncommitted edit\n"
	if err := os.WriteFile(file, []byte(dirty), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "landsess", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	// Simulate the crashed owner: the delta lives only in the checkpoint ref now.
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
		t.Fatalf("checkout: %v", err)
	}

	res, code, err := wipLand(ctx, dir, "landsess", "", false)
	if err != nil || code != 0 {
		t.Fatalf("wipLand rc=%d err=%v res=%+v", code, err, res)
	}
	if !res.Committed || !res.Verified {
		t.Fatalf("expected a verified commit, got %+v", res)
	}
	if res.Materialized != "applied" {
		t.Fatalf("materialized = %q, want applied (a wiped tree re-applies forward)", res.Materialized)
	}
	if got := []string{"note.txt"}; len(res.Files) != 1 || res.Files[0] != got[0] {
		t.Fatalf("Files = %v, want %v", res.Files, got)
	}

	// The landed subject is exactly the audit-OK default we generated.
	wantSubject := wipLandSubject("landsess", res.Files)
	if res.Subject != wantSubject {
		t.Fatalf("Subject = %q, want %q", res.Subject, wantSubject)
	}
	gotSubject, err := gitWipOut(ctx, dir, nil, "log", "-1", "--format=%s", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if gotSubject != wantSubject {
		t.Fatalf("HEAD subject = %q, want %q", gotSubject, wantSubject)
	}

	// The delta actually landed: HEAD's note.txt is the checkpointed content.
	head, err := gitWipOut(ctx, dir, nil, "show", "HEAD:note.txt")
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimRight(head, "\n") != strings.TrimRight(dirty, "\n") {
		t.Fatalf("HEAD:note.txt = %q, want %q", head, dirty)
	}
}

// TestWipLandOwnWipCommitsPresentDelta covers the land-your-own-WIP path: the delta is
// already in the working tree (never wiped), so land commits it without re-applying.
func TestWipLandOwnWipCommitsPresentDelta(t *testing.T) {
	ctx := context.Background()
	dir, file := landTestRepo(t)

	if err := os.WriteFile(file, []byte("base line\nlive edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "ownsess", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	// No wipe: the delta is still dirty in the working tree.

	res, code, err := wipLand(ctx, dir, "ownsess", "", false)
	if err != nil || code != 0 {
		t.Fatalf("wipLand rc=%d err=%v res=%+v", code, err, res)
	}
	if res.Materialized != "present" {
		t.Fatalf("materialized = %q, want present (delta already in the tree)", res.Materialized)
	}
	if !res.Committed || !res.Verified {
		t.Fatalf("expected a verified commit, got %+v", res)
	}
}

// TestWipLandDivergedRefuses proves land refuses (TREE_DIVERGED, exit 3) rather than
// clobbering when the working tree neither matches the baseline nor holds the delta.
func TestWipLandDivergedRefuses(t *testing.T) {
	ctx := context.Background()
	dir, file := landTestRepo(t)

	if err := os.WriteFile(file, []byte("base line\nan uncommitted edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "divsess", true, 1000); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	// Diverge the tree: neither the checkpoint's baseline nor its delta.
	if err := os.WriteFile(file, []byte("entirely different content\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, code, err := wipLand(ctx, dir, "divsess", "", false)
	if code != 3 || err == nil {
		t.Fatalf("expected a TREE_DIVERGED refusal (rc=3, err), got rc=%d err=%v", code, err)
	}
	if res.Reason != "TREE_DIVERGED" || res.Committed {
		t.Fatalf("expected TREE_DIVERGED and no commit, got %+v", res)
	}
}

// TestWipLandNoCheckpoint asserts landing a session with no checkpoint is a clean
// runtime error, not a panic or a spurious commit.
func TestWipLandNoCheckpoint(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	res, code, err := wipLand(ctx, dir, "ghost", "", false)
	if code != 1 || err == nil {
		t.Fatalf("expected rc=1 err for a missing checkpoint, got rc=%d err=%v", code, err)
	}
	if res.Committed {
		t.Fatalf("no checkpoint must not commit, got %+v", res)
	}
}
