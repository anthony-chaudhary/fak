package wipref

import (
	"reflect"
	"strings"
	"testing"
)

// markers builds a conflict region at runtime instead of embedding one literally, so
// this file never itself contains a line that starts with a conflict marker — the
// tree-wide marker scans (and this package's own detector) would otherwise read the
// test that proves the detector as a conflicted file.
func markers(ours, theirs string) string {
	open, mid, shut := strings.Repeat("<", 7), strings.Repeat("=", 7), strings.Repeat(">", 7)
	return "l1\n" + open + " ours\n" + ours + "\n" + mid + "\n" + theirs + "\n" + shut + " theirs\nl2\n"
}

// TestApplyLadderIsStrictFirst pins the ordering #4337 calls load-bearing: an unmoved
// baseline must be taken by the context-strict rung and must never pay for — or pick
// up the metadata of — a three-way merge it did not need.
func TestApplyLadderIsStrictFirst(t *testing.T) {
	l := ApplyLadder()
	if len(l) != 2 {
		t.Fatalf("expected a two-rung ladder (strict, 3way), got %d", len(l))
	}
	if l[0].ThreeWay {
		t.Fatalf("the FIRST rung must be the strict apply, got %+v", l[0])
	}
	if got := strings.Join(l[0].Args, " "); strings.Contains(got, "--3way") {
		t.Fatalf("the strict rung must not carry --3way: %q", got)
	}
	if l[0].OK != "applied" {
		t.Fatalf("strict success grades %q, want %q", l[0].OK, "applied")
	}
	if !l[1].ThreeWay || !strings.Contains(strings.Join(l[1].Args, " "), "--3way") {
		t.Fatalf("the SECOND rung must be the three-way fallback, got %+v", l[1])
	}
	if l[1].OK != "merged" {
		t.Fatalf("three-way success grades %q, want %q", l[1].OK, "merged")
	}
}

// TestCheckLadderForwardTiersReverseDoesNot proves the discriminator gains a 3-way tier
// FORWARD (the RECLAIM-vs-QUARANTINE question) while the reverse "already present"
// question stays strict — a delta that merely merges backwards is not present, and
// grading it present would make land skip a materialize it still owes.
func TestCheckLadderForwardTiersReverseDoesNot(t *testing.T) {
	fwd := CheckLadder(false)
	if len(fwd) != 2 || fwd[0].ThreeWay || !fwd[1].ThreeWay {
		t.Fatalf("forward check ladder must be strict-then-3way, got %+v", fwd)
	}
	for _, r := range fwd {
		if !contains(r.Args, "--check") {
			t.Fatalf("a check rung must carry --check: %+v", r)
		}
	}
	rev := CheckLadder(true)
	if len(rev) != 1 || rev[0].ThreeWay {
		t.Fatalf("reverse check must be strict-only, got %+v", rev)
	}
	if !contains(rev[0].Args, "-R") || rev[0].OK != "present" {
		t.Fatalf("reverse rung must be -R grading present, got %+v", rev[0])
	}
}

// TestThreeWayIndexSetupSeedsAndRefreshes pins both halves of the throwaway-index
// recipe. read-tree alone leaves the stat cache empty, and --3way then rejects an
// untouched file as "does not match index" — measured against git 2.51 — so dropping
// the refresh silently disables the whole three-way tier.
func TestThreeWayIndexSetupSeedsAndRefreshes(t *testing.T) {
	setup := ThreeWayIndexSetup()
	if len(setup) != 2 {
		t.Fatalf("expected read-tree + update-index --refresh, got %+v", setup)
	}
	if setup[0][0] != "read-tree" || !contains(setup[0], "HEAD") {
		t.Fatalf("first step must seed the scratch index from HEAD, got %+v", setup[0])
	}
	if setup[1][0] != "update-index" || !contains(setup[1], "--refresh") {
		t.Fatalf("second step must refresh the stat cache, got %+v", setup[1])
	}
}

// TestHasConflictMarkersRequiresBothAnchored is the anti-false-positive proof. A bare
// `=======` rule is an ordinary setext heading and appears throughout this repo's docs;
// grading that as a conflict would refuse honest lands forever.
func TestHasConflictMarkersRequiresBothAnchored(t *testing.T) {
	cases := []struct {
		name string
		data string
		want bool
	}{
		{"clean", "l1\nl2\nl3\n", false},
		{"real conflict", markers("PEER", "SESSION"), true},
		{"setext heading only", "Title\n" + strings.Repeat("=", 7) + "\nbody\n", false},
		{"opening marker only", strings.Repeat("<", 7) + " ours\nl1\n", false},
		{"closing marker only", "l1\n" + strings.Repeat(">", 7) + " theirs\n", false},
		{"mid-line mention", "see " + strings.Repeat("<", 7) + " and " + strings.Repeat(">", 7) + " markers\n", false},
		{"no trailing newline", strings.TrimSuffix(markers("a", "b"), "\n"), true},
		{"empty", "", false},
	}
	for _, tc := range cases {
		if got := HasConflictMarkers([]byte(tc.data)); got != tc.want {
			t.Errorf("%s: HasConflictMarkers = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestConflictedPathsIsOrderedAndContentDriven proves the shell gets a deterministic,
// content-derived list — the file set land surfaces on a MERGE_CONFLICT refusal.
func TestConflictedPathsIsOrderedAndContentDriven(t *testing.T) {
	order := []string{"a.go", "b.go", "c.go"}
	content := map[string][]byte{
		"a.go": []byte(markers("x", "y")),
		"b.go": []byte("clean\n"),
		"c.go": []byte(markers("p", "q")),
	}
	got := ConflictedPaths(order, content)
	if want := []string{"a.go", "c.go"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ConflictedPaths = %v, want %v", got, want)
	}
	if n := len(ConflictedPaths(order, map[string][]byte{})); n != 0 {
		t.Fatalf("absent content must grade clean, got %d conflicted", n)
	}
}

// TestGradeApplyFoldsTheLadder walks every reachable ladder outcome, including the two
// #4337 exists to separate: a stale-but-mergeable delta must grade "merged" (not
// "conflict"), and a conflicted merge must grade "merged_with_conflicts" (not
// "applied", and not "conflict" — it IS in the tree, it just may not be committed).
func TestGradeApplyFoldsTheLadder(t *testing.T) {
	cases := []struct {
		name          string
		strictOK      bool
		threeWayClean bool
		conflicted    []string
		want          string
	}{
		{"unmoved baseline", true, false, nil, "applied"},
		{"stale base, merges clean", false, true, nil, "merged"},
		{"stale base, conflicts", false, false, []string{"f.txt"}, "merged_with_conflicts"},
		{"nothing applied", false, false, nil, "conflict"},
		// Content beats exit status: a rung that reported success while leaving
		// markers is still a conflicted merge, or land would commit them.
		{"clean exit but markers present", false, true, []string{"f.txt"}, "merged_with_conflicts"},
		{"strict ok but markers present", true, false, []string{"f.txt"}, "merged_with_conflicts"},
	}
	for _, tc := range cases {
		if got := GradeApply(tc.strictOK, tc.threeWayClean, tc.conflicted); got != tc.want {
			t.Errorf("%s: GradeApply = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// TestStateCuts pins the two DIFFERENT cuts across the vocabulary. Exactly one state —
// the conflicted merge — is recoverable but not committable; collapsing the two cuts
// into one is what turns a mergeable delta into a lost one.
func TestStateCuts(t *testing.T) {
	cases := []struct {
		state       string
		inTree      bool
		committable bool
	}{
		{"applied", true, true},
		{"present", true, true},
		{"merged", true, true},
		{"merged_with_conflicts", true, false},
		{"conflict", false, false},
		{"bogus", false, false}, // fail-closed on an unknown state
		{"", false, false},
	}
	for _, tc := range cases {
		if got := InTree(tc.state); got != tc.inTree {
			t.Errorf("InTree(%q) = %v, want %v", tc.state, got, tc.inTree)
		}
		if got := Committable(tc.state); got != tc.committable {
			t.Errorf("Committable(%q) = %v, want %v", tc.state, got, tc.committable)
		}
	}
	if Committable("merged_with_conflicts") {
		t.Fatal("merged_with_conflicts must never be committable — that is the MERGE_CONFLICT guard")
	}
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}
