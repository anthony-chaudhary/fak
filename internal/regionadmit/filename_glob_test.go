package regionadmit

import "testing"

func TestFilenameGlobDoesNotClaimUnrelatedConcreteSibling(t *testing.T) {
	tax := Taxonomy{Trees: map[string][]string{
		"cmd":      {"cmd/**"},
		"accounts": {"internal/accounts/**", "cmd/fak/accounts*.go"},
	}, Exclusive: map[string]bool{}}
	for _, tt := range []struct{ path, lane string }{
		{"cmd/fak/cudaarch_test.go", "cmd"},
		{"cmd/fak/claude_glm_gcp_test.go", "cmd"},
		{"cmd/fak/accounts.go", "accounts"},
		{"cmd/fak/accounts_test.go", "accounts"},
		{"cmd/fak/accounts/deep.go", "cmd"},
	} {
		if got := LaneOf([]string{tt.path}, tax); got != tt.lane {
			t.Errorf("LaneOf(%q) = %q, want %q", tt.path, got, tt.lane)
		}
	}
	live := []Lease{{ID: "benchlineagegate-fix", Holder: "other", Tree: []string{"cmd/coalescebench/**", "cmd/modelbench/**", "cmd/mtpbench/**"}}}
	if got := Decide(Request{Actor: "me", Lane: "cmd", Tree: []string{"cmd/fak/cudaarch_test.go"}}, live, tax); !got.Admit {
		t.Fatalf("disjoint narrowed cmd regions refused: %+v", got)
	}
	if got := Decide(Request{Actor: "me", Lane: "cmd"}, live, tax); got.Admit || got.Rung != RungSameLane {
		t.Fatalf("whole cmd lane must remain serialized: %+v", got)
	}
	if got := Decide(Request{Actor: "me", Lane: "cmd", Tree: []string{"cmd/modelbench/main.go"}}, live, tax); got.Admit || got.Rung != RungTreeOverlap {
		t.Fatalf("overlapping narrowed cmd regions must refuse: %+v", got)
	}
	if strictlyWithinLane([]string{"cmd/fak/accounts_test.go"}, "cmd", tax) {
		t.Fatal("actual accounts filename must retain its tighter lane")
	}
}

func TestFilenameGlobPreservesConservativeAndExclusiveClassification(t *testing.T) {
	tax := Taxonomy{Trees: map[string][]string{"cmd": {"cmd/**"}, "accounts": {"cmd/fak/accounts*.go"}}, Exclusive: map[string]bool{"accounts": true}}
	resolved := LaneOf([]string{"cmd/fak/accounts_test.go"}, tax)
	if resolved != "accounts" || !tax.Exclusive[resolved] {
		t.Fatalf("matching filename lost exclusive lane classification: %q", resolved)
	}
	// Isolate this classifier's contract from LaneForPath's separate precedence.
	if got := Decide(Request{Actor: "me", Lane: "cmd", Tree: []string{"cmd/other/main.go"}}, []Lease{{ID: "accounts", Holder: "other", Lane: resolved, Tree: []string{"cmd/fak/accounts_test.go"}}}, tax); got.Admit || got.Rung != RungExclusiveLive {
		t.Fatalf("matching exclusive filename lease must block: %+v", got)
	}
	if got := LaneOf([]string{"cmd/fak/unrelated*.go"}, tax); got != "accounts" {
		t.Fatalf("non-concrete containment must retain conservative prior behavior, got %q", got)
	}
	tax.Trees["accounts"] = []string{"cmd/fak/accounts[.go"}
	if got := LaneOf([]string{"cmd/fak/cudaarch_test.go"}, tax); got != "accounts" {
		t.Fatalf("malformed pattern must retain conservative prior behavior, got %q", got)
	}
}
