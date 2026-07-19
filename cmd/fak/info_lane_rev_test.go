package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

// TestGuardInfoStartupHeaderLaneTag is the #2491 pane-render witness: the info header shows the
// active lane's module rev IMMEDIATELY BESIDE the binary stamp when a lane resolves, and is left
// byte-for-byte at its pre-#2491 shape (no stray "lane"/"@r" text on the header line) when it does
// not. The tag is additive — its absence must never perturb the header an operator already reads.
func TestGuardInfoStartupHeaderLaneTag(t *testing.T) {
	const laneTag = "lane internal/modver@r42"
	headerLine := func(s string) string {
		if i := strings.IndexByte(s, '\n'); i >= 0 {
			return s[:i]
		}
		return s
	}

	withTag := headerLine(guardInfoStartupHeader("http://127.0.0.1:8321", laneTag, 2*time.Second, 0))
	// "beside the binary stamp": the lane tag renders on the header line, adjacent to and after
	// the version stamp — asserting the exact join guardInfoStartupHeader builds.
	if want := guardInfoVersionTag() + " · " + laneTag; !strings.Contains(withTag, want) {
		t.Fatalf("lane rev tag not rendered beside the binary stamp\n  want substring: %q\n  header line:    %q", want, withTag)
	}

	// Empty tag → the header line carries no lane text and no stray separator.
	withoutTag := headerLine(guardInfoStartupHeader("http://127.0.0.1:8321", "", 2*time.Second, 0))
	if strings.Contains(withoutTag, "lane ") || strings.Contains(withoutTag, "@r") {
		t.Fatalf("empty lane tag must not render any lane text on the header line:\n%s", withoutTag)
	}
}

// TestGuardInfoLaneRevTag exercises the resolver arms behind the tag: a working set under one
// module resolves to that module's rev; an untracked path is skipped (does not defeat a clean
// single-lane resolution); a set spanning two modules is ambiguous and stays silent; and an empty
// or wholly-untracked set resolves to nothing.
func TestGuardInfoLaneRevTag(t *testing.T) {
	rep := modver.Report{Modules: []modver.Module{
		{Name: "internal/modver", Rev: 42},
		{Name: "cmd/fak", Rev: 7},
	}}
	cases := []struct {
		name    string
		changed []string
		want    string
	}{
		{"single module", []string{"internal/modver/modver.go"}, "lane internal/modver@r42"},
		{"module root exact", []string{"cmd/fak"}, "lane cmd/fak@r7"},
		{"backslash path normalized", []string{`internal\modver\snapshot.go`}, "lane internal/modver@r42"},
		{"untracked path skipped", []string{"internal/modver/x.go", "README.md"}, "lane internal/modver@r42"},
		{"spans two modules is silent", []string{"internal/modver/x.go", "cmd/fak/info.go"}, ""},
		{"no tracked module", []string{"README.md"}, ""},
		{"empty working set", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardInfoLaneRevTag(rep, tc.changed); got != tc.want {
				t.Fatalf("guardInfoLaneRevTag(%v) = %q, want %q", tc.changed, got, tc.want)
			}
		})
	}
}

// TestGuardInfoWorkingSet checks the -z porcelain parse: plain edits contribute their path, and a
// rename entry contributes its NEW path while its origin follower is dropped.
func TestGuardInfoWorkingSet(t *testing.T) {
	// " M internal/modver/modver.go\0R  cmd/fak/new.go\0cmd/fak/old.go\0"
	out := " M internal/modver/modver.go\x00R  cmd/fak/new.go\x00cmd/fak/old.go\x00"
	run := func(_ ...string) ([]byte, error) { return []byte(out), nil }
	got, err := guardInfoWorkingSetFrom(run)
	if err != nil {
		t.Fatalf("guardInfoWorkingSetFrom: %v", err)
	}
	want := []string{"internal/modver/modver.go", "cmd/fak/new.go"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("working set = %v, want %v", got, want)
	}
}
