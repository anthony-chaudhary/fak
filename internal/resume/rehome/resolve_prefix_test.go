package rehome

import (
	"os"
	"path/filepath"
	"testing"
)

// writePrefixTranscript creates <home>/.claude/projects/<proj>/<sid>.jsonl so LocateMatches
// (and the prefix scan) can find it, mirroring the on-disk layout resume reads.
func writePrefixTranscript(t *testing.T, home, sid string) {
	t.Helper()
	dir := filepath.Join(home, ".claude", "projects", "proj")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, sid+".jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
}

func TestLooksLikeFullSessionID(t *testing.T) {
	cases := map[string]bool{
		"":                                     false,
		"aaaaaaaa":                             false, // the 8-char paste-truncation footgun
		"aaaaaaaa-1111-2222":                   false,
		"aaaaaaaa-1111-2222-3333-444444444444": true, // canonical 36-char UUID
		"aaaaaaaa11112222333344444444aaaa":     true, // 32-char hex, no hyphens
	}
	for sid, want := range cases {
		if got := looksLikeFullSessionID(sid); got != want {
			t.Errorf("looksLikeFullSessionID(%q) = %v, want %v", sid, got, want)
		}
	}
}

func TestResolvePrefixCounts(t *testing.T) {
	home := t.TempDir()
	writePrefixTranscript(t, home, "aaaaaaaa-1111-2222-3333-444444444444")
	writePrefixTranscript(t, home, "aaaaaaaa-5555-6666-7777-888888888888")
	writePrefixTranscript(t, home, "bbbbbbbb-1111-2222-3333-444444444444")

	if full, n := resolvePrefix("bbbbbbbb", home); n != 1 || full != "bbbbbbbb-1111-2222-3333-444444444444" {
		t.Errorf("unique prefix: got (%q, %d), want the bbbb id and 1", full, n)
	}
	if full, n := resolvePrefix("aaaaaaaa", home); n != 2 || full != "" {
		t.Errorf("ambiguous prefix: got (%q, %d), want (\"\", 2)", full, n)
	}
	if full, n := resolvePrefix("zzzzzzzz", home); n != 0 || full != "" {
		t.Errorf("no-match prefix: got (%q, %d), want (\"\", 0)", full, n)
	}
}

// TestResolvePartialIDIsNotSilentlyPinned is the #3782 regression: a short id must be
// disambiguated, never silently PIN_FRESH'd; a full-but-absent id keeps landing fresh.
func TestResolvePartialIDIsNotSilentlyPinned(t *testing.T) {
	home := t.TempDir()
	writePrefixTranscript(t, home, "aaaaaaaa-1111-2222-3333-444444444444")
	writePrefixTranscript(t, home, "aaaaaaaa-5555-6666-7777-888888888888")
	writePrefixTranscript(t, home, "bbbbbbbb-1111-2222-3333-444444444444")

	// Partial id matching nothing -> refuse (NOT_FULL_ID), do NOT pin a fresh seat.
	got := Resolve(ResolveInput{SID: "deadbeef", Home: home})
	if got.OK || got.Action != ActionNotFullID {
		t.Errorf("no-match partial: got OK=%v action=%q, want NOT_FULL_ID", got.OK, got.Action)
	}

	// Partial id matching >1 session -> AMBIGUOUS_PREFIX with candidates listed.
	got = Resolve(ResolveInput{SID: "aaaaaaaa", Home: home})
	if got.OK || got.Action != ActionAmbiguousPrefix || len(got.PrefixCandidates) != 2 {
		t.Errorf("ambiguous partial: got OK=%v action=%q cands=%v, want AMBIGUOUS_PREFIX with 2 candidates",
			got.OK, got.Action, got.PrefixCandidates)
	}

	// Partial id uniquely prefix-matching one session -> resolved to the full id and
	// run through the normal ladder, recording what the caller typed.
	got = Resolve(ResolveInput{SID: "bbbbbbbb", Home: home, DryRun: true, OwnerStatus: &OwnerStatus{Available: true}})
	if got.PrefixResolvedFrom != "bbbbbbbb" || got.Session != "bbbbbbbb-1111-2222-3333-444444444444" {
		t.Errorf("unique partial: got resolvedFrom=%q session=%q, want the bbbb id resolved from \"bbbbbbbb\"",
			got.PrefixResolvedFrom, got.Session)
	}
	if got.Action == ActionNotFullID || got.Action == ActionAmbiguousPrefix {
		t.Errorf("unique partial: got refusal action %q, want the resolved session's normal verdict", got.Action)
	}

	// A FULL, genuinely-absent id must still be allowed to land fresh (empty roster
	// here degrades PIN_FRESH to NOT_FOUND) -- it is NOT treated as a typo.
	got = Resolve(ResolveInput{SID: "cccccccc-9999-2222-3333-444444444444", Home: home})
	if got.Action == ActionNotFullID || got.Action == ActionAmbiguousPrefix {
		t.Errorf("full absent id: got %q, want the new-session landing path (not a partial-id refusal)", got.Action)
	}
}
