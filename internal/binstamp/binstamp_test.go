package binstamp

import (
	"runtime/debug"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

func TestStampFromBuildInfo(t *testing.T) {
	if got := stampFrom(nil); got.HasVCS || got.Revision != "" {
		t.Fatalf("nil BuildInfo: got %+v, want zero stamp", got)
	}
	bi := &debug.BuildInfo{Settings: []debug.BuildSetting{
		{Key: "vcs.revision", Value: "abcdef1234567890"},
		{Key: "vcs.modified", Value: "true"},
	}}
	got := stampFrom(bi)
	if !got.HasVCS || got.Revision != "abcdef1234567890" || !got.Dirty {
		t.Fatalf("got %+v, want rev set + dirty + HasVCS", got)
	}
}

func TestCompareFreshnessRules(t *testing.T) {
	const head = "abcdef1234567890abcdef1234567890abcdef12"

	cases := []struct {
		name    string
		running Stamp
		head    string
		want    Freshness
	}{
		{"equal full rev => fresh",
			Stamp{Revision: head, HasVCS: true}, head, Fresh},
		{"short prefix matches => fresh",
			Stamp{Revision: "abcdef1234567", HasVCS: true}, head, Fresh},
		{"different clean rev => stale",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true}, head, Stale},
		{"no embedded rev => unknown",
			Stamp{HasVCS: false}, head, Unknown},
		{"empty head => unknown",
			Stamp{Revision: head, HasVCS: true}, "", Unknown},
		{"dirty build => unknown (never restart)",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true, Dirty: true}, head, Unknown},
		{"too-short prefix doesn't falsely match",
			Stamp{Revision: "abcd", HasVCS: true}, head, Stale},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Compare(c.running, c.head); got != c.want {
				t.Fatalf("Compare = %v, want %v", got, c.want)
			}
		})
	}
}

func TestExplainDistinguishesUnknownCauses(t *testing.T) {
	const head = "abcdef1234567890abcdef1234567890abcdef12"

	cases := []struct {
		name      string
		running   Stamp
		head      string
		wantFresh Freshness
		wantCause Cause
	}{
		{"fresh matches",
			Stamp{Revision: head, HasVCS: true}, head, Fresh, CauseMatched},
		{"clean different rev diverged",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true}, head, Stale, CauseDiverged},
		{"no stamp is unstamped, not a silent unknown",
			Stamp{HasVCS: false}, head, Unknown, CauseUnstamped},
		{"unstamped dominates even with no head",
			Stamp{HasVCS: false}, "", Unknown, CauseUnstamped},
		{"stamped but no head",
			Stamp{Revision: head, HasVCS: true}, "", Unknown, CauseNoHead},
		{"dirty build is its own cause",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true, Dirty: true}, head, Unknown, CauseDirty},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotFresh, gotCause := Explain(c.running, c.head)
			if gotFresh != c.wantFresh {
				t.Fatalf("Explain freshness = %v, want %v", gotFresh, c.wantFresh)
			}
			if gotCause != c.wantCause {
				t.Fatalf("Explain cause = %v, want %v", gotCause, c.wantCause)
			}
			// The invariant that lets Compare delegate: Explain's verdict must equal Compare's.
			if got := Compare(c.running, c.head); got != gotFresh {
				t.Fatalf("Compare = %v but Explain freshness = %v — they must never drift", got, gotFresh)
			}
		})
	}
}

func TestRevisionsMatch(t *testing.T) {
	if !revisionsMatch("ABCDEF1234567", "abcdef1234567890") {
		t.Fatal("case-insensitive prefix should match")
	}
	if revisionsMatch("abcde", "abcdef1234567890") {
		t.Fatal("a <7 char short rev must not match (too weak)")
	}
	if revisionsMatch("1234567", "abcdef1234567890") {
		t.Fatal("non-prefix must not match")
	}
}

func TestSelfUsesInjectedCommitWhenGoVCSStampMissing(t *testing.T) {
	old := appversion.BuildCommit
	appversion.BuildCommit = "0123456789abcdef0123456789abcdef01234567"
	t.Cleanup(func() { appversion.BuildCommit = old })

	// Self reads this test binary's normal build info. Go test binaries generally carry no
	// vcs.revision here, so the injected installer provenance must remain observable.
	got := Self()
	if !got.HasVCS || got.Revision != appversion.BuildCommit || got.Dirty {
		t.Fatalf("Self() = %+v, want clean injected commit %s", got, appversion.BuildCommit)
	}
}
