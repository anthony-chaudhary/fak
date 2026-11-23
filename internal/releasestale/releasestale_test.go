package releasestale

import (
	"context"
	"strings"
	"testing"
)

func TestComputeVerdicts(t *testing.T) {
	th := DefaultThresholds()
	cases := []struct {
		name     string
		facts    Facts
		want     Verdict
		wantOK   bool
		findHave string // substring expected in NextAction
	}{
		{
			name:   "unreachable git is unknown and does not gate",
			facts:  Facts{Reachable: false},
			want:   Unknown,
			wantOK: true,
		},
		{
			name:   "reachable but no tag is unknown (nothing published)",
			facts:  Facts{Reachable: true, LatestTag: "", HeadSHA: "abc"},
			want:   Unknown,
			wantOK: true,
		},
		{
			name:   "tag at HEAD is fresh",
			facts:  Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 0},
			want:   Fresh,
			wantOK: true,
		},
		{
			name:   "small lag within thresholds is fresh",
			facts:  Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 5, DaysBehind: 2},
			want:   Fresh,
			wantOK: true,
		},
		{
			name:     "lag past stale commits is stale and gates",
			facts:    Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 25, DaysBehind: 1},
			want:     Stale,
			wantOK:   false,
			findHave: "cut a release",
		},
		{
			name:     "lag past stale days (few commits) is stale",
			facts:    Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 3, DaysBehind: 20},
			want:     Stale,
			wantOK:   false,
			findHave: "cut a release",
		},
		{
			name:   "lag past very-stale commits is very_stale",
			facts:  Facts{Reachable: true, LatestTag: "v0.34.0", CommitsBehind: 1594, DaysBehind: 60},
			want:   VeryStale,
			wantOK: false,
		},
		{
			name:   "lag past very-stale days alone is very_stale",
			facts:  Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 30, DaysBehind: 50},
			want:   VeryStale,
			wantOK: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := Compute(tc.facts, th, "/ws")
			if got := p.Verdict; got != tc.want.String() {
				t.Fatalf("verdict = %q, want %q", got, tc.want.String())
			}
			if p.OK != tc.wantOK {
				t.Fatalf("OK = %v, want %v", p.OK, tc.wantOK)
			}
			if tc.findHave != "" && !strings.Contains(p.NextAction, tc.findHave) {
				t.Fatalf("NextAction %q does not contain %q", p.NextAction, tc.findHave)
			}
			if p.Schema != Schema {
				t.Fatalf("schema = %q, want %q", p.Schema, Schema)
			}
		})
	}
}

func TestClassifyWallClockAgePromotesOnQuietTrunk(t *testing.T) {
	th := DefaultThresholds() // StaleDays=14, VeryStaleDays=45
	// The ticket's scenario: a tag 13 committer-days behind a frozen HEAD (CommitsBehind
	// and DaysBehind both below their stale bounds), but published 60 wall-clock days ago.
	// The trunk-delta inputs alone read Fresh; the wall-clock publish age must promote it to
	// VeryStale — this is the freeze bug #4024 fixes.
	frozen := Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 13, DaysBehind: 13}

	// Without the age axis (AgeDays=0) the old behavior stands: Fresh.
	if got := classify(frozen, th); got != Fresh {
		t.Fatalf("frozen trunk with no age should still be Fresh (proves the regression), got %v", got)
	}

	// With 60 wall-clock days of publish age, the same frozen trunk is VeryStale and gates.
	aged := frozen
	aged.AgeDays = 60
	p := Compute(aged, th, "/ws")
	if p.Verdict != VeryStale.String() {
		t.Fatalf("60d publish age must classify very_stale, got %q", p.Verdict)
	}
	if p.OK {
		t.Fatalf("very_stale must gate (OK=false)")
	}
	if p.AgeDays != 60 {
		t.Fatalf("Payload.AgeDays = %v, want 60 (surfaced for the control pane)", p.AgeDays)
	}
	if !strings.Contains(p.Reason, "published 60d ago") {
		t.Fatalf("Reason should surface the wall-clock age, got %q", p.Reason)
	}
	if !strings.Contains(Render(p), "60 day(s) ago") {
		t.Fatalf("Render should surface the wall-clock age, got %q", Render(p))
	}

	// 30 days of age (between StaleDays=14 and VeryStaleDays=45) on the same frozen trunk
	// promotes only to Stale, not VeryStale — the age axis honors both bounds.
	midaged := frozen
	midaged.AgeDays = 30
	if got := classify(midaged, th); got != Stale {
		t.Fatalf("30d publish age should be Stale (between the day bounds), got %v", got)
	}

	// Age is immaterial when the tag is AT HEAD: nothing newer to publish stays Fresh.
	atHead := Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 0, AgeDays: 999}
	if got := classify(atHead, th); got != Fresh {
		t.Fatalf("tag at HEAD must stay Fresh regardless of age, got %v", got)
	}
}

func TestComputeVersionAheadOfTagSwitchesAction(t *testing.T) {
	th := DefaultThresholds()
	// VERSION is a higher semver than the tag => a cut landed but was never tagged.
	f := Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 40, DaysBehind: 5, VersionFile: "1.3.0"}
	p := Compute(f, th, "/ws")
	if !p.VersionAheadOfTag {
		t.Fatalf("expected VersionAheadOfTag=true for VERSION 1.3.0 > tag v1.2.3")
	}
	if !strings.Contains(p.NextAction, "untagged") {
		t.Fatalf("expected the 'untagged cut' next-action, got %q", p.NextAction)
	}
	// VERSION equal to tag must NOT read ahead.
	p2 := Compute(Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 40, VersionFile: "1.2.3"}, th, "/ws")
	if p2.VersionAheadOfTag {
		t.Fatalf("VERSION == tag must not be ahead")
	}
	if !strings.Contains(p2.NextAction, "cut a release") {
		t.Fatalf("expected ordinary cut-a-release action, got %q", p2.NextAction)
	}
}

func TestClassifyDisabledThresholdNeverTrips(t *testing.T) {
	// A zero bound disables that rung; with everything disabled, any positive lag stays Fresh.
	th := Thresholds{StaleCommits: 0, StaleDays: 0, VeryStaleCommits: 0, VeryStaleDays: 0}
	if got := classify(Facts{Reachable: true, LatestTag: "v1.0.0", CommitsBehind: 9999, DaysBehind: 9999}, th); got != Fresh {
		t.Fatalf("disabled thresholds should keep any lag Fresh, got %v", got)
	}
}

func TestDaysBetween(t *testing.T) {
	// 13.5 days apart -> 13.5; head before tag (impossible ordering) -> 0; unknown -> 0.
	const day = int64(86400)
	if got := daysBetween(1000, 1000+13*day+day/2); got != 13.5 {
		t.Fatalf("daysBetween = %v, want 13.5", got)
	}
	if got := daysBetween(0, 5000); got != 0 {
		t.Fatalf("unknown tag epoch must yield 0, got %v", got)
	}
	if got := daysBetween(5000, 4000); got != 0 {
		t.Fatalf("head-before-tag must clamp to 0, got %v", got)
	}
}

// fixedClock returns a Clock that always reports epoch, so Gather's now-anchored AgeDays is
// deterministic in the transcript-driven tests.
func fixedClock(epoch int64) Clock { return func() int64 { return epoch } }

// fakeRunner replays a canned git transcript keyed by the joined command line. A missing
// key returns ok=false, exactly as a failed git invocation would.
func fakeRunner(t *testing.T, responses map[string]string) Runner {
	t.Helper()
	return func(_ context.Context, _ string, name string, args ...string) (string, bool) {
		key := name + " " + strings.Join(args, " ")
		out, ok := responses[key]
		return out, ok
	}
}

func TestGatherParsesGitFacts(t *testing.T) {
	// The tag listing is intentionally noisy: a non-semver tag and a stable/ channel tag
	// must be skipped; the newest vX.Y.Z is chosen. HEAD is 10 days after the tag.
	const day = 86400
	const tagEpoch = 1000000
	run := fakeRunner(t, map[string]string{
		"git --no-optional-locks rev-parse HEAD":                      "headsha000000000000",
		"git --no-optional-locks tag --sort=-v:refname --merged HEAD": "nightly-2026\nstable/2026-06\nv0.34.0\nv0.33.0",
		"git --no-optional-locks rev-list -n1 v0.34.0":                "tagsha1111111111111",
		"git --no-optional-locks rev-list --count v0.34.0..HEAD":      "1594",
		"git --no-optional-locks log -1 --format=%ct v0.34.0":         itoa64(tagEpoch),
		"git --no-optional-locks log -1 --format=%ct HEAD":            itoa64(tagEpoch + 10*day),
	})

	// HEAD is 10 days after the tag, but wall-clock now is 40 days after the tag: DaysBehind
	// (the frozen trunk delta) is 10, while AgeDays (the real publish age) is 40.
	f := Gather(context.Background(), run, fixedClock(tagEpoch+40*day), "/repo", "0.34.0")
	if !f.Reachable {
		t.Fatalf("expected Reachable=true")
	}
	if f.LatestTag != "v0.34.0" {
		t.Fatalf("LatestTag = %q, want v0.34.0 (noisy tags must be skipped)", f.LatestTag)
	}
	if f.TagSHA != "tagsha1111111111111" || f.HeadSHA != "headsha000000000000" {
		t.Fatalf("shas wrong: tag=%q head=%q", f.TagSHA, f.HeadSHA)
	}
	if f.CommitsBehind != 1594 {
		t.Fatalf("CommitsBehind = %d, want 1594", f.CommitsBehind)
	}
	if f.DaysBehind != 10 {
		t.Fatalf("DaysBehind = %v, want 10", f.DaysBehind)
	}
	if f.AgeDays != 40 {
		t.Fatalf("AgeDays = %v, want 40 (wall-clock age from the injected clock)", f.AgeDays)
	}
	if f.VersionFile != "0.34.0" {
		t.Fatalf("VersionFile = %q, want 0.34.0", f.VersionFile)
	}
}

func TestGatherUnreadableHeadIsUnreachable(t *testing.T) {
	run := fakeRunner(t, map[string]string{}) // every call fails
	f := Gather(context.Background(), run, fixedClock(5000), "/repo", "0.34.0")
	if f.Reachable {
		t.Fatalf("expected Reachable=false when HEAD is unreadable")
	}
	// And that flows through to an Unknown verdict that does not gate.
	p := Compute(f, DefaultThresholds(), "/repo")
	if p.Verdict != Unknown.String() || !p.OK {
		t.Fatalf("unreachable should be unknown+OK, got verdict=%q ok=%v", p.Verdict, p.OK)
	}
}

func TestGatherNoSemverTag(t *testing.T) {
	run := fakeRunner(t, map[string]string{
		"git --no-optional-locks rev-parse HEAD":                      "headsha",
		"git --no-optional-locks tag --sort=-v:refname --merged HEAD": "nightly-2026\nstable/2026-06",
	})
	f := Gather(context.Background(), run, fixedClock(5000), "/repo", "")
	if !f.Reachable {
		t.Fatalf("HEAD readable so Reachable should be true")
	}
	if f.LatestTag != "" {
		t.Fatalf("no semver tag should yield empty LatestTag, got %q", f.LatestTag)
	}
}

func TestComputePrereleaseNewerThanLatestReadsLatestLag(t *testing.T) {
	th := DefaultThresholds()
	// Newest tag v0.57.0 is a pre-release 194 commits / 8d behind; GitHub Latest is v0.56.0,
	// 3409 commits / 41d behind. The newest-tag axis alone is very stale on commits, so use a
	// tag lag inside the thresholds to prove Latest is what promotes the verdict.
	f := Facts{
		Reachable: true, LatestTag: "v0.57.0", CommitsBehind: 10, DaysBehind: 2, AgeDays: 2,
		LatestRelease: "v0.56.0", ReleaseReachable: true,
		ReleaseCommitsBehind: 3409, ReleaseDaysBehind: 41, ReleaseAgeDays: 41,
	}
	if got := tagLevel(f, th); got != Fresh {
		t.Fatalf("newest-tag axis should read fresh on its own, got %v", got)
	}
	p := Compute(f, th, "/ws")
	if p.Verdict != VeryStale.String() || p.OK {
		t.Fatalf("Latest 3409 commits behind must be very_stale and gate, got verdict=%q ok=%v", p.Verdict, p.OK)
	}
	if p.LatestReleaseCheck != ReleaseMismatch {
		t.Fatalf("LatestReleaseCheck = %q, want %q", p.LatestReleaseCheck, ReleaseMismatch)
	}
	if p.ReleaseCommitsBehind != 3409 || p.ReleaseAgeDays != 41 {
		t.Fatalf("release lag not surfaced: commits=%d age=%v", p.ReleaseCommitsBehind, p.ReleaseAgeDays)
	}
	if !strings.Contains(p.Reason, "v0.56.0") || !strings.Contains(p.NextAction, "promote v0.57.0") {
		t.Fatalf("reason/next action must name Latest and the promotion, got %q / %q", p.Reason, p.NextAction)
	}
	if !strings.Contains(Render(p), "GitHub Latest: v0.56.0 (mismatch)") {
		t.Fatalf("Render should show the Latest mismatch, got %q", Render(p))
	}

	// A fresh Latest that is merely older than a just-cut pre-release is a mismatch, not a gate.
	small := f
	small.ReleaseCommitsBehind, small.ReleaseDaysBehind, small.ReleaseAgeDays = 12, 3, 3
	ps := Compute(small, th, "/ws")
	if ps.Verdict != Fresh.String() || !ps.OK || ps.LatestReleaseCheck != ReleaseMismatch {
		t.Fatalf("small Latest lag: verdict=%q ok=%v check=%q", ps.Verdict, ps.OK, ps.LatestReleaseCheck)
	}
	if !strings.Contains(ps.NextAction, "promote v0.57.0") {
		t.Fatalf("fresh mismatch should still point at promotion, got %q", ps.NextAction)
	}
}

func TestComputeLatestEqualsNewestTag(t *testing.T) {
	th := DefaultThresholds()
	f := Facts{
		Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 5, DaysBehind: 1, AgeDays: 1,
		LatestRelease: "v1.2.3", ReleaseReachable: true,
		ReleaseCommitsBehind: 5, ReleaseDaysBehind: 1, ReleaseAgeDays: 1,
	}
	p := Compute(f, th, "/ws")
	if p.Verdict != Fresh.String() || !p.OK || p.LatestReleaseCheck != ReleaseMatch {
		t.Fatalf("equal anchors: verdict=%q ok=%v check=%q", p.Verdict, p.OK, p.LatestReleaseCheck)
	}
	if strings.Contains(p.NextAction, "promote") {
		t.Fatalf("matching Latest must not ask for a promotion, got %q", p.NextAction)
	}
}

func TestComputeLatestUncheckedOrUnresolvedNeverPromotes(t *testing.T) {
	th := DefaultThresholds()
	base := Facts{Reachable: true, LatestTag: "v1.2.3", CommitsBehind: 5, DaysBehind: 1}
	if p := Compute(base, th, "/ws"); p.LatestReleaseCheck != ReleaseUnchecked || p.Verdict != Fresh.String() {
		t.Fatalf("no Latest supplied: check=%q verdict=%q", p.LatestReleaseCheck, p.Verdict)
	}
	unresolved := base
	unresolved.LatestRelease = "v0.1.0"
	unresolved.ReleaseCommitsBehind = 9999 // must be ignored when not resolved
	if p := Compute(unresolved, th, "/ws"); p.LatestReleaseCheck != ReleaseUnresolved || p.Verdict != Fresh.String() {
		t.Fatalf("unresolved Latest: check=%q verdict=%q", p.LatestReleaseCheck, p.Verdict)
	}
}

func TestGatherWithReleaseMeasuresLatestLag(t *testing.T) {
	const day = 86400
	const relEpoch = 1000000
	const tagEpoch = relEpoch + 33*day
	run := fakeRunner(t, map[string]string{
		"git --no-optional-locks rev-parse HEAD":                      "headsha",
		"git --no-optional-locks tag --sort=-v:refname --merged HEAD": "v0.57.0\nv0.56.0",
		"git --no-optional-locks rev-list -n1 v0.57.0":                "tagsha",
		"git --no-optional-locks rev-list --count v0.57.0..HEAD":      "194",
		"git --no-optional-locks rev-list --count v0.56.0..HEAD":      "3409",
		"git --no-optional-locks log -1 --format=%ct v0.57.0":         itoa64(tagEpoch),
		"git --no-optional-locks log -1 --format=%ct v0.56.0":         itoa64(relEpoch),
		"git --no-optional-locks log -1 --format=%ct HEAD":            itoa64(relEpoch + 40*day),
	})
	f := GatherWithRelease(context.Background(), run, fixedClock(relEpoch+41*day), "/repo", "", " v0.56.0 ")
	if f.LatestRelease != "v0.56.0" || !f.ReleaseReachable {
		t.Fatalf("Latest not resolved: %+v", f)
	}
	if f.ReleaseCommitsBehind != 3409 || f.ReleaseDaysBehind != 40 || f.ReleaseAgeDays != 41 {
		t.Fatalf("release lag = %d / %v / %v, want 3409 / 40 / 41", f.ReleaseCommitsBehind, f.ReleaseDaysBehind, f.ReleaseAgeDays)
	}
	if f.CommitsBehind != 194 || f.AgeDays != 8 {
		t.Fatalf("newest-tag lag = %d / age %v, want 194 / 8", f.CommitsBehind, f.AgeDays)
	}

	// A Latest tag absent from the clone stays unresolved.
	g := GatherWithRelease(context.Background(), run, fixedClock(relEpoch+41*day), "/repo", "", "v9.9.9")
	if g.ReleaseReachable || g.LatestRelease != "v9.9.9" {
		t.Fatalf("unknown Latest should be unresolved, got %+v", g)
	}
}

func TestGatherReadsLatestFromEnv(t *testing.T) {
	t.Setenv(LatestReleaseEnv, "v1.0.0")
	run := fakeRunner(t, map[string]string{
		"git --no-optional-locks rev-parse HEAD":                      "headsha",
		"git --no-optional-locks tag --sort=-v:refname --merged HEAD": "v1.1.0\nv1.0.0",
		"git --no-optional-locks rev-list --count v1.0.0..HEAD":       "7",
	})
	f := Gather(context.Background(), run, fixedClock(5000), "/repo", "")
	if f.LatestRelease != "v1.0.0" || f.ReleaseCommitsBehind != 7 {
		t.Fatalf("Gather must read %s, got %+v", LatestReleaseEnv, f)
	}
}

func itoa64(n int64) string {
	// local helper to avoid importing strconv in the test for one call
	neg := n < 0
	if neg {
		n = -n
	}
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
