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
	run := fakeRunner(t, map[string]string{
		"git --no-optional-locks rev-parse HEAD":                      "headsha000000000000",
		"git --no-optional-locks tag --sort=-v:refname --merged HEAD": "nightly-2026\nstable/2026-06\nv0.34.0\nv0.33.0",
		"git --no-optional-locks rev-list -n1 v0.34.0":                "tagsha1111111111111",
		"git --no-optional-locks rev-list --count v0.34.0..HEAD":      "1594",
		"git --no-optional-locks log -1 --format=%ct v0.34.0":         "1000000",
		"git --no-optional-locks log -1 --format=%ct HEAD":            itoa64(1000000 + 10*day),
	})

	f := Gather(context.Background(), run, "/repo", "0.34.0")
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
	if f.VersionFile != "0.34.0" {
		t.Fatalf("VersionFile = %q, want 0.34.0", f.VersionFile)
	}
}

func TestGatherUnreadableHeadIsUnreachable(t *testing.T) {
	run := fakeRunner(t, map[string]string{}) // every call fails
	f := Gather(context.Background(), run, "/repo", "0.34.0")
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
	f := Gather(context.Background(), run, "/repo", "")
	if !f.Reachable {
		t.Fatalf("HEAD readable so Reachable should be true")
	}
	if f.LatestTag != "" {
		t.Fatalf("no semver tag should yield empty LatestTag, got %q", f.LatestTag)
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
