package balance

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// TestFoldResumeBudget pins the tally: each resume state lands in its bucket, an empty
// slice folds to a MEASURED all-zero rollup (a fleet with nothing stranded is a real
// fact, not "unmeasured"), and settled rows are counted but never feed the alarm.
func TestFoldResumeBudget(t *testing.T) {
	got := FoldResumeBudget([]resume.ResumeState{
		resume.ResumeTook, resume.ResumeTook, resume.ResumeReStranded,
		resume.ResumeGaveUp, resume.ResumeLaunched, resume.ResumePending, resume.ResumeSettled,
	})
	want := ResumeBudget{Took: 2, ReStranded: 1, GaveUp: 1, Launched: 1, Pending: 1, Settled: 1, Measured: true}
	if got != want {
		t.Errorf("fold: got %+v, want %+v", got, want)
	}

	empty := FoldResumeBudget(nil)
	if !empty.Measured || empty.Took != 0 || empty.ReStranded != 0 {
		t.Errorf("empty fold must be measured all-zero, got %+v", empty)
	}
}

// TestReStrandingOutpacesCompletion pins the RED test: strictly-more re-stranded than
// took fires; a tie does not; and a never-measured (or empty) rollup is never red — the
// alarm needs witnessed slippage, never absence of data.
func TestReStrandingOutpacesCompletion(t *testing.T) {
	cases := []struct {
		name string
		b    ResumeBudget
		want bool
	}{
		{"more re-stranded than took is red", ResumeBudget{Took: 1, ReStranded: 2, Measured: true}, true},
		{"a tie is not red", ResumeBudget{Took: 2, ReStranded: 2, Measured: true}, false},
		{"more took than re-stranded is not red", ResumeBudget{Took: 5, ReStranded: 1, Measured: true}, false},
		{"empty measured fleet is not red", ResumeBudget{Measured: true}, false},
		{"unmeasured zero is not red", ResumeBudget{}, false},
	}
	for _, tc := range cases {
		if got := tc.b.ReStrandingOutpacesCompletion(); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestStatusPrecedence pins the closed headline vocabulary and its ordering — most
// alarming first. The load-bearing case is that a RED resume budget outranks a mix lean:
// a recovery budget underwater is a hard alarm, a lopsided mix is only a soft nudge.
func TestStatusPrecedence(t *testing.T) {
	leaning := &superloop.WorkMix{Gardening: 4, Throughput: 1, TargetThroughputPct: 50, Favor: superloop.WorkThroughput}
	onTarget := &superloop.WorkMix{Gardening: 2, Throughput: 2, TargetThroughputPct: 50}
	red := ResumeBudget{Took: 1, ReStranded: 3, Measured: true}
	healthy := ResumeBudget{Took: 3, ReStranded: 1, Measured: true}

	cases := []struct {
		name string
		ev   Evidence
		want string
	}{
		{"neither measured is no data", Evidence{}, "no data"},
		{"red resume outranks a mix lean", Evidence{Resume: red, Mix: leaning}, "red"},
		{"red fires even with no mix", Evidence{Resume: red}, "red"},
		{"healthy resume + off-target mix leans", Evidence{Resume: healthy, Mix: leaning}, "leaning"},
		{"mix-only off-target leans", Evidence{Mix: leaning}, "leaning"},
		{"healthy resume + on-target mix is ok", Evidence{Resume: healthy, Mix: onTarget}, "ok"},
		{"healthy resume, no mix is ok", Evidence{Resume: healthy}, "ok"},
		{"mix-only on-target is ok", Evidence{Mix: onTarget}, "ok"},
	}
	for _, tc := range cases {
		if got := tc.ev.Status(); got != tc.want {
			t.Errorf("%s: Status = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// TestSharePct pins the throughput-share readout, including the no-ratio guard: with no
// gardening/throughput members there is no ratio to take, so it reads "n/a" rather than
// dividing by zero — the same guard superloop's own favor decision uses.
func TestSharePct(t *testing.T) {
	cases := []struct {
		g, tp int
		want  string
	}{
		{2, 2, "50%"},
		{4, 1, "20%"},
		{1, 3, "75%"},
		{0, 0, "n/a"},
		{3, 0, "0%"},
	}
	for _, tc := range cases {
		if got := sharePct(tc.g, tc.tp); got != tc.want {
			t.Errorf("sharePct(%d,%d) = %q, want %q", tc.g, tc.tp, got, tc.want)
		}
	}
}

// TestRenderFixtures is the fixture-driven golden test: each Evidence renders EXACTLY its
// pinned rows. It covers every degradation combination — both halves measured (healthy /
// red / leaning), each half measured alone, and neither — so a change to the surface text
// or the graceful-degradation wording is a visible diff, not a silent drift.
func TestRenderFixtures(t *testing.T) {
	cases := []struct {
		name string
		ev   Evidence
		want []string
	}{
		{
			name: "both measured, balanced and keeping up",
			ev: Evidence{
				Resume: ResumeBudget{Took: 5, ReStranded: 2, GaveUp: 1, Launched: 3, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 2, Throughput: 2, Neutral: 1, TargetThroughputPct: 50},
			},
			want: []string{
				"night balance — ok",
				"  resume   took 5  re-stranded 2  gave-up 1  ·  launched 3  pending 0   ✓ recovery keeping up (2≤5)",
				"  work     gardening 2  throughput 2  neutral 1  ·  target 50%  mix 50%   → on target",
			},
		},
		{
			name: "red resume outpaces completion; mix lean is subordinate",
			ev: Evidence{
				Resume: ResumeBudget{Took: 1, ReStranded: 4, GaveUp: 2, Pending: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 4, Throughput: 1, TargetThroughputPct: 50, Favor: superloop.WorkThroughput},
			},
			want: []string{
				"night balance — red",
				"  resume   took 1  re-stranded 4  gave-up 2  ·  launched 0  pending 1   ⚠ re-stranding outpaces completion (4>1)",
				"  work     gardening 4  throughput 1  neutral 0  ·  target 50%  mix 20%   → lean throughput",
			},
		},
		{
			name: "healthy resume, mix leaning gardening",
			ev: Evidence{
				Resume: ResumeBudget{Took: 3, ReStranded: 1, Measured: true},
				Mix:    &superloop.WorkMix{Gardening: 1, Throughput: 3, TargetThroughputPct: 50, Favor: superloop.WorkGardening},
			},
			want: []string{
				"night balance — leaning",
				"  resume   took 3  re-stranded 1  gave-up 0  ·  launched 0  pending 0   ✓ recovery keeping up (1≤3)",
				"  work     gardening 1  throughput 3  neutral 0  ·  target 50%  mix 75%   → lean gardening",
			},
		},
		{
			name: "resume unmeasured, mix only (neutral-only worklist, share n/a)",
			ev: Evidence{
				Mix: &superloop.WorkMix{Neutral: 3, TargetThroughputPct: 50},
			},
			want: []string{
				"night balance — ok",
				"  resume   not measured — no resume ledger read this render",
				"  work     gardening 0  throughput 0  neutral 3  ·  target 50%  mix n/a   → on target",
			},
		},
		{
			name: "mix unmeasured, resume only (settled rows shown)",
			ev: Evidence{
				Resume: ResumeBudget{Took: 4, Launched: 1, Settled: 2, Measured: true},
			},
			want: []string{
				"night balance — ok",
				"  resume   took 4  re-stranded 0  gave-up 0  ·  launched 1  pending 0  settled 2   ✓ recovery keeping up (0≤4)",
				"  work     not measured — no superloop walk this render",
			},
		},
		{
			name: "neither measured — honest no-data surface",
			ev:   Evidence{},
			want: []string{
				"night balance — no data",
				"  resume   not measured — no resume ledger read this render",
				"  work     not measured — no superloop walk this render",
			},
		},
		{
			name: "red fires with no mix at all",
			ev: Evidence{
				Resume: ResumeBudget{ReStranded: 1, Measured: true},
			},
			want: []string{
				"night balance — red",
				"  resume   took 0  re-stranded 1  gave-up 0  ·  launched 0  pending 0   ⚠ re-stranding outpaces completion (1>0)",
				"  work     not measured — no superloop walk this render",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Render(tc.ev)
			if len(got) != len(tc.want) {
				t.Fatalf("row count = %d, want %d\n got: %s", len(got), len(tc.want), strings.Join(got, "\n      "))
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("row %d:\n got %q\nwant %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}
