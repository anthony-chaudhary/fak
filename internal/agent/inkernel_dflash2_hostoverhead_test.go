package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// hostOverheadGapFieldSet and hostOverheadGapReasonSet are the closed receipt-v3
// gap vocabulary: field names spelled exactly as the receipt JSON keys, and the
// three machine-readable reasons. Gap assertions below match this closed set,
// never free text.
var (
	hostOverheadGapFieldSet = map[string]bool{
		"host_phase_us":                true,
		"kernel_launch_count":          true,
		"inter_kernel_gap_sum_us":      true,
		"h2d_copy_bytes":               true,
		"sync_roundtrips":              true,
		"graph_replay_engaged_tokens":  true,
		"graph_replay_degraded_tokens": true,
	}
	hostOverheadGapReasonSet = map[string]bool{
		"disabled":    true,
		"unobserved":  true,
		"unsupported": true,
	}
)

// dflash2HostOverheadRun drives one scripted DFlash2 generation whose ctx carries
// the per-request decode-host-work accumulator, returning the accumulator beside
// the driver result.
func dflash2HostOverheadRun(t *testing.T, r *dflash2ScriptRequest, maxNew int,
	preview func([]int) (dflash2StopPreview, error)) (*nativeperf.DecodeHostWorkAccumulator, inKernelGenerateResult, error) {
	t.Helper()
	p := dflash2ScriptPlanner(t, r)
	ctx, acc := nativeperf.WithDecodeHostWork(context.Background())
	res, err := p.generateDFlash2(ctx, []int{0}, maxNew, preview, func(dflash2StopPreview) error { return nil })
	return acc, res, err
}

// dflash2HostOverheadPreview builds the standard token-ID stop preview with no
// string stops and a constant piece decoder.
func dflash2HostOverheadPreview(stops map[int]bool) func([]int) (dflash2StopPreview, error) {
	return func(ids []int) (dflash2StopPreview, error) {
		return previewDFlash2Stop(ids, stops, newIncrementalStopScanner(0, nil), func([]int) (string, error) { return "a", nil })
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2HostOverheadRecordsDecodeWindow(t *testing.T) {
	for _, tc := range []struct {
		name        string
		rounds      [][]int
		maxNew      int
		stopIDs     map[int]bool
		wantGen     int
		wantStopped bool
	}{
		{"stop-terminated", [][]int{{1, 3}}, 8, map[int]bool{3: true}, 1, true},
		{"maxnew-terminated", [][]int{{1, 2}, {4}}, 3, nil, 3, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &dflash2ScriptRequest{rounds: tc.rounds}
			acc, res, err := dflash2HostOverheadRun(t, r, tc.maxNew, dflash2HostOverheadPreview(tc.stopIDs))
			if err != nil {
				t.Fatalf("generation failed: %v", err)
			}
			if res.gen != tc.wantGen || res.stopped != tc.wantStopped {
				t.Fatalf("result gen=%d stopped=%v want gen=%d stopped=%v", res.gen, res.stopped, tc.wantGen, tc.wantStopped)
			}
			committed := 0
			for _, c := range r.committed {
				committed += len(c)
			}
			if committed != tc.wantGen {
				t.Fatalf("committed tokens=%d want %d", committed, tc.wantGen)
			}
			totals, ok := acc.Totals()
			if !ok {
				t.Fatal("successful DFlash2 request left no claimable decode window")
			}
			if totals.Windows < 1 {
				t.Fatalf("windows=%d want >=1", totals.Windows)
			}
			if totals.Steps != committed {
				t.Fatalf("steps=%d want committed tokens=%d", totals.Steps, committed)
			}
		})
	}
}

func TestDFlash2HostOverheadFailureRecordsNoWindow(t *testing.T) {
	sentinel := errors.New("dflash2 host-overhead failure sentinel")
	// previewCalls counts preview invocations across the sequential subtests so
	// the mid-loop case commits one full round before failing.
	var previewCalls int
	for _, tc := range []struct {
		name   string
		rounds [][]int
		// setup arms the scripted fault; preview replaces the standard preview when non-nil.
		setup   func(r *dflash2ScriptRequest)
		preview func([]int) (dflash2StopPreview, error)
	}{
		{name: "prepare-fail", rounds: [][]int{{1, 2}},
			setup: func(r *dflash2ScriptRequest) { r.prepareErr = sentinel }},
		{name: "commit-fail", rounds: [][]int{{1, 2}, {3, 4}},
			setup: func(r *dflash2ScriptRequest) { r.commitErr = sentinel }},
		{name: "preview-midloop-fail", rounds: [][]int{{1, 2}, {3, 4}},
			preview: func(ids []int) (dflash2StopPreview, error) {
				previewCalls++
				if previewCalls > 1 {
					return dflash2StopPreview{}, sentinel
				}
				return previewDFlash2Stop(ids, nil, newIncrementalStopScanner(0, nil), func([]int) (string, error) { return "a", nil })
			}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &dflash2ScriptRequest{rounds: tc.rounds}
			if tc.setup != nil {
				tc.setup(r)
			}
			preview := tc.preview
			if preview == nil {
				preview = dflash2HostOverheadPreview(nil)
			}
			acc, _, err := dflash2HostOverheadRun(t, r, 4, preview)
			if !errors.Is(err, sentinel) {
				t.Fatalf("failure not surfaced: %v", err)
			}
			if _, ok := acc.Totals(); ok {
				t.Fatal("abandoned or incomplete span recorded a decode window")
			}
		})
	}
}

func TestDFlash2HostOverheadClaimGapsAndOnceOnly(t *testing.T) {
	r := &dflash2ScriptRequest{rounds: [][]int{{1, 2}}}
	acc, res, err := dflash2HostOverheadRun(t, r, 2, dflash2HostOverheadPreview(nil))
	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}
	if res.gen != 2 {
		t.Fatalf("gen=%d want 2", res.gen)
	}
	var rep nativeperf.Repetition
	if !acc.ClaimHostOverhead(&rep) {
		t.Fatal("claim refused on a request with a recorded decode window")
	}
	// FAK_H2D_CENSUS is unset in the test environment: absent bytes read as
	// disabled, never unobserved, and never a fabricated zero.
	if rep.H2DCopyBytes != nil {
		t.Fatalf("h2d_copy_bytes present with census disabled: %d", *rep.H2DCopyBytes)
	}
	requireExactlyGap(t, rep.HostOverheadGaps, "h2d_copy_bytes=disabled")
	// The fixture backend is not capture-once capable: the engagement witness
	// reads as unsupported on both fields.
	if rep.GraphReplayEngagedTokens != nil || rep.GraphReplayDegradedTokens != nil {
		t.Fatalf("graph witness present on a non-capable backend: engaged=%v degraded=%v",
			rep.GraphReplayEngagedTokens, rep.GraphReplayDegradedTokens)
	}
	requireExactlyGap(t, rep.HostOverheadGaps, "graph_replay_engaged_tokens=unsupported")
	requireExactlyGap(t, rep.HostOverheadGaps, "graph_replay_degraded_tokens=unsupported")
	// Every gap entry matches the closed "<field>=<reason>" grammar.
	for _, g := range rep.HostOverheadGaps {
		field, reason, ok := strings.Cut(g, "=")
		if !ok || !hostOverheadGapFieldSet[field] || !hostOverheadGapReasonSet[reason] {
			t.Fatalf("gap %q escapes the closed field/reason vocabulary", g)
		}
	}
	// Once-only: a second claim is refused and alters neither repetition.
	first := rep
	var second nativeperf.Repetition
	if acc.ClaimHostOverhead(&second) {
		t.Fatal("second claim admitted on the same accumulator")
	}
	if !reflect.DeepEqual(second, nativeperf.Repetition{}) {
		t.Fatalf("refused claim wrote %+v", second)
	}
	if !reflect.DeepEqual(rep, first) {
		t.Fatal("second claim altered the first repetition")
	}
}

// requireExactlyGap asserts that gaps contains exactly one entry equal to want.
func requireExactlyGap(t *testing.T, gaps []string, want string) {
	t.Helper()
	n := 0
	for _, g := range gaps {
		if g == want {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("gap %q appears %d times in %v", want, n, gaps)
	}
}
