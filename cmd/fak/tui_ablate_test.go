package main

import (
	"bytes"
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

// testAblateReport is a deterministic, hand-built ablation report exercising every render
// axis: a baseline arm (no badges), a witnessed/lossless vdso arm whose win is avoided
// CALLS (a negative Δtokens, zero token-equiv, empty bar), a recoverable/simulated fak
// compaction saving, and a lossless provider prompt-cache saving (the max, a full bar). No
// sweep runs — so the tests never spawn a subprocess.
func testAblateReport() *ablate.Report {
	const wh = "wh_test_0d1e2f3a"
	arm := func(id string, in, out, vdso int64, ms gateway.MechanismSavings, effects ...ablate.CacheEffect) ablate.AblationRun {
		return ablate.AblationRun{
			ArmID:            id,
			Features:         map[string]string{},
			WorkloadHash:     wh,
			Arm:              metrics.Arm{InTokens: in, OutTokens: out, VDSOHits: vdso},
			MechanismSavings: ms,
			CacheEffects:     effects,
		}
	}
	eff := func(feature, fidelity, evidence, status string) ablate.CacheEffect {
		return ablate.CacheEffect{Feature: feature, Component: feature, Owner: "fak", Fidelity: fidelity, Evidence: evidence, Status: status}
	}
	return &ablate.Report{
		Provenance:   metrics.Provenance{SliceID: "tau2-test", EngineModel: "mock-offline", WorkloadHash: wh},
		WorkloadHash: wh,
		Baseline:     "all-off",
		Runs: []ablate.AblationRun{
			arm("all-off", 10000, 2000, 0, gateway.MechanismSavings{}),
			arm("vdso", 9500, 2000, 3, gateway.MechanismSavings{FakVDSOAvoidedCalls: 3},
				eff("vdso", "lossless", "witnessed", "active")),
			arm("compressor", 10000, 2000, 0, gateway.MechanismSavings{FakCompactionShedTokens: 2500},
				eff("compressor", "recoverable", "simulated", "active")),
			arm("provcache", 10000, 2000, 0,
				gateway.MechanismSavings{ProviderPromptCacheReadTokenEquiv: 7500, ProviderPromptCacheWritePremiumTokenEquiv: -500},
				eff("provcache", "lossless", "simulated", "active")),
		},
	}
}

func findAblateRow(v ablateView, concept string) (ablateViewRow, bool) {
	for _, r := range v.Rows {
		if r.Concept == concept {
			return r, true
		}
	}
	return ablateViewRow{}, false
}

func TestBuildAblateViewProjectsConceptsAndBaseline(t *testing.T) {
	v := buildAblateView(testAblateReport())
	if len(v.Rows) != 4 {
		t.Fatalf("rows=%d want 4", len(v.Rows))
	}
	if v.Baseline != "all-off" || v.SliceID != "tau2-test" || v.EngineModel != "mock-offline" {
		t.Fatalf("header fields wrong: %+v", v)
	}
	// Baseline carries no per-concept badges.
	if base, ok := findAblateRow(v, "all-off"); !ok || !base.IsBaseline || base.Fidelity != "" || base.Status != "" {
		t.Fatalf("baseline row = %+v (want IsBaseline, empty badges)", base)
	}
	// vdso: witnessed/lossless, avoided-call win shows as a NEGATIVE Δtokens with ZERO token-equiv.
	vdso, ok := findAblateRow(v, "vdso")
	if !ok || vdso.Fidelity != "lossless" || vdso.Evidence != "witnessed" || vdso.Status != "active" {
		t.Fatalf("vdso badges = %+v", vdso)
	}
	if vdso.DeltaTokens != -500 || vdso.TotalTokEq != 0 {
		t.Fatalf("vdso delta=%d tokeq=%.0f want -500 / 0", vdso.DeltaTokens, vdso.TotalTokEq)
	}
	// compressor: recoverable/simulated fak saving.
	comp, _ := findAblateRow(v, "compressor")
	if comp.Fidelity != "recoverable" || comp.Evidence != "simulated" || comp.TotalTokEq != 2500 {
		t.Fatalf("compressor row = %+v", comp)
	}
	// provcache is the max saving (read rebate 7500 minus write premium 500 = 7000).
	prov, _ := findAblateRow(v, "provcache")
	if prov.ProviderTokEq != 7000 || prov.TotalTokEq != 7000 {
		t.Fatalf("provcache provider=%.0f total=%.0f want 7000/7000", prov.ProviderTokEq, prov.TotalTokEq)
	}
	if v.MaxTokEq != 7000 {
		t.Fatalf("max token-equiv=%.0f want 7000", v.MaxTokEq)
	}
}

var sgrRe = regexp.MustCompile("\x1b\\[[0-9;]*m")

func stripSGR(s string) string { return sgrRe.ReplaceAllString(s, "") }

// assertNoOverflow checks every rendered line fits the width budget in terminal CELLS
// (after stripping any color escapes), the core "never wraps" contract.
func assertNoOverflow(t *testing.T, out string, width int) {
	t.Helper()
	for _, line := range strings.Split(strings.TrimRight(out, "\n"), "\n") {
		if w := dispWidthTUI(stripSGR(line)); w > width {
			t.Fatalf("line exceeds width %d (got %d cells): %q", width, w, stripSGR(line))
		}
	}
}

func TestRenderAblateViewNoColor(t *testing.T) {
	v := buildAblateView(testAblateReport())
	out := renderAblateView(v, 100, false)
	if strings.Contains(out, "\x1b") {
		t.Fatalf("NO_COLOR render leaked an ANSI escape:\n%s", out)
	}
	for _, want := range []string{
		"caching-concept ablation", "engine mock-offline", "workload#wh_test_",
		"concept", "fidelity", "evidence", "Δtokens", "status",
		"all-off", "vdso", "lossless", "witnessed",
		"compressor", "recoverable", "simulated", "provcache",
		"legend", "provider-cache save",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
	assertNoOverflow(t, out, 100)
}

func TestRenderAblateViewColorPaintsAndFits(t *testing.T) {
	v := buildAblateView(testAblateReport())
	out := renderAblateView(v, 120, true)
	if !strings.Contains(out, "\x1b[") {
		t.Fatalf("color render emitted no SGR escape:\n%s", out)
	}
	// The witnessed vdso badge and the active status carry their semantic colors.
	if !strings.Contains(out, tuiSGRGreenBold) || !strings.Contains(out, tuiSGRGreen) {
		t.Fatalf("expected witnessed/active green SGR in colored render")
	}
	// Visible width (escapes stripped) still fits the budget — color never desyncs columns.
	assertNoOverflow(t, out, 120)
}

func TestRenderAblateViewNarrowNoOverflow(t *testing.T) {
	v := buildAblateView(testAblateReport())
	for _, w := range []int{80, 90, 72, 60} {
		assertNoOverflow(t, renderAblateView(v, w, false), w)
		assertNoOverflow(t, renderAblateView(v, w, true), w)
	}
}

func TestRenderAblateEmptyBarHint(t *testing.T) {
	// A report with no token-equiv savings (a vdso-only style sweep) renders the guidance
	// hint instead of a silently blank chart.
	rep := &ablate.Report{
		Provenance:   metrics.Provenance{SliceID: "s", EngineModel: "mock-offline", WorkloadHash: "wh"},
		WorkloadHash: "wh", Baseline: "all-off",
		Runs: []ablate.AblationRun{
			{ArmID: "all-off", WorkloadHash: "wh", Arm: metrics.Arm{InTokens: 100}},
			{ArmID: "vdso", WorkloadHash: "wh", Arm: metrics.Arm{InTokens: 100}, MechanismSavings: gateway.MechanismSavings{FakVDSOAvoidedCalls: 2},
				CacheEffects: []ablate.CacheEffect{{Feature: "vdso", Component: "vdso", Fidelity: "lossless", Evidence: "witnessed", Status: "active"}}},
		},
	}
	out := renderAblateView(buildAblateView(rep), 120, false)
	if !strings.Contains(out, "no token-equiv savings in this sweep") {
		t.Fatalf("expected empty-bar hint, got:\n%s", out)
	}
}

func TestAblateViewJSONRoundTrips(t *testing.T) {
	v := buildAblateView(testAblateReport())
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got ablateView
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Schema != tuiAblateSchema || got.Baseline != "all-off" || len(got.Rows) != 4 {
		t.Fatalf("round-trip lost fields: %+v", got)
	}
	prov, ok := findAblateRow(got, "provcache")
	if !ok || prov.TotalTokEq != 7000 {
		t.Fatalf("round-trip provcache = %+v", prov)
	}
}

func TestAblatePaneIsRegistered(t *testing.T) {
	p, ok := tuiplugin.Lookup("ablate")
	if !ok {
		t.Fatal("ablate pane not registered")
	}
	if p.Run == nil || p.Schema != tuiAblateSchema || !p.BuiltIn {
		t.Fatalf("ablate pane registration = %+v", p)
	}
}

// TestFollowAblateLoopStopsOnContext drives the live redraw loop with a short-timeout
// context and a tiny interval, so it paints a few frames then returns cleanly — a
// deterministic smoke that the loop honors cancellation and never hangs.
func TestFollowAblateLoopStopsOnContext(t *testing.T) {
	load := func() (*ablate.Report, error) { return testAblateReport(), nil }
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	var out, errb bytes.Buffer

	done := make(chan int, 1)
	go func() { done <- followAblateLoop(ctx, &out, &errb, load, 100, false, 5*time.Millisecond) }()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("followAblateLoop code=%d want 0", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("followAblateLoop did not return after context cancel (hang)")
	}
	if !strings.Contains(out.String(), "caching-concept ablation") {
		t.Fatalf("follow loop painted no frame:\n%s", out.String())
	}
}
