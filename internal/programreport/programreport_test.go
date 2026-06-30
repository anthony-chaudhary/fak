package programreport

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/worktype"
)

func kernelAdvancing() Signal {
	return Signal{Class: worktype.KernelOptimization, Label: "kernel-optimization", Frontier: "perf work landing", Metric: 3, Direction: "advancing", Activity: 3, Window: "7d", OK: true}
}

func cacheHolding() Signal {
	return Signal{Class: worktype.CacheOptimization, Label: "cache-optimization", Frontier: "realized reuse 0.600 -> 0.620", Metric: 0.62, Direction: "holding", OK: true, Note: "marginal-over-tuned-warm-KV"}
}

// TestInterpretProgramsTally folds clean signals and pins the tally + verdict.
func TestInterpretProgramsTally(t *testing.T) {
	p := InterpretPrograms([]Signal{kernelAdvancing(), cacheHolding()})
	if p.Err != "" || !p.OK {
		t.Fatalf("two measured programs must fold cleanly, got err=%q ok=%v", p.Err, p.OK)
	}
	if p.Tracked != 2 || p.Measured != 2 {
		t.Fatalf("tracked/measured = %d/%d, want 2/2", p.Tracked, p.Measured)
	}
	if p.Advancing != 1 || p.Regressed != 0 {
		t.Fatalf("advancing/regressed = %d/%d, want 1/0", p.Advancing, p.Regressed)
	}
}

// TestInterpretProgramsAllUnmeasuredGates proves that when every program's signal
// fails to read, the dimension errors (the unmeasured gate) — never a silent zero.
func TestInterpretProgramsAllUnmeasuredGates(t *testing.T) {
	p := InterpretPrograms([]Signal{
		{Class: worktype.KernelOptimization, Label: "kernel-optimization", Err: "git log failed"},
		{Class: worktype.CacheOptimization, Label: "cache-optimization", Err: "ledger unreadable"},
	})
	if p.Err == "" || p.OK {
		t.Fatalf("all-unmeasured must error the dimension, got err=%q ok=%v", p.Err, p.OK)
	}
	r := Fold(p, FoldOpts{Date: "2026-06-29"})
	if r.OK || r.Finding != "programs_unmeasured" {
		t.Fatalf("an unmeasured dimension must be ACTION/unmeasured, got %+v", r)
	}
	if code, msg := CheckGate(r); code != 1 || !strings.Contains(msg, "INCOMPLETE") {
		t.Fatalf("unmeasured must gate 1, got %d %q", code, msg)
	}
}

// TestInterpretProgramsPartialIsAdvisory proves one readable + one unreadable program
// is MEASURED (no gate) with a non-gating partial note.
func TestInterpretProgramsPartialIsAdvisory(t *testing.T) {
	p := InterpretPrograms([]Signal{
		kernelAdvancing(),
		{Class: worktype.CacheOptimization, Label: "cache-optimization", Err: "ledger unreadable"},
	})
	if p.Err != "" || !p.OK {
		t.Fatalf("a partial failure must NOT error the dimension, got err=%q ok=%v", p.Err, p.OK)
	}
	if p.PartialNote == "" {
		t.Fatalf("a partial failure must record a partial note")
	}
	if code, _ := CheckGate(Fold(p, FoldOpts{})); code != 0 {
		t.Fatalf("a partial (measured) report must gate 0, got %d", code)
	}
}

// TestFoldRegressedFrontierIsAdvisoryNotGated is the load-bearing posture test: a
// regressed program frontier is a MEASURED fact (advisory), not an incomplete report,
// so it must still gate 0.
func TestFoldRegressedFrontierIsAdvisoryNotGated(t *testing.T) {
	regressed := Signal{Class: worktype.CacheOptimization, Label: "cache-optimization", Frontier: "realized reuse fell 0.700 -> 0.500", Metric: 0.5, Direction: "regressed", OK: true}
	p := InterpretPrograms([]Signal{kernelAdvancing(), regressed})
	r := Fold(p, FoldOpts{Date: "2026-06-29"})
	if !r.OK || r.Finding != "programs_advisory" {
		t.Fatalf("a regressed frontier must record OK/advisory, got %+v", r)
	}
	if !strings.Contains(r.Reason, "regressed") {
		t.Fatalf("the advisory reason must name the regression, got %q", r.Reason)
	}
	if code, _ := CheckGate(r); code != 0 {
		t.Fatalf("an advisory report must gate 0, got %d", code)
	}
}

// TestLedgerRoundTripAndPerClassColumns proves the durable row round-trips and that
// the per-class metric columns are stamped from the right signal regardless of order.
func TestLedgerRoundTripAndPerClassColumns(t *testing.T) {
	p := InterpretPrograms([]Signal{cacheHolding(), kernelAdvancing()}) // cache first on purpose
	r := Fold(p, FoldOpts{Date: "2026-06-29", Commit: "abc", GeneratedAt: "2026-06-29T00:00:00Z"})
	row := RowFromReport(r)
	if row.KernelMetric != 3 || row.CacheMetric != 0.62 {
		t.Fatalf("per-class columns mis-stamped: kernel=%.3f cache=%.3f, want 3/0.62", row.KernelMetric, row.CacheMetric)
	}
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	rows := ParseLedger("\n" + line + "\nnot-json\n{}\n")
	if len(rows) != 1 || rows[0].Schema != LedgerSchema || rows[0].Date != "2026-06-29" {
		t.Fatalf("round-trip lost fields: %+v", rows)
	}
}

// TestTrendDirections pins the trend math: a rise in either metric is improved, a fall
// (with no rise) is regressed, equal is flat, no prior is new.
func TestTrendDirections(t *testing.T) {
	base := LedgerRow{Date: "2026-06-28", KernelMetric: 2, CacheMetric: 0.60, Advancing: 1, GeneratedAt: "t0"}
	if d := TrendVsLast(base, nil).Direction; d != "new" {
		t.Fatalf("first tick must be new, got %q", d)
	}
	up := LedgerRow{Date: "2026-06-29", KernelMetric: 4, CacheMetric: 0.60, GeneratedAt: "t1"}
	if d := TrendVsLast(up, []LedgerRow{base}).Direction; d != "improved" {
		t.Fatalf("a higher kernel metric must be improved, got %q", d)
	}
	down := LedgerRow{Date: "2026-06-29", KernelMetric: 2, CacheMetric: 0.40, GeneratedAt: "t1"}
	if d := TrendVsLast(down, []LedgerRow{base}).Direction; d != "regressed" {
		t.Fatalf("a lower cache metric must be regressed, got %q", d)
	}
	flat := LedgerRow{Date: "2026-06-29", KernelMetric: 2, CacheMetric: 0.60, GeneratedAt: "t1"}
	if d := TrendVsLast(flat, []LedgerRow{base}).Direction; d != "flat" {
		t.Fatalf("equal metrics must be flat, got %q", d)
	}
}

// TestRenderNoCompletionPercent is the key honesty test: the program render must NEVER
// show a completion % — an ongoing program has no 100%.
func TestRenderNoCompletionPercent(t *testing.T) {
	p := InterpretPrograms([]Signal{kernelAdvancing(), cacheHolding()})
	r := Fold(p, FoldOpts{Date: "2026-06-29", Commit: "abc"})
	r = r.WithTrend(TrendVsLast(RowFromReport(r), nil))
	out := Render(r)
	for _, want := range []string{"program report", "kernel-optimization", "cache-optimization", "frontier:", "trend:", "never 'done'"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
	if strings.Contains(out, "% complete") || strings.Contains(out, "100%") {
		t.Fatalf("a program report must never render a completion %%\n%s", out)
	}
}
