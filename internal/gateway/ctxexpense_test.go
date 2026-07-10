package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// ctxexpense_test.go — the "warns/blocks for unusually expensive sessions" witness.
// It proves the pure policy grades the per-turn as-sent volume against the
// warn/block lines, that detection is on by default, that the verdict rides the
// CtxValueReport and the --debug-stats field, and that the block advisory ACTUATES
// only behind the soak gate and only once per session.

func TestAssessCtxExpenseRungs(t *testing.T) {
	cases := []struct {
		name      string
		st        ctxExpenseState
		wantLevel ExpenseLevel
		wantBasis string
	}{
		{"no footprint", ctxExpenseState{TurnTokens: 0, WarnTokens: 100, BlockTokens: 200}, ExpenseOK, "none"},
		{"no thresholds", ctxExpenseState{TurnTokens: 500, WarnTokens: 0, BlockTokens: 0}, ExpenseOK, "none"},
		{"below warn", ctxExpenseState{TurnTokens: 99, WarnTokens: 100, BlockTokens: 200}, ExpenseOK, "per_turn_volume"},
		{"at warn", ctxExpenseState{TurnTokens: 100, WarnTokens: 100, BlockTokens: 200}, ExpenseWarn, "per_turn_volume"},
		{"between warn and block", ctxExpenseState{TurnTokens: 150, WarnTokens: 100, BlockTokens: 200}, ExpenseWarn, "per_turn_volume"},
		{"at block", ctxExpenseState{TurnTokens: 200, WarnTokens: 100, BlockTokens: 200}, ExpenseBlock, "per_turn_volume"},
		{"above block", ctxExpenseState{TurnTokens: 999, WarnTokens: 100, BlockTokens: 200}, ExpenseBlock, "per_turn_volume"},
		{"warn-only tier", ctxExpenseState{TurnTokens: 999, WarnTokens: 100, BlockTokens: 0}, ExpenseWarn, "per_turn_volume"},
		{"block-only tier below", ctxExpenseState{TurnTokens: 150, WarnTokens: 0, BlockTokens: 200}, ExpenseOK, "per_turn_volume"},
		{"block-only tier hit", ctxExpenseState{TurnTokens: 250, WarnTokens: 0, BlockTokens: 200}, ExpenseBlock, "per_turn_volume"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := assessCtxExpense(tc.st)
			if got.Level != tc.wantLevel {
				t.Fatalf("level = %q, want %q (reason: %s)", got.Level, tc.wantLevel, got.Reason)
			}
			if got.Basis != tc.wantBasis {
				t.Fatalf("basis = %q, want %q", got.Basis, tc.wantBasis)
			}
			if got.Provenance != "DECISION" {
				t.Fatalf("provenance = %q, want DECISION (Law A2: a decision over an ESTIMATED input)", got.Provenance)
			}
			if got.Reason == "" {
				t.Fatal("every verdict must carry a reason naming the deciding numbers")
			}
		})
	}
}

func TestCtxExpenseThresholdOr(t *testing.T) {
	if got := ctxExpenseThresholdOr(0, 120_000); got != 120_000 {
		t.Fatalf("0 should take the default (on-by-default), got %d", got)
	}
	if got := ctxExpenseThresholdOr(50_000, 120_000); got != 50_000 {
		t.Fatalf("a positive value should override, got %d", got)
	}
	if got := ctxExpenseThresholdOr(-1, 120_000); got != 0 {
		t.Fatalf("a negative value should disable the tier (0), got %d", got)
	}
}

func TestFormatExpenseField(t *testing.T) {
	ok := assessCtxExpense(ctxExpenseState{TurnTokens: 10, WarnTokens: 100, BlockTokens: 200})
	if f := formatExpenseField(ok, true); f != "" {
		t.Fatalf("ok tier must render no field (clean line), got %q", f)
	}
	if f := formatExpenseField(assessCtxExpense(ctxExpenseState{TurnTokens: 150, WarnTokens: 100, BlockTokens: 200}), false); f != "" {
		t.Fatalf("have=false must render no field, got %q", f)
	}
	warn := assessCtxExpense(ctxExpenseState{TurnTokens: 150, WarnTokens: 100, BlockTokens: 200})
	if f := formatExpenseField(warn, true); !strings.HasPrefix(f, " expense=warn vol=") {
		t.Fatalf("warn field shape = %q", f)
	}
	block := assessCtxExpense(ctxExpenseState{TurnTokens: 250, WarnTokens: 100, BlockTokens: 200})
	if f := formatExpenseField(block, true); !strings.HasPrefix(f, " expense=block vol=") {
		t.Fatalf("block field shape = %q", f)
	}
}

// observeExpenseFootprint drives a real inbound footprint through the live seam so
// the report/peek paths read a genuine v.footprint, then returns the ESTIMATED
// per-turn token volume the verdict grades on.
func observeExpenseFootprint(t *testing.T, srv *Server, trace string) int {
	t.Helper()
	req := &agent.AnthropicMessagesRequest{
		System:   strings.Repeat("x", 8000),
		Messages: []agent.Message{{Role: "user", Content: "hi"}},
	}
	srv.observeCtxFootprint(trace, req)
	fp := srv.CtxValueReportFor(trace).Footprint
	if fp == nil {
		t.Fatal("expected a footprint after observeCtxFootprint")
	}
	return fp.TotalBytes / estBytesPerToken
}

func TestCtxExpenseReportWiring(t *testing.T) {
	srv := newExposeServer(t)
	const trace = "t-expense"
	tok := observeExpenseFootprint(t, srv, trace)

	// Straddle the observed volume so it lands in the warn band.
	srv.ctxExpenseWarn = tok - 1
	srv.ctxExpenseBlock = tok + 1_000_000
	if got := srv.CtxValueReportFor(trace).Expense; got.Level != ExpenseWarn {
		t.Fatalf("report expense level = %q, want warn (vol=%d tok)", got.Level, tok)
	}
	// The peek surface (debug line) must agree with the report.
	if e, ok := srv.peekCtxExpense(trace); !ok || e.Level != ExpenseWarn {
		t.Fatalf("peek disagreed with report: ok=%v level=%q", ok, e.Level)
	}

	// Drop the block line below the volume: now block.
	srv.ctxExpenseBlock = tok - 1
	if got := srv.CtxValueReportFor(trace).Expense; got.Level != ExpenseBlock {
		t.Fatalf("report expense level = %q, want block", got.Level)
	}
}

func TestCtxExpenseReportZeroTraceIsDecidable(t *testing.T) {
	srv := newExposeServer(t)
	// A trace with no served turn still returns a decidable ok/none verdict carrying
	// the configured lines — never an error, never a phantom warn.
	e := srv.CtxValueReportFor("never-seen").Expense
	if e.Level != ExpenseOK || e.Basis != "none" {
		t.Fatalf("zero-trace expense = %q/%q, want ok/none", e.Level, e.Basis)
	}
	if e.WarnTokens != srv.ctxExpenseWarn || e.BlockTokens != srv.ctxExpenseBlock {
		t.Fatalf("zero-trace verdict must carry the configured lines, got warn=%d block=%d", e.WarnTokens, e.BlockTokens)
	}
}

func TestCtxExpenseNoteOnceGated(t *testing.T) {
	srv := newExposeServer(t)
	const trace = "t-gate"
	tok := observeExpenseFootprint(t, srv, trace)
	srv.ctxExpenseBlock = tok - 1 // force block tier
	srv.ctxExpenseWarn = 1

	// Gate OFF (the default): the block tier is view-only, no in-band note.
	srv.ctxExpenseGate = false
	if note := srv.ctxExpenseNoteOnce(trace); note != "" {
		t.Fatalf("gate off must emit no advisory, got %q", note)
	}

	// Gate ON: the first block-tier turn emits one advisory, then dedups.
	srv.ctxExpenseGate = true
	first := srv.ctxExpenseNoteOnce(trace)
	if !strings.HasPrefix(first, "[fak] ") || !strings.Contains(first, "context-expense gate") {
		t.Fatalf("gated block advisory shape = %q", first)
	}
	if again := srv.ctxExpenseNoteOnce(trace); again != "" {
		t.Fatalf("advisory must fire once per session, got a repeat: %q", again)
	}
}

func TestCtxExpenseNoteOnceNotBlockedIsSilent(t *testing.T) {
	srv := newExposeServer(t)
	const trace = "t-warn-only"
	tok := observeExpenseFootprint(t, srv, trace)
	// Warn tier but NOT block: the gated advisory only fires at block.
	srv.ctxExpenseGate = true
	srv.ctxExpenseWarn = tok - 1
	srv.ctxExpenseBlock = tok + 1_000_000
	if note := srv.ctxExpenseNoteOnce(trace); note != "" {
		t.Fatalf("warn tier must not trip the block advisory, got %q", note)
	}
}
