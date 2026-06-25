package main

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// TestTokenLedgerInvariants is the CI dog-food of this demo's data path: it replays
// every shipped suite through the REAL kernel under the file world (the exact
// buildLedger path -print / -json / -selfcheck drive) and asserts the documented
// two-meter invariants. The DEDUP / DENY classification underneath is the kernel's
// live verdict, so a drift in the kernel (a read that stops caching, a write that
// stops being denied) fails HERE, not silently in an operator's terminal.
func TestTokenLedgerInvariants(t *testing.T) {
	ctx := context.Background()
	for _, ks := range knownSuites {
		ks := ks
		t.Run(ks.ID, func(t *testing.T) {
			l, err := buildLedger(ctx, ks.ID)
			if err != nil {
				t.Fatalf("buildLedger(%s): %v", ks.ID, err)
			}
			exp, known := selfcheckExpect[ks.ID]
			if !known {
				t.Fatalf("no documented expectation for suite %q", ks.ID)
			}
			if l.Denies != exp.denies {
				t.Errorf("denies = %d, want %d", l.Denies, exp.denies)
			}
			if l.Dedups != exp.dedups {
				t.Errorf("dedups = %d, want %d", l.Dedups, exp.dedups)
			}
			if l.ContextTokensKept != exp.contextTokensKept {
				t.Errorf("context_tokens_kept_out = %d, want %d", l.ContextTokensKept, exp.contextTokensKept)
			}
			if l.RoundtripsCollapsed != exp.roundtripsCollapsed {
				t.Errorf("roundtrips_collapsed = %d, want %d", l.RoundtripsCollapsed, exp.roundtripsCollapsed)
			}
			if l.ToolTokensFromCache != exp.toolTokensFromCache {
				t.Errorf("tool_tokens_from_cache = %d, want %d", l.ToolTokensFromCache, exp.toolTokensFromCache)
			}
			// Load-bearing honesty invariants, true for EVERY suite:
			// (1) the MODEL-CONTEXT meter never costs more behind fak than raw, and
			// (2) a dedup-only suite (re-reads, no denies) cuts ZERO model context — the
			//     cached content is still re-served to the model, so the win is tool-side
			//     only. This is the exact overclaim the demo must not make.
			if l.CtxWith > l.CtxWithout {
				t.Errorf("model-context: WITH fak (%d) > WITHOUT fak (%d) — impossible", l.CtxWith, l.CtxWithout)
			}
			if l.Denies == 0 && l.CtxWith != l.CtxWithout {
				t.Errorf("dedup-only suite cut model context (without=%d with=%d) — that would be an overclaim; dedup is a tool-side win only",
					l.CtxWithout, l.CtxWith)
			}
			// The model-context win equals exactly the denied-result tokens kept out.
			if l.CtxWithout-l.CtxWith != l.ContextTokensKept {
				t.Errorf("ctx without(%d) - with(%d) != context_tokens_kept_out(%d)", l.CtxWithout, l.CtxWith, l.ContextTokensKept)
			}
		})
	}
}

// TestCleanControlInflatesNothing pins the anti-inflation guarantee on its own: a
// session with no bad call and no re-read saves EXACTLY zero on BOTH meters — both
// arms ingest the same model tokens and run each tool once. This is the "fak does not
// cry wolf" claim, machine-checked.
func TestCleanControlInflatesNothing(t *testing.T) {
	l, err := buildLedger(context.Background(), "clean-control")
	if err != nil {
		t.Fatal(err)
	}
	if l.ContextTokensKept != 0 {
		t.Errorf("clean control kept %d model tokens out, want 0 (fak must not inflate a clean path)", l.ContextTokensKept)
	}
	if l.RoundtripsCollapsed != 0 {
		t.Errorf("clean control collapsed %d round-trips, want 0", l.RoundtripsCollapsed)
	}
	if l.CtxWith != l.CtxWithout {
		t.Errorf("clean control: model-context with=%d != without=%d", l.CtxWith, l.CtxWithout)
	}
}

// TestDedupIsToolSideNotContext is the regression guard for the honesty fix: a re-read
// served from the content cache is a TOOL-SIDE win (the tool does not re-run), NOT a
// model-context cut — the gateway still re-materializes the cached bytes to the model.
// If a future change makes a dedup'd re-read claim a model-context saving, this fails.
func TestDedupIsToolSideNotContext(t *testing.T) {
	l, err := buildLedger(context.Background(), "reread-same-file")
	if err != nil {
		t.Fatal(err)
	}
	if l.Dedups == 0 {
		t.Fatal("expected the reread suite to produce real tier-2 dedup hits")
	}
	if l.ContextTokensKept != 0 {
		t.Errorf("dedup claimed %d model-context tokens kept out — overclaim; dedup is a tool-side win only", l.ContextTokensKept)
	}
	if l.RoundtripsCollapsed != l.Dedups {
		t.Errorf("roundtrips_collapsed (%d) should equal dedup hits (%d)", l.RoundtripsCollapsed, l.Dedups)
	}
}

func TestTimingProofShowsRepeatedReadsServedByVDSO(t *testing.T) {
	proof, err := buildTimingProof(context.Background(), "reread-same-file", time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if proof.RawEngineCalls != 6 {
		t.Fatalf("raw engine calls = %d, want 6", proof.RawEngineCalls)
	}
	if proof.FakEngineCalls != 3 {
		t.Fatalf("fak engine calls = %d, want 3 (first read/list only)", proof.FakEngineCalls)
	}
	if proof.VDSOHits != 3 || proof.RoundtripsCollapsed != 3 {
		t.Fatalf("vdso hits=%d roundtrips=%d, want 3/3", proof.VDSOHits, proof.RoundtripsCollapsed)
	}
	if proof.ToolTokensFromCache != 900 {
		t.Fatalf("tool tokens from cache = %d, want 900", proof.ToolTokensFromCache)
	}

	hits := 0
	for _, c := range proof.Calls {
		if c.FakSource != "vdso_tier2" {
			continue
		}
		hits++
		if c.EngineRanFak || c.FakEngineDelta != 0 {
			t.Fatalf("vdso row ran the engine: %+v", c)
		}
		if c.KernelVDSODelta != 1 {
			t.Fatalf("vdso row delta = %d, want 1: %+v", c.KernelVDSODelta, c)
		}
		if c.FakToolTimeNs <= 0 || c.RawToolTimeNs <= 0 {
			t.Fatalf("timing fields must be populated: %+v", c)
		}
	}
	if hits != 3 {
		t.Fatalf("vdso_tier2 rows = %d, want 3", hits)
	}
}

// TestDenyVerdictBounded guards the one constant the prefilter win rests on: a deny
// verdict that enters context in place of an executed bad call's result must stay a
// SMALL, bounded size (the refusal vocabulary is closed). If it ever grows large, the
// prefilter win is no longer real and this fails.
func TestDenyVerdictBounded(t *testing.T) {
	if denyVerdictTokens <= 0 || denyVerdictTokens > 128 {
		t.Errorf("denyVerdictTokens = %d, want a small positive bound (a closed-vocabulary refusal, not a tool result)", denyVerdictTokens)
	}
}

// TestResultTokensAnnotation checks the documented knob is read from trace meta and
// falls back cleanly when absent or malformed.
func TestResultTokensAnnotation(t *testing.T) {
	cases := []struct {
		meta map[string]string
		want int
	}{
		{map[string]string{"result_tokens": "512"}, 512},
		{map[string]string{"result_tokens": "0"}, 0},
		{map[string]string{"result_tokens": " 64 "}, 64},
		{map[string]string{"result_tokens": "-5"}, defaultResultTokens},  // negative -> fallback
		{map[string]string{"result_tokens": "abc"}, defaultResultTokens}, // malformed -> fallback
		{map[string]string{}, defaultResultTokens},                       // absent -> fallback
		{nil, defaultResultTokens},
	}
	for i, c := range cases {
		got := resultTokens(turnbench.Call{Meta: c.meta})
		if got != c.want {
			t.Errorf("case %d: resultTokens = %d, want %d", i, got, c.want)
		}
	}
}

func TestJSONDefaultsToAllSuites(t *testing.T) {
	if got := selectedSuiteForJSON("prefilter-bad-calls", false); got != "all" {
		t.Fatalf("implicit -json suite = %q, want all", got)
	}
	if got := selectedSuiteForJSON("clean-control", true); got != "clean-control" {
		t.Fatalf("explicit -json suite = %q, want clean-control", got)
	}
}
