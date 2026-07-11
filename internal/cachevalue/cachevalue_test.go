package cachevalue

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// issueFixture is the five-session 2026-07-01 evidence table from #1992 as
// ledger JSONL. The T04:13 and T04:42 lines are verbatim rows from
// docs/nightrun/cache-savings.jsonl; the T12:57 / T13:17 / T15:07 rows have
// since been rotated out of the tracked ledger, so they carry the issue
// table's cache_read/cache_creation with input_tokens reconstructed to
// reproduce the published hit% (the derived columns below are what the issue
// asserts, not what this package assumes).
//
//	generated_at  cache_read  cache_creation  hit%   write-amp
//	T12:57        12,545,503     136,801      98.8     0.011   (best)
//	T13:17        11,865,727     131,099      98.8     0.011
//	T15:07         9,120,490     788,202      91.3     0.086   (churn spike, borderline-healthy)
//	T04:13           434,880      73,054      83.6     0.168
//	T04:42            48,975      12,538      79.6     0.256   (worst)
const issueFixture = `{"schema":"fak-cache-savings-ledger/1","date":"2026-07-01","session_type":"guard","provider":"anthropic","mechanism":"provider_prompt_cache","context":"claude","generated_at":"2026-07-01T12:57:00Z","input_tokens":15573,"cache_read_tokens":12545503,"cache_creation_tokens":136801,"output_tokens":0}
{"schema":"fak-cache-savings-ledger/1","date":"2026-07-01","session_type":"guard","provider":"anthropic","mechanism":"provider_prompt_cache","context":"claude","generated_at":"2026-07-01T13:17:00Z","input_tokens":13019,"cache_read_tokens":11865727,"cache_creation_tokens":131099,"output_tokens":0}
{"schema":"fak-cache-savings-ledger/1","date":"2026-07-01","session_type":"guard","provider":"anthropic","mechanism":"provider_prompt_cache","context":"claude","generated_at":"2026-07-01T15:07:00Z","input_tokens":80892,"cache_read_tokens":9120490,"cache_creation_tokens":788202,"output_tokens":0}
{"schema":"fak-cache-savings-ledger/1","date":"2026-07-01","session_type":"guard","provider":"anthropic","mechanism":"provider_prompt_cache","context":"claude","generated_at":"2026-07-01T04:13:31Z","input_tokens":12361,"cache_read_tokens":434880,"cache_creation_tokens":73054,"output_tokens":1644,"saved_token_equiv":373128.5,"net_saved_token_equiv":373128.5,"rebate_usd":0,"write_premium_usd":0,"spend_usd":0,"net_usd":0}
{"schema":"fak-cache-savings-ledger/1","date":"2026-07-01","session_type":"guard","provider":"anthropic","mechanism":"provider_prompt_cache","context":"claude","generated_at":"2026-07-01T04:42:02Z","input_tokens":2,"cache_read_tokens":48975,"cache_creation_tokens":12538,"output_tokens":935,"saved_token_equiv":40943,"net_saved_token_equiv":40943,"rebate_usd":0,"write_premium_usd":0,"spend_usd":0,"net_usd":0}
`

// tol matches the issue table's rounding: hit% to 0.1 point, write-amp to
// three decimals, both within 5e-4 of the exact ratios.
const tol = 5e-4

func within(got, want, tol float64) bool {
	d := got - want
	return d <= tol && d >= -tol
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func foldFixture(t *testing.T) []Metrics {
	t.Helper()
	ms, err := Fold(strings.NewReader(issueFixture))
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(ms) != 5 {
		t.Fatalf("Fold returned %d sessions, want 5", len(ms))
	}
	return ms
}

func TestFoldMatchesIssueTable(t *testing.T) {
	ms := foldFixture(t)
	want := []struct {
		generatedAt       string
		hitRate, writeAmp float64
	}{
		{"2026-07-01T12:57:00Z", 0.988, 0.011},
		{"2026-07-01T13:17:00Z", 0.988, 0.011},
		{"2026-07-01T15:07:00Z", 0.913, 0.086},
		{"2026-07-01T04:13:31Z", 0.836, 0.168},
		{"2026-07-01T04:42:02Z", 0.796, 0.256},
	}
	for i, w := range want {
		m := ms[i]
		if m.GeneratedAt != w.generatedAt {
			t.Fatalf("session %d = %q, want %q (ledger order must be preserved)", i, m.GeneratedAt, w.generatedAt)
		}
		if !m.HitRateKnown || !m.WriteAmpKnown {
			t.Errorf("%s: known bits hit=%v wa=%v, want both true", w.generatedAt, m.HitRateKnown, m.WriteAmpKnown)
		}
		if !within(m.HitRate, w.hitRate, tol) {
			t.Errorf("%s: hit rate = %.6f, want %.3f +/- %g", w.generatedAt, m.HitRate, w.hitRate, tol)
		}
		if !within(m.WriteAmp, w.writeAmp, tol) {
			t.Errorf("%s: write-amp = %.6f, want %.3f +/- %g", w.generatedAt, m.WriteAmp, w.writeAmp, tol)
		}
	}
}

func TestFlagRegressionsDefaults(t *testing.T) {
	ms := foldFixture(t)
	flags := FlagRegressions(ms, DefaultThresholds())

	flagged := map[string][]string{}
	for _, f := range flags {
		flagged[f.GeneratedAt] = f.Reasons
	}
	// The worst session (T04:42, 79.6% / 0.256) must be flagged on both rungs.
	worst, ok := flagged["2026-07-01T04:42:02Z"]
	if !ok {
		t.Fatalf("worst session T04:42 not flagged; flags = %v", flagged)
	}
	if !contains(worst, ReasonHitFloor) || !contains(worst, ReasonWriteAmpCeil) {
		t.Errorf("worst session reasons = %v, want both %q and %q", worst, ReasonHitFloor, ReasonWriteAmpCeil)
	}
	// The best session (T12:57, 98.8% / 0.011) must not be flagged.
	if r, ok := flagged["2026-07-01T12:57:00Z"]; ok {
		t.Errorf("best session T12:57 flagged with %v, want unflagged", r)
	}
	// T04:13 (83.6% / 0.168) regresses both rungs too; the T15:07 churn spike
	// (91.3% / 0.086) sits just inside both default gates and stays quiet.
	if _, ok := flagged["2026-07-01T04:13:31Z"]; !ok {
		t.Errorf("T04:13 not flagged; flags = %v", flagged)
	}
	if r, ok := flagged["2026-07-01T15:07:00Z"]; ok {
		t.Errorf("borderline T15:07 flagged with %v, want unflagged under defaults", r)
	}
	if len(flags) != 2 {
		t.Errorf("flag count = %d (%v), want exactly 2 (T04:13, T04:42)", len(flags), flagged)
	}

	// The zero-value Thresholds normalizes to the same defaults.
	zero := FlagRegressions(ms, Thresholds{})
	if len(zero) != len(flags) {
		t.Errorf("zero-value thresholds flagged %d sessions, want %d (must normalize to defaults)", len(zero), len(flags))
	}
}

func TestFlagRegressionsCustomThresholds(t *testing.T) {
	ms := foldFixture(t)
	// A 0.92 floor pulls the T15:07 churn spike (91.3%) into the flagged set.
	flags := FlagRegressions(ms, Thresholds{HitRateFloor: 0.92, WriteAmpCeil: DefaultWriteAmpCeil})
	found := false
	for _, f := range flags {
		if f.GeneratedAt == "2026-07-01T15:07:00Z" {
			found = contains(f.Reasons, ReasonHitFloor)
		}
	}
	if !found {
		t.Errorf("T15:07 not hit-flagged under a 0.92 floor; flags = %+v", flags)
	}
}

func TestDivideByZeroGuards(t *testing.T) {
	fixture := `{"generated_at":"all-zero","input_tokens":0,"cache_read_tokens":0,"cache_creation_tokens":0}
{"generated_at":"writes-no-reads","input_tokens":10,"cache_read_tokens":0,"cache_creation_tokens":5000}
`
	ms, err := Fold(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(ms) != 2 {
		t.Fatalf("Fold returned %d sessions, want 2", len(ms))
	}

	zero := ms[0]
	if zero.HitRateKnown {
		t.Errorf("all-zero row: HitRateKnown = true, want false (no phantom 0%% hit rate)")
	}
	if !zero.WriteAmpKnown || zero.WriteAmp != 0 {
		t.Errorf("all-zero row: write-amp = %v known=%v, want 0 known (no writes, no amplification)", zero.WriteAmp, zero.WriteAmpKnown)
	}

	unbounded := ms[1]
	if unbounded.WriteAmpKnown {
		t.Errorf("writes-no-reads row: WriteAmpKnown = true, want false (unbounded)")
	}

	flags := FlagRegressions(ms, DefaultThresholds())
	if len(flags) != 1 || flags[0].GeneratedAt != "writes-no-reads" {
		t.Fatalf("flags = %+v, want exactly the writes-no-reads session (idle all-zero row must stay quiet)", flags)
	}
	if !contains(flags[0].Reasons, ReasonWriteAmpUnbounded) {
		t.Errorf("writes-no-reads reasons = %v, want %q", flags[0].Reasons, ReasonWriteAmpUnbounded)
	}
}

func TestFoldSkipsBlankLines(t *testing.T) {
	fixture := "\n{\"generated_at\":\"a\",\"cache_read_tokens\":1,\"cache_creation_tokens\":1,\"input_tokens\":1}\n\n  \n"
	ms, err := Fold(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(ms) != 1 || ms[0].GeneratedAt != "a" {
		t.Fatalf("metrics = %+v, want the single non-blank row", ms)
	}
}

func TestFoldMalformedLineErrors(t *testing.T) {
	fixture := "{\"generated_at\":\"ok\"}\nnot json\n"
	if _, err := Fold(strings.NewReader(fixture)); err == nil {
		t.Fatal("Fold on a malformed line = nil error, want the line-numbered parse error")
	} else if !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("error = %v, want it to name line 2", err)
	}
}

func TestFoldFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	if err := os.WriteFile(path, []byte(issueFixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ms, err := FoldFile(path)
	if err != nil {
		t.Fatalf("FoldFile: %v", err)
	}
	if len(ms) != 5 {
		t.Fatalf("FoldFile returned %d sessions, want 5", len(ms))
	}
	if _, err := FoldFile(filepath.Join(t.TempDir(), "absent.jsonl")); err == nil {
		t.Fatal("FoldFile on an absent path = nil error, want open error")
	}
}
