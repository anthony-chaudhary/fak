package defaultvaluescore

import (
	"strings"
	"testing"
)

// These mirror the conflationscore_test.go discipline: each KPI on a DEFECT fixture (the
// regression it must catch) and a CLEAN fixture (the honest shape it must pass). The
// fixtures are tiny synthetic source snippets in exactly the form the parser reads, so a
// clean live tree's missing defect branches are still proven here.

// --- Check 1: value-flag default-on / gated-with-reason ----------------------------------

func TestValueFlag_OffWithoutReasonIsDebt(t *testing.T) {
	// A value-flag (name carries a value token) shipped default-OFF and NOT allow-listed.
	src := `fs.Int("compact-mystery-budget", 0, "shed old turns to this budget")`
	flags := ParseFlags(src, "cmd/fak/serve.go")
	if len(flags) != 1 {
		t.Fatalf("expected 1 value-flag parsed, got %d: %+v", len(flags), flags)
	}
	if flags[0].defaultOn {
		t.Error("a 0-default Int value-flag must judge default-OFF")
	}
	k := kpiValueFlagDefaultOn(flags)
	if len(k.Defects) == 0 {
		t.Error("an OFF value-flag with no allow-list reason must be debt (VALUE_FLAG_OFF)")
	}
	if k.Score >= 100.0 {
		t.Errorf("score should drop below 100 on a defect, got %v", k.Score)
	}
}

func TestValueFlag_DefaultOnIsClean(t *testing.T) {
	// A value-flag shipped default-ON (a named-constant budget) is the shipped-enabled idiom.
	src := `fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "shed old turns")`
	flags := ParseFlags(src, "cmd/fak/guard.go")
	if len(flags) != 1 || !flags[0].defaultOn {
		t.Fatalf("named-constant default must judge default-ON, got %+v", flags)
	}
	k := kpiValueFlagDefaultOn(flags)
	if len(k.Defects) != 0 {
		t.Errorf("a default-on value-flag is clean, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestValueFlag_OffWithAllowlistReasonIsClean(t *testing.T) {
	// An allow-listed OFF value-flag (a genuine gate with a documented reason) is honest.
	src := `fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine")`
	flags := ParseFlags(src, "cmd/fak/serve.go")
	if len(flags) != 1 || flags[0].defaultOn {
		t.Fatalf("empty-string default must judge default-OFF, got %+v", flags)
	}
	k := kpiValueFlagDefaultOn(flags)
	if len(k.Defects) != 0 {
		t.Errorf("an allow-listed OFF value-flag is honest, got %v", k.Defects)
	}
}

func TestValueFlag_TransportFlagIsOutOfScope(t *testing.T) {
	// A pure transport/identity flag is NOT a value lever even if its help mentions cache;
	// only the NAME is matched, so it must be ignored entirely.
	src := `fs.String("session-id", "", "default trace/session id; affects the cache key")`
	flags := ParseFlags(src, "cmd/fak/serve.go")
	if len(flags) != 0 {
		t.Errorf("a transport flag whose help mentions cache must be out of scope, got %+v", flags)
	}
}

// --- Check 2: no vacuous kernel.Counters fold on the proxy -------------------------------

func TestCounterFold_NoProxyGuardIsDebt(t *testing.T) {
	src := `
func formatBad(kc kernel.Counters) string {
	return fmt.Sprintf("fak guard: amplification %dx", kc.VDSOHits+kc.Transforms)
}`
	k := kpiNoVacuousCounterFold(src, "cmd/fak/guard.go")
	if len(k.Defects) == 0 {
		t.Error("a kernel.Counters fold into a `fak guard:` line with no proxy marker is debt (VACUOUS_ON_GUARD)")
	}
	if k.Score >= 100.0 {
		t.Errorf("score should drop on a vacuous fold, got %v", k.Score)
	}
}

func TestCounterFold_ProxyAwareIsClean(t *testing.T) {
	// The canonical formatAmplification shape: it reads counters BUT splits the proxy path
	// and frames the line honestly ("proxy path: ... Decide ..."), so it is not debt.
	src := `
func formatGood(kc kernel.Counters) string {
	if kc.VDSOHits == 0 && kc.Transforms == 0 {
		return "fak guard: floor effect (proxy path: the kernel adjudicates with Decide, so the in-kernel axis does not apply)"
	}
	return fmt.Sprintf("fak guard: amplification %dx", kc.VDSOHits)
}`
	k := kpiNoVacuousCounterFold(src, "cmd/fak/guard.go")
	if len(k.Defects) != 0 {
		t.Errorf("a proxy-aware counter fold is honest, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestCounterFold_NonGuardLineIgnored(t *testing.T) {
	// A function that reads counters but renders no `fak guard:` exit line is not an exit
	// surface; it must not count toward the fold check at all.
	src := `
func tally(kc kernel.Counters) int64 { return kc.VDSOHits + kc.Denies }`
	k := kpiNoVacuousCounterFold(src, "cmd/fak/guard.go")
	if len(k.Defects) != 0 {
		t.Errorf("a non-exit counter read must be ignored, got %v", k.Defects)
	}
}

// --- Check 3: observed-not-modeled default headline -------------------------------------

func TestModeledDefault_PlannedHeadlineIsDebt(t *testing.T) {
	surfaces := map[string]string{"x/score.go": `activeSource := "planned"`}
	k := kpiObservedNotModeledDefault(surfaces)
	if len(k.Defects) == 0 {
		t.Error("a default headline source of \"planned\" is debt (C_MODELED_NOT_OBSERVED)")
	}
	if k.Score != 0.0 {
		t.Errorf("score should be 0 on the only surface defaulting modeled, got %v", k.Score)
	}
}

func TestModeledDefault_ObservedHeadlineIsClean(t *testing.T) {
	surfaces := map[string]string{"x/score.go": `activeSource := "telemetry" // observed`}
	k := kpiObservedNotModeledDefault(surfaces)
	if len(k.Defects) != 0 {
		t.Errorf("an observed default headline is honest, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

// --- envelope + live-tree floor ---------------------------------------------------------

func TestBuildEnvelopeShape(t *testing.T) {
	p := Build("../..") // internal/defaultvaluescore -> repo root
	if p.Schema != Schema {
		t.Errorf("schema=%q want %q", p.Schema, Schema)
	}
	for _, key := range []string{DebtKey, "grade", "score", "value_flags_seen", "value_flags_off", "score_surfaces"} {
		if _, ok := p.Corpus[key]; !ok {
			t.Errorf("corpus missing key %q: %v", key, p.Corpus)
		}
	}
	if p.Verdict == "" || p.Finding == "" || p.NextAction == "" {
		t.Error("envelope prose fields must be populated")
	}
}

func TestLiveTreeFloorPinned(t *testing.T) {
	// The regression sentinel: the real flag + exit + score surfaces must not regrow
	// default-value debt above the known, tracked backlog (CleanFloor).
	p := Build("../..")
	got := anyIntCorpus(p.Corpus[DebtKey])
	if got > CleanFloor {
		t.Errorf("default-value debt rose above the floor %d: %d (%s)", CleanFloor, got, p.Reason)
	}
}

// anyIntCorpus coerces the corpus debt (an int written by the kernel) to int for the floor
// comparison without importing the kernel's unexported helper.
func anyIntCorpus(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

// guard against an unused-import lint if the strings import is ever dropped from a fixture.
var _ = strings.Contains
