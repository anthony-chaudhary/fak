package defaultvaluescore

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
	k := kpiValueFlagDefaultOn(flags, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
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
	k := kpiValueFlagDefaultOn(flags, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
	if len(k.Defects) != 0 {
		t.Errorf("a default-on value-flag is clean, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestValueFlag_BoolNamedConstantDefaultIsOn(t *testing.T) {
	// A Bool value-flag defaulted from a NAMED CONSTANT (the shipped-enabled idiom) is ON,
	// not debt -- the --elide-stale-reads regression: gateway.DefaultElideStaleReads is
	// true, so the flag ships enabled, but the Bool judge once matched only the bare `true`
	// literal and false-positived it as VALUE_FLAG_OFF. This pins the named-constant rule
	// for Bool the same way TestValueFlag_DefaultOnIsClean pins it for Int.
	src := `fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "ON by default: replace a stale Read result")`
	flags := ParseFlags(src, "cmd/fak/guard.go")
	if len(flags) != 1 || !flags[0].defaultOn {
		t.Fatalf("a Bool value-flag with a named-constant default must judge default-ON, got %+v", flags)
	}
	k := kpiValueFlagDefaultOn(flags, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
	if len(k.Defects) != 0 {
		t.Errorf("a Bool named-constant default-on value-flag is clean, got %v", k.Defects)
	}
	// A bare `false` literal is still the off sentinel and remains debt when not allow-listed.
	off := ParseFlags(`fs.Bool("elide-mystery", false, "shed something")`, "cmd/fak/serve.go")
	if len(off) != 1 || off[0].defaultOn {
		t.Fatalf("a bare false Bool default must still judge default-OFF, got %+v", off)
	}
}

func TestValueFlag_OffWithAllowlistReasonIsClean(t *testing.T) {
	// An allow-listed OFF value-flag (a genuine gate with a documented reason) is honest.
	src := `fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine")`
	flags := ParseFlags(src, "cmd/fak/serve.go")
	if len(flags) != 1 || flags[0].defaultOn {
		t.Fatalf("empty-string default must judge default-OFF, got %+v", flags)
	}
	k := kpiValueFlagDefaultOn(flags, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
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

func TestOptInGateReviewDateBoundsOmissionHarm(t *testing.T) {
	flags := ParseFlags(`fs.Bool("vdso-proxy-fill", false, "value cache speedup")`, FlagSources[0])
	before := kpiValueFlagDefaultOn(flags, time.Date(2026, 9, 30, 0, 0, 0, 0, time.UTC))
	if len(before.Defects) != 0 {
		t.Fatalf("current reviewed gate should be inside the window: %v", before.Defects)
	}
	after := kpiValueFlagDefaultOn(flags, time.Date(2026, 10, 2, 0, 0, 0, 0, time.UTC))
	if len(after.Defects) != 1 || !strings.Contains(after.Defects[0], "OPT_IN_REVIEW_DUE") {
		t.Fatalf("stale gate must expose omission harm as typed debt: %v", after.Defects)
	}
}

func TestInvalidOptInReviewDateIsDebt(t *testing.T) {
	if validReviewDate("later", time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)) {
		t.Fatal("an unparseable review date must not hold the opt-in gate open")
	}
}

func TestBuildAsOfReportsAgenticDefaultWindow(t *testing.T) {
	p := BuildAsOf("../..", time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
	if p.Schema != Schema {
		t.Fatalf("schema = %q, want %q", p.Schema, Schema)
	}
	if got := p.Corpus["reviewed_opt_in_flags"]; got != len(offWithReason) {
		t.Fatalf("reviewed_opt_in_flags = %v, want %d", got, len(offWithReason))
	}
	if got := p.Corpus["next_default_review"]; got != "2026-10-01" {
		t.Fatalf("next_default_review = %v, want 2026-10-01", got)
	}
	if !strings.Contains(p.Finding, "agentic default window") {
		t.Fatalf("finding must spell out the monitored boundary, got %q", p.Finding)
	}
}

// --- Check 2: cross-context default parity (VALUE_FLAG_CONTEXT_DRIFT) ---------------------

func TestContextParity_DriftIsDebt(t *testing.T) {
	// The same value saver, offered on BOTH serving surfaces, but default-ON under guard and
	// default-OFF under serve: the win is silently disabled on the serve path. This is the
	// class kpiValueFlagDefaultOn cannot see (it judges each surface alone -- both look fine).
	flags := append(
		ParseFlags(`fs.Bool("compact-anchor-head", true, "anchor the compaction head")`, "cmd/fak/guard.go"),
		ParseFlags(`fs.Bool("compact-anchor-head", false, "anchor the compaction head")`, "cmd/fak/serve.go")...)
	if len(flags) != 2 {
		t.Fatalf("expected the same value-flag parsed from 2 surfaces, got %d: %+v", len(flags), flags)
	}
	// Each surface in isolation is clean (guard ON is fine; serve OFF alone would be a
	// separate VALUE_FLAG_OFF, but the point here is parity specifically).
	k := kpiValueFlagContextParity(flags)
	if len(k.Defects) != 1 {
		t.Fatalf("a saver ON in guard but OFF in serve must be 1 parity defect, got %d: %v", len(k.Defects), k.Defects)
	}
	if !strings.Contains(k.Defects[0], "VALUE_FLAG_CONTEXT_DRIFT") {
		t.Errorf("defect must carry the VALUE_FLAG_CONTEXT_DRIFT class, got %q", k.Defects[0])
	}
	// The witness must name both surfaces and their opposed postures, deterministically.
	if !strings.Contains(k.Defects[0], "guard.go=on") || !strings.Contains(k.Defects[0], "serve.go=off") {
		t.Errorf("defect must witness both per-surface postures, got %q", k.Defects[0])
	}
	if k.Score >= 100.0 {
		t.Errorf("score should drop below 100 on a parity drift, got %v", k.Score)
	}
}

func TestContextParity_AgreementIsClean(t *testing.T) {
	// The shipped idiom: a saver default-ON on BOTH serving surfaces. Parity holds, no debt.
	flags := append(
		ParseFlags(`fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "shed old turns")`, "cmd/fak/guard.go"),
		ParseFlags(`fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "shed old turns")`, "cmd/fak/serve.go")...)
	k := kpiValueFlagContextParity(flags)
	if len(k.Defects) != 0 {
		t.Errorf("a saver default-ON on both surfaces is parity-clean, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean parity score=%v want 100", k.Score)
	}
}

func TestContextParity_RoleSpecificSingleSurfaceIsClean(t *testing.T) {
	// A value saver offered on ONLY one serving surface (a serve-only engine knob, a
	// guard-only session lever) makes NO cross-context parity claim, so it is never a false
	// positive -- even when its lone default is OFF. This is the guard against the naive
	// "missing from the other surface = defect" over-reach.
	flags := ParseFlags(`fs.Bool("engine-cache-require-exact-span", false, "serve-only inference-engine knob")`, "cmd/fak/serve.go")
	if len(flags) != 1 {
		t.Fatalf("expected 1 value-flag parsed, got %d: %+v", len(flags), flags)
	}
	k := kpiValueFlagContextParity(flags)
	if len(k.Defects) != 0 {
		t.Errorf("a saver present on a single surface makes no parity claim, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("a single-surface saver is vacuously parity-clean, score=%v want 100", k.Score)
	}
}

// --- Check 3: no vacuous kernel.Counters fold on the proxy -------------------------------

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

func TestLiveTreeContextParityWiredAndClean(t *testing.T) {
	// Proves the cross-context parity KPI is (a) actually folded into the payload the
	// `fak score default-value` verb renders -- Payload.KPIs IS what the verb prints -- and
	// (b) currently clean on the real guard.go/serve.go: every value saver shared by both
	// serving surfaces ships the same default. If a future edit sets a shared saver ON in
	// one surface but OFF in the other, this fails before the drift ships.
	p := Build("../..")
	var parity *scorecard.KPI
	for i := range p.KPIs {
		if p.KPIs[i].Key == "value_flag_context_parity" {
			parity = &p.KPIs[i]
			break
		}
	}
	if parity == nil {
		t.Fatalf("value_flag_context_parity KPI must be present in the payload the verb renders; got keys %v", kpiKeys(p.KPIs))
	}
	if len(parity.Defects) != 0 {
		t.Errorf("live tree has a cross-context saver default drift: %v", parity.Defects)
	}
}

func kpiKeys(kpis []scorecard.KPI) []string {
	keys := make([]string, 0, len(kpis))
	for _, k := range kpis {
		keys = append(keys, k.Key)
	}
	return keys
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
