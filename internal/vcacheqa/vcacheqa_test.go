package vcacheqa

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// These three tests are the crux the issue names explicitly: a QA harness that
// cannot catch a planted violation is worthless. Each plants exactly one of
// the three violation classes named in #1495's acceptance criteria and
// proves the harness's own detector flags it, alongside a clean fixture
// proving the same detector does NOT flag honest code/prose — the same
// defect+clean pairing convention internal/conflationscore's own test file
// already uses (conflationscore_test.go).

// --- 1. warmth-gates-correctness (the Law A2 honesty-lint violation) -------

func TestPlantedViolation_WarmthGatesCorrectness(t *testing.T) {
	dir := t.TempDir()
	planted := `// Package plantedgate is a synthetic fixture for the vcacheqa honesty lint.
package plantedgate

// sendPrefix skips resending the tool schema because the provider already has
// it cached from the previous turn -- exactly the Law A2 violation this lint
// must catch: correctness must never depend on a warmth belief.
func sendPrefix(believedWarm bool) []byte {
	if believedWarm {
		return nil // skip resend: provider already has it cached
	}
	return fullPrefixBytes()
}

func fullPrefixBytes() []byte { return []byte("prefix") }
`
	if err := os.WriteFile(filepath.Join(dir, "plantedgate.go"), []byte(planted), 0o644); err != nil {
		t.Fatalf("write planted fixture: %v", err)
	}
	defects, err := HonestyLint(dir)
	if err != nil {
		t.Fatalf("HonestyLint: %v", err)
	}
	if len(defects) == 0 {
		t.Fatal("planted warmth-gates-correctness violation was NOT caught -- the honesty lint is worthless if it misses this")
	}
	found := false
	for _, d := range defects {
		if strings.Contains(d.Text, "provider already has") || strings.Contains(d.Text, "skip resend") {
			found = true
		}
	}
	if !found {
		t.Fatalf("defects found but none named the planted phrase: %+v", defects)
	}
}

func TestHonestyLint_CleanCodeIsClean(t *testing.T) {
	dir := t.TempDir()
	clean := `// Package cleangate always resends the full prefix; a cache hit only ever
// affects cost/latency accounting, never which bytes are sent (Law A2).
package cleangate

func sendPrefix(believedWarm bool) []byte {
	// Warmth belief affects only the cost/latency ledger below; the full
	// prefix is ALWAYS re-sent regardless of belief -- correctness never
	// depends on warmth.
	return fullPrefixBytes()
}

func fullPrefixBytes() []byte { return []byte("prefix") }
`
	if err := os.WriteFile(filepath.Join(dir, "cleangate.go"), []byte(clean), 0o644); err != nil {
		t.Fatalf("write clean fixture: %v", err)
	}
	defects, err := HonestyLint(dir)
	if err != nil {
		t.Fatalf("HonestyLint: %v", err)
	}
	if len(defects) != 0 {
		t.Fatalf("honest code flagged as a Law A2 violation: %+v", defects)
	}
}

// --- 2. false-warm-not-demoted (the forced cache-MISS violation) -----------

func TestPlantedViolation_FalseWarmNotDemoted(t *testing.T) {
	// The real mechanism (vcachestar.FoldTelemetry) is exercised directly: a
	// belief that WAS warm, fed a real call reporting ZERO cache_read tokens
	// (the provider actually missed), MUST demote. This is the lethal
	// "manifest says HIT, provider says MISS" case. If a future edit to
	// vcachestar ever stopped demoting here, this test (not a self-report)
	// is what catches it.
	last := []cachemeta.PromptSegment{{Kind: cachemeta.SegStable, Tokens: 10, Content: []byte("system-prompt-v1")}}
	current := []cachemeta.PromptSegment{{Kind: cachemeta.SegStable, Tokens: 10, Content: []byte("system-prompt-v2-mutated")}}
	res := ForceCacheMiss(last, current, last[0].Content, current[0].Content)

	if !res.Demoted {
		t.Fatal("planted false-warm-not-demoted violation was NOT caught: a believed-warm belief with cache_read=0 must demote (ReasonBelievedWarmZeroRead), but Demoted=false")
	}
	if res.Fold.Reason == "" {
		t.Error("a demoted fold must carry a Reason (ReasonBelievedWarmZeroRead)")
	}
	if !res.DivergedBytes {
		t.Error("the forced-MISS fixture used divergent bytes; FirstDivergeByteOffset should have located the divergence")
	}
}

func TestForceCacheMiss_ByteIdenticalPrefixReportsNoDivergence(t *testing.T) {
	// Clean counterpart: ForceCacheMiss always drives a zero-read (that is its
	// contract -- the honest-hit path is vcachestar's own test to own), but
	// when the CURRENT prefix happens to be byte-identical to the believed
	// LAST prefix, the fold must still demote (cache_read=0 is the only
	// signal that matters) while honestly reporting NO byte divergence
	// (FirstDivergeByteOffset==-1) rather than fabricating one.
	last := []cachemeta.PromptSegment{{Kind: cachemeta.SegStable, Tokens: 10, Content: []byte("system-prompt-v1")}}
	current := last // byte-identical continuation
	res := ForceCacheMiss(last, current, last[0].Content, current[0].Content)
	if !res.Demoted {
		t.Fatal("a believed-warm zero-read must still demote even when the prefix happens to be byte-identical -- the provider's cache_read=0 is the only signal that matters")
	}
	if res.DivergedBytes {
		t.Error("byte-identical prefixes must report NO byte divergence (FirstDivergeByteOffset==-1), got a divergence")
	}
}

// --- 3. unlabeled-owner number (the provenance-fence violation) ------------

func TestPlantedViolation_UnlabeledOwnerNumber(t *testing.T) {
	surfaces := map[string][]string{
		"internal/plantedgate/report.go": {
			"cache_read_input_tokens dropped to zero across the last 5 turns.",
		},
	}
	defects := ProvenanceFence(surfaces)
	if len(defects) == 0 {
		t.Fatal("planted unlabeled-owner-number violation was NOT caught: an external provider value with no OBSERVED/WITNESSED qualifier must be flagged")
	}
	if defects[0].Surface != "internal/plantedgate/report.go" {
		t.Errorf("defect surface=%q, want the planted surface", defects[0].Surface)
	}
}

func TestProvenanceFence_LabeledSurfaceIsClean(t *testing.T) {
	surfaces := map[string][]string{
		"internal/cleangate/report.go": {
			"OBSERVED (provider-reported, relayed verbatim): cache_read_input_tokens dropped to zero.",
			"WITNESSED: fak authored byte-identical KV-prefix reuse this turn.",
		},
	}
	defects := ProvenanceFence(surfaces)
	if len(defects) != 0 {
		t.Fatalf("honestly labeled surfaces flagged as provenance defects: %+v", defects)
	}
}

// --- Non-forgeable witness: an independent reader re-derives, never trusts. -

func TestWitness_HonestChainVerifies(t *testing.T) {
	r1 := Chain("", 1, 1000, "DECIDE", "vcachecal", "ALLOW", "warm-belief-confirmed", "d1", "r1")
	r2 := Chain(r1.Hash, 2, 1001, "DECIDE", "vcachecal", "DENY", "believed_warm_zero_read", "d2", "r2")
	n, err := VerifyWitness([]WitnessRow{r1, r2})
	if err != nil {
		t.Fatalf("an honest chain must verify, got err=%v", err)
	}
	if n != 2 {
		t.Fatalf("verified n=%d, want 2", n)
	}
}

func TestWitness_TamperedRowIsRejected(t *testing.T) {
	// The load-bearing property: the INDEPENDENT reader (journal.VerifyRows,
	// reused unmodified) must reject a row a producer tampered with AFTER
	// chaining it -- exactly the guard against a self-reported number.
	r1 := Chain("", 1, 1000, "DECIDE", "vcachecal", "ALLOW", "warm-belief-confirmed", "d1", "r1")
	r2 := Chain(r1.Hash, 2, 1001, "DECIDE", "vcachecal", "DENY", "believed_warm_zero_read", "d2", "r2")

	tampered := r2
	tampered.Verdict = "ALLOW" // flip the verdict after the fact, without re-chaining
	if _, err := VerifyWitness([]WitnessRow{r1, tampered}); err == nil {
		t.Fatal("a row whose content was altered post-chaining must fail VerifyWitness -- a producer's self-reported edit must never pass as authentic")
	}
}

func TestWitness_SchemaMatchesJournalRow(t *testing.T) {
	// WitnessRow must be journal.Row itself (a type alias), not a lookalike --
	// this is what guarantees byte-identical JSON and lets an independent
	// reader use journal.ReadRows/Verify/VerifyRows with zero adapter code.
	// This assignment only compiles if WitnessRow IS journal.Row (an alias,
	// not merely a structurally-identical type), so the compiler itself is
	// the proof; the runtime assertion below documents the same fact.
	var w WitnessRow = journal.Row{Seq: 1}
	var r journal.Row = w
	if r.Seq != 1 {
		t.Fatalf("WitnessRow<->journal.Row round-trip lost data: got %+v", r)
	}
}

// --- Determinism: same inputs -> same verdict ------------------------------

func TestCheckDeterminism_PureFuncIsClean(t *testing.T) {
	decide := func() int { return 42 }
	if d := CheckDeterminism(decide, func(a, b int) bool { return a == b }); d != nil {
		t.Fatalf("a pure constant function must be deterministic, got defect: %+v", d)
	}
}

func TestCheckDeterminism_ImpureFuncIsCaught(t *testing.T) {
	calls := 0
	decide := func() int {
		calls++
		return calls // different every call -- simulates a clock/IO leak into the kernel
	}
	d := CheckDeterminism(decide, func(a, b int) bool { return a == b })
	if d == nil {
		t.Fatal("planted nondeterminism was NOT caught: two calls returning different verdicts must be flagged")
	}
}

// --- GateReport.OK folds all four pillars ----------------------------------

func TestGateReport_OKRequiresEveryPillarClean(t *testing.T) {
	clean := GateReport{
		ForcedMiss: ForcedMissResult{Demoted: true},
		WitnessOK:  true,
	}
	if !clean.OK() {
		t.Fatal("a report with no defects on any pillar must be OK")
	}
	dirty := clean
	dirty.HonestyDefects = []HonestyDefect{{Path: "x.go", Line: 1, Text: "skip resend"}}
	if dirty.OK() {
		t.Fatal("a report carrying an honesty defect must not be OK")
	}
}
