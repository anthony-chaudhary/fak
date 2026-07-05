package sessionobs

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// TestNetTrueGoldenSessions is the issue #1159 acceptance: two golden sessions --
// one reuse-favorable, one intervention-heavy -- produce the two expected verdicts.
//
//   - reuse-favorable: a session that SHIPPED, where fak's local reuse (WITNESSED
//     radix/vDSO) saved materially more than its interventions added -> HELPED.
//   - intervention-heavy: a session where fak's transforms/quarantines ADDED far more
//     tokens than its reuse saved -> HURT, even though the session asserted value.
func TestNetTrueGoldenSessions(t *testing.T) {
	reuseFavorable := Record{
		SessionID: "reuse-favorable", AssistantTurns: 14, ToolCalls: 28, OutputTokens: 4000,
		Outcome: OutcomeShipped, Signals: Signals{Commits: 1, GoalEvents: 1},
	}
	reuseMed := Mediation{
		TokensAdded: 250, TokensSaved: 6000, MediationNanos: 1_200_000,
		SavedProvenance: ProvWitnessed, // fak's own kernel realized the reuse locally
	}
	got := NetTrue(reuseFavorable, reuseMed)
	if got.Verdict != "HELPED" {
		t.Fatalf("reuse-favorable session should be HELPED, got %q (%s)", got.Verdict, got.Detail)
	}
	if got.Provenance != "WITNESSED" {
		t.Errorf("a saving driven by fak's own local reuse must be WITNESSED, got %q", got.Provenance)
	}
	if got.NetTokens != 5750 {
		t.Errorf("net tokens should be saved-added = 5750, got %d", got.NetTokens)
	}

	interventionHeavy := Record{
		SessionID: "intervention-heavy", AssistantTurns: 11, ToolCalls: 22, OutputTokens: 1800,
		Outcome: OutcomeClaimed, Signals: Signals{ToolErrors: 3, GuardRefusals: 2},
	}
	heavyMed := Mediation{
		TokensAdded: 7000, TokensSaved: 400, MediationNanos: 9_000_000,
		SavedProvenance: ProvWitnessed,
	}
	got = NetTrue(interventionHeavy, heavyMed)
	if got.Verdict != "HURT" {
		t.Fatalf("intervention-heavy session should be HURT, got %q (%s)", got.Verdict, got.Detail)
	}
	if got.NetTokens != -6600 {
		t.Errorf("net tokens should be saved-added = -6600, got %d", got.NetTokens)
	}
	// The verdict must hold even though the session asserted value (Claimed): a costly
	// mediation that returned nothing is a HURT regardless of the outcome label.
	if !strings.Contains(got.Detail, "HURT") {
		t.Errorf("detail should restate the verdict, got %q", got.Detail)
	}
}

// TestNetTrueWashIsTheHonestMiddle: a net move inside the band is neither help nor hurt.
func TestNetTrueWashIsTheHonestMiddle(t *testing.T) {
	rec := Record{SessionID: "w", AssistantTurns: 9, OutputTokens: 5000, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}}
	med := Mediation{TokensAdded: 500, TokensSaved: 600, MediationNanos: 800_000, SavedProvenance: ProvWitnessed}
	row := NetTrue(rec, med)
	// throughput=6100 -> band=305; net=+100 < band -> WASH.
	if row.Verdict != "WASH" {
		t.Fatalf("a net move inside the band should be WASH, got %q (band=%d net=%d)", row.Verdict, row.Band, row.NetTokens)
	}
}

// TestNetTrueStalledSessionIsHurt: a Stopped session that cost fak tokens/ns with no
// net saving is HURT via the stall clause, even when the token net is inside the band.
func TestNetTrueStalledSessionIsHurt(t *testing.T) {
	rec := Record{SessionID: "stall", AssistantTurns: 7, OutputTokens: 800, Outcome: OutcomeStopped, Signals: Signals{StopEvents: 1}}
	med := Mediation{TokensAdded: 100, TokensSaved: 50, MediationNanos: 5_000_000, SavedProvenance: ProvWitnessed}
	row := NetTrue(rec, med)
	// throughput=950 -> band=256(floor); net=-50 is inside the band, but the stall clause fires.
	if row.Verdict != "HURT" {
		t.Fatalf("a stalled session that cost mediation should be HURT, got %q (band=%d net=%d)", row.Verdict, row.Band, row.NetTokens)
	}
	// And a HELPED can never be claimed on a session that stalled.
	helpAttempt := NetTrue(
		Record{SessionID: "stall2", OutputTokens: 1000, Outcome: OutcomeStopped},
		Mediation{TokensAdded: 100, TokensSaved: 9000, MediationNanos: 1000, SavedProvenance: ProvWitnessed})
	if helpAttempt.Verdict == "HELPED" {
		t.Errorf("a stalled session must never read HELPED, got %q", helpAttempt.Verdict)
	}
}

// TestNetTrueProvenanceLabels exercises all three net-true provenance terms.
func TestNetTrueProvenanceLabels(t *testing.T) {
	// OBSERVED: a HELPED driven by a provider prompt-cache hit (relayed, not authored).
	observed := NetTrue(
		Record{SessionID: "obs", OutputTokens: 2000, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}},
		Mediation{TokensAdded: 100, TokensSaved: 4000, MediationNanos: 500_000, SavedProvenance: ProvObserved})
	if observed.Verdict != "HELPED" || observed.Provenance != "OBSERVED" {
		t.Errorf("provider-cache-driven help should be HELPED/OBSERVED, got %s/%s", observed.Verdict, observed.Provenance)
	}
	// WITNESSED: a cost-dominated row -- fak authored the tokens it added.
	witnessed := NetTrue(
		Record{SessionID: "wit", OutputTokens: 1000, Outcome: OutcomeClaimed},
		Mediation{TokensAdded: 3000, TokensSaved: 0, MediationNanos: 1000})
	if witnessed.Provenance != "WITNESSED" {
		t.Errorf("a cost-dominated row should be WITNESSED, got %s", witnessed.Provenance)
	}
	// MODELED: a session with no measured mediation is a projection, never a silent zero.
	modeled := NetTrue(Record{SessionID: "mod", OutputTokens: 1000, Outcome: OutcomeNoOp}, Mediation{})
	if modeled.Provenance != "MODELED" || modeled.Verdict != "WASH" {
		t.Errorf("no-mediation session should be WASH/MODELED, got %s/%s", modeled.Verdict, modeled.Provenance)
	}
}

// TestNetTrueBandFloor: a tiny session cannot flip on a handful of tokens -- the
// absolute minNetBand floor forces WASH below a real, scale-relative move.
func TestNetTrueBandFloor(t *testing.T) {
	row := NetTrue(
		Record{SessionID: "tiny", OutputTokens: 100, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}},
		Mediation{TokensAdded: 0, TokensSaved: 200, MediationNanos: 10_000, SavedProvenance: ProvWitnessed})
	if row.Band != minNetBand {
		t.Errorf("a tiny session's band should clamp to the floor %d, got %d", minNetBand, row.Band)
	}
	if row.Verdict != "WASH" {
		t.Errorf("a +200 net under the %d floor must stay WASH, got %q", minNetBand, row.Verdict)
	}
}

// TestNetTrueDeterministic: same (Record, Mediation) in -> byte-identical row out, so a
// verdict is a witness a third party can re-derive, not a one-run reading.
func TestNetTrueDeterministic(t *testing.T) {
	rec := Record{SessionID: "d", OutputTokens: 3000, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}}
	med := Mediation{TokensAdded: 300, TokensSaved: 5000, MediationNanos: 2_000_000, SavedProvenance: ProvWitnessed}
	a := NetTrue(rec, med)
	b := NetTrue(rec, med)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("NetTrue must be deterministic:\n a=%+v\n b=%+v", a, b)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if !bytes.Equal(ja, jb) {
		t.Fatal("NetTrue JSON must be byte-identical across runs")
	}
	if a.Schema != netTrueSchema {
		t.Errorf("row should be self-describing with schema %q, got %q", netTrueSchema, a.Schema)
	}
}

// TestNetTrueWarmCompactionShedIsDiscounted is the #2804 acceptance: a compaction
// saving must be valued at the cache-read marginal (0.1x) on a WARM fire and only kept
// at full input (1.0x) on an OBSERVED-COLD fire -- the same B1 basis the Track-2 report
// applies. The repro is the contrast: an IDENTICAL 6000-token shed reads HELPED when it
// was booked at 1.0x (the pre-fix defect) but is honestly WASH once the warm shed is
// re-valued, while the cold fire keeps the full-value HELPED.
func TestNetTrueWarmCompactionShedIsDiscounted(t *testing.T) {
	rec := Record{SessionID: "warm-fire", AssistantTurns: 12, OutputTokens: 4000,
		Outcome: OutcomeShipped, Signals: Signals{Commits: 1}}

	// WARM: the 6000 shed tokens were live provider cache-reads -> worth 0.1x, not 1.0x.
	warm := Mediation{
		TokensAdded: 250, TokensSaved: 6000, CompactionShedTokens: 6000, ColdFire: false,
		MediationNanos: 1_000_000, SavedProvenance: ProvObserved,
	}
	warmRow := NetTrue(rec, warm)
	// effectiveSaved = 6000 - (6000 - floor(6000*0.1)) = 600; net = 600 - 250 = 350.
	if warmRow.TokensSaved != 600 {
		t.Errorf("warm shed should re-value 6000 -> 600 (0.1x), got %d", warmRow.TokensSaved)
	}
	if warmRow.NetTokens != 350 {
		t.Errorf("warm net should be 600-250 = 350, got %d", warmRow.NetTokens)
	}
	if warmRow.Verdict != "WASH" {
		t.Fatalf("a warm compaction saving must not read HELPED after the 0.1x discount, got %q (band=%d net=%d)",
			warmRow.Verdict, warmRow.Band, warmRow.NetTokens)
	}

	// The pre-fix defect, made explicit: booking the SAME shed at 1.0x would have been a
	// clear HELPED. The discount is what turns the over-valued win into an honest WASH.
	grossNet := warm.TokensSaved - warm.TokensAdded // 5750, the un-netted number
	if grossNet < warmRow.Band {
		t.Fatalf("test is not exercising the defect: gross net %d should clear band %d (would be HELPED unfixed)",
			grossNet, warmRow.Band)
	}

	// COLD: same shed, but observed-cold -> the tokens were NOT cache-reads, keep 1.0x.
	cold := warm
	cold.ColdFire = true
	coldRow := NetTrue(rec, cold)
	if coldRow.TokensSaved != 6000 {
		t.Errorf("cold shed must keep full 1.0x basis, got %d (want 6000)", coldRow.TokensSaved)
	}
	if coldRow.NetTokens != 5750 {
		t.Errorf("cold net should be 6000-250 = 5750, got %d", coldRow.NetTokens)
	}
	if coldRow.Verdict != "HELPED" {
		t.Fatalf("a cold (full-basis) compaction saving above band should be HELPED, got %q", coldRow.Verdict)
	}

	// The warm detail must SAY it re-valued the shed, so the ledger is auditable, not silent.
	if !strings.Contains(warmRow.Detail, "re-valued") {
		t.Errorf("warm detail should disclose the B1 re-valuation, got %q", warmRow.Detail)
	}
}

// TestNetTrueCompactionBasisAgreesWithReport pins sessionobs' warm-fire basis to the
// SAME 0.1x the Track-2 report uses (internal/cachevaluereport/track2.go
// `providerCacheReadMultiplier`). sessionobs is tier-1 and imports nothing internal, so
// the two constants are copies; this test is the guard that they cannot silently drift
// -- the #2804 "sessionobs and cachevaluereport agree on the compaction net" acceptance.
func TestNetTrueCompactionBasisAgreesWithReport(t *testing.T) {
	const reportProviderCacheReadMultiplier = 0.1 // cachevaluereport/track2.go:46
	if compactionCacheReadMarginal != reportProviderCacheReadMultiplier {
		t.Fatalf("sessionobs compaction basis %.3f must match the report's cache-read marginal %.3f",
			compactionCacheReadMarginal, reportProviderCacheReadMultiplier)
	}
	// A pure local-reuse saving (no compaction shed) must be untouched by the basis -- the
	// discount only ever removes phantom warm-shed value, never real vDSO/radix reuse.
	local := Mediation{TokensAdded: 100, TokensSaved: 5000, MediationNanos: 1000, SavedProvenance: ProvWitnessed}
	if got := local.effectiveSaved(); got != 5000 {
		t.Errorf("pure local reuse must not be discounted, got %d (want 5000)", got)
	}
	// A mislabeled shed larger than the saving it belongs to clamps, never goes negative.
	over := Mediation{TokensSaved: 1000, CompactionShedTokens: 5000}
	if got := over.effectiveSaved(); got != 100 { // clamp shed->1000, 0.1x -> 100
		t.Errorf("an over-large shed should clamp to the saving, got %d (want 100)", got)
	}
}

// TestRenderNetTrueSmoke: the per-session terminal view renders the verdict + detail.
func TestRenderNetTrueSmoke(t *testing.T) {
	var buf bytes.Buffer
	row := NetTrue(
		Record{SessionID: "r", OutputTokens: 4000, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}},
		Mediation{TokensAdded: 250, TokensSaved: 6000, MediationNanos: 1_200_000, SavedProvenance: ProvWitnessed})
	RenderNetTrue(&buf, row)
	if buf.Len() == 0 {
		t.Fatal("RenderNetTrue produced no output")
	}
	if !bytes.Contains(buf.Bytes(), []byte("HELPED")) {
		t.Errorf("render should surface the verdict, got:\n%s", buf.String())
	}
}
