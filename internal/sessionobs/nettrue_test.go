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
