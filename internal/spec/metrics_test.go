package spec

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// TestAcceptanceMeterPureAccumulator pins the derived metrics, the clamping, and the
// zero-division guards of the pure accumulator (no model in the loop).
func TestAcceptanceMeterPureAccumulator(t *testing.T) {
	var m AcceptanceMeter

	// Empty meter: every derived metric is 0, no division by zero.
	if got := m.AcceptanceRate(); got != 0 {
		t.Fatalf("empty AcceptanceRate = %v, want 0", got)
	}
	if got := m.EffectiveTokensPerVerify(); got != 0 {
		t.Fatalf("empty EffectiveTokensPerVerify = %v, want 0", got)
	}

	// Two rounds: 4 drafted/3 accepted/4 advanced, then 4 drafted/1 accepted/2 advanced.
	m.Observe(4, 3, 4)
	m.Observe(4, 1, 2)
	if m.Rounds() != 2 || m.Drafted() != 8 || m.Accepted() != 4 || m.Advanced() != 6 {
		t.Fatalf("counts = rounds %d drafted %d accepted %d advanced %d, want 2/8/4/6",
			m.Rounds(), m.Drafted(), m.Accepted(), m.Advanced())
	}
	if got, want := m.AcceptanceRate(), 4.0/8.0; got != want {
		t.Fatalf("AcceptanceRate = %v, want %v", got, want)
	}
	if got, want := m.EffectiveTokensPerVerify(), 6.0/2.0; got != want {
		t.Fatalf("EffectiveTokensPerVerify = %v, want %v", got, want)
	}

	// Snapshot must agree with the live accessors (derived fields recomputed, never stale).
	s := m.Snapshot()
	if s.Rounds != m.Rounds() || s.Drafted != m.Drafted() || s.Accepted != m.Accepted() ||
		s.Advanced != m.Advanced() || s.AcceptanceRate != m.AcceptanceRate() ||
		s.EffectiveTokensPerVerify != m.EffectiveTokensPerVerify() {
		t.Fatalf("Snapshot %+v disagrees with the live meter", s)
	}

	// Clamping: a miscounting caller cannot push the rate above 1 or below 0.
	var c AcceptanceMeter
	c.Observe(-5, 99, -3) // drafted<0 → 0, accepted capped at drafted, advanced<0 → 0
	if c.Drafted() != 0 || c.Accepted() != 0 || c.Advanced() != 0 {
		t.Fatalf("negative-clamp counts = %d/%d/%d, want 0/0/0", c.Drafted(), c.Accepted(), c.Advanced())
	}
	c.Observe(3, 10, 4) // accepted (10) capped at drafted (3)
	if c.Accepted() != 3 {
		t.Fatalf("accepted not capped at drafted: got %d, want 3", c.Accepted())
	}
	if r := c.AcceptanceRate(); r < 0 || r > 1 {
		t.Fatalf("AcceptanceRate %v escaped [0,1] after a miscounting Observe", r)
	}
}

// TestSpeculativeGreedyMeteredTracksAcceptance proves the meter faithfully mirrors the
// decode's OWN counts on a real (CPU synthetic) speculative decode, that wiring the meter
// does not change the lossless output, and that the acceptance metric discriminates a
// full-accept drafter from an adversarial one (a non-vacuous metric).
func TestSpeculativeGreedyMeteredTracksAcceptance(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	abi.ResetForTest()
	defer abi.ResetForTest()

	sink := Install()
	if sink == nil {
		t.Fatal("Install returned nil with the lane enabled")
	}

	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("acceptance rate metrics for speculative decoding"))
	const N, K = 24, 4
	want := greedyDecode(target, prompt, N)

	// (a) Self-draft: the TARGET model drafts for itself (a separate session of the same
	//     deterministic weights), so every draft equals the target's argmax → FULL accept.
	var selfMeter AcceptanceMeter
	gotSelf, draftedSelf, acceptedSelf, _ := SpeculativeGreedyMetered(
		context.Background(), sink, target.NewSession(), prompt, N, K,
		newModelDrafter(target, prompt), &selfMeter)
	assertEqualTokens(t, "self-draft", gotSelf, want)

	// The meter mirrors the decode's own returned totals exactly.
	if selfMeter.Drafted() != draftedSelf || selfMeter.Accepted() != acceptedSelf {
		t.Fatalf("self-draft meter (%d drafted, %d accepted) != returned totals (%d, %d)",
			selfMeter.Drafted(), selfMeter.Accepted(), draftedSelf, acceptedSelf)
	}
	if selfMeter.Advanced() != len(gotSelf) {
		t.Fatalf("self-draft Advanced %d != real tokens committed %d", selfMeter.Advanced(), len(gotSelf))
	}
	if selfMeter.Rounds() < 1 {
		t.Fatalf("self-draft observed %d rounds, want >= 1", selfMeter.Rounds())
	}
	if got, want := selfMeter.AcceptanceRate(), float64(acceptedSelf)/float64(draftedSelf); got != want {
		t.Fatalf("self-draft AcceptanceRate %v != accepted/drafted %v", got, want)
	}
	if got, want := selfMeter.EffectiveTokensPerVerify(), float64(selfMeter.Advanced())/float64(selfMeter.Rounds()); got != want {
		t.Fatalf("self-draft EffectiveTokensPerVerify %v != advanced/rounds %v", got, want)
	}
	// Full accept ⇒ rate exactly 1.0 and a high effective E (well above a plain decode's 1).
	if selfMeter.AcceptanceRate() != 1.0 {
		t.Fatalf("self-draft should fully accept: AcceptanceRate = %v, want 1.0", selfMeter.AcceptanceRate())
	}
	if selfMeter.EffectiveTokensPerVerify() <= 2.0 {
		t.Fatalf("self-draft effective E = %v, want > 2.0 (a full-accept speedup)", selfMeter.EffectiveTokensPerVerify())
	}

	// (b) Adversarial draft: forces rejections, so the acceptance rate and effective E
	//     must both fall below the full-accept case — the metric actually moves.
	var advMeter AcceptanceMeter
	gotAdv, _, _, rolledAdv := SpeculativeGreedyMetered(
		context.Background(), sink, target.NewSession(), prompt, N, K, &advDrafter{}, &advMeter)
	assertEqualTokens(t, "adversarial-draft", gotAdv, want) // still lossless
	if rolledAdv == 0 {
		t.Fatal("VACUOUS: adversarial draft caused 0 rollbacks — acceptance discrimination untested")
	}
	if advMeter.AcceptanceRate() >= selfMeter.AcceptanceRate() {
		t.Fatalf("adversarial AcceptanceRate %v not below self-draft %v — metric did not discriminate",
			advMeter.AcceptanceRate(), selfMeter.AcceptanceRate())
	}
	if advMeter.EffectiveTokensPerVerify() >= selfMeter.EffectiveTokensPerVerify() {
		t.Fatalf("adversarial effective E %v not below self-draft %v — speedup proxy did not discriminate",
			advMeter.EffectiveTokensPerVerify(), selfMeter.EffectiveTokensPerVerify())
	}

	// The plain SpeculativeGreedy wrapper (nil meter) must produce the identical output —
	// wiring the meter changed nothing on the decode path.
	gotNil, _, _, _ := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K, &advDrafter{})
	assertEqualTokens(t, "nil-meter-wrapper", gotNil, want)
}
