package agentdojo

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ASRSteward is the dynamic replacement for the static poison.json check: a
// single-invariant validator that runs the adaptive attack matrix through the
// full-stack defense and fires iff any attack achieves the attacker's goal (ASR >
// 0). Per the steward discipline it never blocks on its own opinion — it returns a
// violation only with an INDEPENDENTLY-REPRODUCIBLE witness (the winning attack,
// which any auditor can re-run through Defense.Run to confirm).
type ASRSteward struct {
	attacks []Attack
	newDef  func() *Defense
}

// NewASRSteward builds the steward over the EXPANDED battery (seed Matrix UNION its
// deterministic generative expansion) and the full-stack defense — the
// stronger-than-fixed regression gate the expander was built to feed (expand.go).
// Gating on ExpandedMatrix rather than the seed Matrix alone means the shipped
// steward fires the moment a regression lets ANY phrasing through, including the
// generated paraphrases that search the lexical gate's semantic blind spot; the full
// stack holds ASR at zero across the whole expanded set (IFC taints by provenance),
// so the abstain is unchanged on a healthy stack. Pass custom attacks/def for a
// tighter or broader battery.
func NewASRSteward() *ASRSteward {
	return &ASRSteward{attacks: ExpandedMatrix(), newDef: NewFullStack}
}

func (s *ASRSteward) Name() string { return "agentdojo-asr-zero" }

// Check runs the matrix and reports a violation iff the full-stack ASR exceeds
// zero. The witness names the first winning attack and its measured ASR.
func (s *ASRSteward) Check(ctx context.Context) (violated bool, witness string) {
	rep := s.newDef().Score(ctx, s.attacks)
	if rep.Succeeded == 0 {
		return false, ""
	}
	w := rep.Wins[0]
	return true, fmt.Sprintf("full-stack ASR=%.0f%% (%d/%d): %q (%s/%s) reached an attacker sink",
		rep.ASR*100, rep.Succeeded, rep.Total, w.Name, w.Vector, w.Adaptivity)
}

// Compile-time assertion that ASRSteward satisfies the frozen steward interface.
var _ abi.Steward = (*ASRSteward)(nil)
