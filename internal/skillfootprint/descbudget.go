package skillfootprint

// descbudget.go — the resident `.claude/skills` DESCRIPTION byte budget and its
// two-sided ratchet (#5444, epic #3229).
//
// This is the userland twin of internal/mcpfootprint/descbudget.go, deliberately
// built to the same shape so a reader who knows one knows the other:
//
//   - GROWTH is refused (SKILL_DESC_BUDGET_EXCEEDED). Adding a skill, or fattening
//     an existing skill's frontmatter `description`, pushes the measured floor past
//     the committed ceiling and reds the gate. The ONLY way through is to raise
//     SkillDescriptionBudgetBytes in the SAME commit — the reviewable diff line
//     that binds the new resident tax to the change that caused it, instead of the
//     tax being discovered a month later as a 30% regression.
//
//   - A BANKED WIN is required (SKILL_DESC_BUDGET_STALE). Trimming the resident
//     descriptions well below the ceiling ALSO reds the gate, until the ceiling is
//     re-pinned down. Otherwise the recovered slack silently becomes headroom for
//     the next skill and the ratchet stops ratcheting. Same discipline
//     internal/pythongate applies to the tools/*.py baseline: it only ever tightens.
//
// DENOMINATION. The budget is in BYTES, not estimated tokens, because bytes are
// what `fak skill footprint` measures and what capindex actually holds — pricing it
// in tokens would insert a divisor between the gate and the scorecard and let the
// two drift. See the package doc for the provenance caveat: this is fak's model of
// the resident index, never a provider-billed count.

import "fmt"

// SkillDescriptionBudgetBytes is the committed ceiling on the SUM of resident
// `.claude/skills` frontmatter `description` bytes. It is the measured baseline
// pinned in docs/context-budget/skill-description-floor.md.
//
// Changing this number is the whole point: it is the reviewable justification for a
// change to the resident userland tax. Raise it only alongside the skill
// description that grew it; lower it whenever a trim banks a win. Pinned at the
// measured floor at the time of the #5444 pin (see the baseline doc for the exact
// skill count and the regeneration command).
const SkillDescriptionBudgetBytes = 31724

// SkillDescriptionRatchetSlackBytes is how far the measured floor may sit BELOW the
// budget before the gate demands the ceiling be re-pinned. It absorbs incidental
// churn — a reworded trigger sentence, a retitled skill — without nagging, while
// still forcing a real reduction to be banked into the constant. It is sized at
// roughly one heavy skill description (~2 kB, a shade over 4% of the floor), so
// retiring or trimming a whole skill can never pass unbanked.
const SkillDescriptionRatchetSlackBytes = 2000

// Gate refusal reasons — closed-vocabulary tokens registered in dos.toml
// [reasons.*], so `dos_check_reason` resolves each as a known refusal rather than
// UNCLASSIFIED free-text drift.
const (
	ReasonSkillDescBudgetExceeded = "SKILL_DESC_BUDGET_EXCEEDED"
	ReasonSkillDescBudgetStale    = "SKILL_DESC_BUDGET_STALE"
)

// BytesPerTokenEstimate is the house ~4-bytes/token divisor, the same one
// EstimateAnthropicTokens walks. It is named here so the verb and any future
// reporter share ONE divisor and the printed token figure can never drift from the
// gated byte figure.
const BytesPerTokenEstimate = 4

// ApproxTokens converts a resident byte floor to the house ESTIMATED token figure.
// It is an estimate, never a provider-billed count.
func ApproxTokens(bytes int) int { return bytes / BytesPerTokenEstimate }

// SkillDescBudgetError is a structured resident-floor refusal: which direction the
// floor moved, what was measured, and what the committed ceiling says. Callers
// branch on Reason rather than string-matching the message.
type SkillDescBudgetError struct {
	Reason     string // ReasonSkillDescBudgetExceeded | ReasonSkillDescBudgetStale
	Measured   int    // the measured resident description floor, in BYTES
	Budget     int    // the committed ceiling (SkillDescriptionBudgetBytes)
	SkillCount int    // how many skill cards were priced
}

func (e *SkillDescBudgetError) Error() string {
	switch e.Reason {
	case ReasonSkillDescBudgetExceeded:
		return fmt.Sprintf(
			"%s: the resident .claude/skills description floor grew to %d bytes (~%d est. tokens) across %d skills, over the "+
				"committed budget of %d (+%d). Every skill's frontmatter description is held resident, so this prose is a "+
				"per-session tax paid forever and it grows with every skill anyone adds. Justify it by raising "+
				"skillfootprint.SkillDescriptionBudgetBytes to %d in the same commit (and re-pin the baseline in "+
				"docs/context-budget/skill-description-floor.md), or trim the description — the skill is still invocable by "+
				"name, so the resident prose only has to say WHEN to load it.",
			e.Reason, e.Measured, ApproxTokens(e.Measured), e.SkillCount, e.Budget, e.Measured-e.Budget, e.Measured)
	case ReasonSkillDescBudgetStale:
		return fmt.Sprintf(
			"%s: the resident .claude/skills description floor fell to %d bytes (~%d est. tokens) across %d skills, %d below "+
				"the committed budget of %d — a trim was won but never banked. Re-pin "+
				"skillfootprint.SkillDescriptionBudgetBytes to %d (and the baseline in "+
				"docs/context-budget/skill-description-floor.md) so the ratchet tightens and the recovered slack cannot be "+
				"silently refilled by the next skill.",
			e.Reason, e.Measured, ApproxTokens(e.Measured), e.SkillCount, e.Budget-e.Measured, e.Budget, e.Measured)
	default:
		return fmt.Sprintf("%s: resident description floor %d bytes vs budget %d", e.Reason, e.Measured, e.Budget)
	}
}

// CheckDescriptions gates a folded resident floor against the committed budget. It
// returns nil when the measured floor sits inside the band
// [Budget-Slack, Budget], and a *SkillDescBudgetError naming the direction otherwise.
//
// The gate fails CLOSED on a broken measurement: an unreadable or empty skills tree
// folds to 0 bytes, which lands far under the ratchet floor and refuses as STALE
// rather than passing vacuously. A gate that greens on "I measured nothing" is
// worse than no gate — it is a gate that reports success for a broken probe.
func CheckDescriptions(f Floor) error {
	return checkDescriptionsAgainst(f, SkillDescriptionBudgetBytes, SkillDescriptionRatchetSlackBytes)
}

// checkDescriptionsAgainst is CheckDescriptions with the budget and slack injected,
// so a test can drive both refusal directions without editing the committed
// constant (and so the band edges can be pinned at small, legible numbers).
func checkDescriptionsAgainst(f Floor, budget, slack int) error {
	measured := f.DescFloor
	if measured > budget {
		return &SkillDescBudgetError{
			Reason:     ReasonSkillDescBudgetExceeded,
			Measured:   measured,
			Budget:     budget,
			SkillCount: f.SkillCount,
		}
	}
	if measured < budget-slack {
		return &SkillDescBudgetError{
			Reason:     ReasonSkillDescBudgetStale,
			Measured:   measured,
			Budget:     budget,
			SkillCount: f.SkillCount,
		}
	}
	return nil
}
