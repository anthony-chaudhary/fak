package rsiloop

// skillcandidate_batch.go — the batch entry point a skill-creation pass calls to
// wire proposed skills through the #2872 promotion gate. Hermes creates a skill and
// adds it to the active set on the LLM's say-so; the fak entry point instead runs
// every proposed skill through PromoteSkill against its held fixture corpus, keeps
// only the ones that WITNESS a strict gain, and leaves a structured, per-decision
// revert trail (the curator ledger) for the rest — so a skill-creation pass can
// never silently promote an unmeasured skill.

// PromoteSkills runs a batch of proposed skills through the promotion gate against
// their held fixture corpora and journals every REVERT into the curator ledger,
// returning only the promoted set. It is the entry point a skill-creation pass
// calls: propose N skills, keep the ones that witness a gain, and record a
// structured ReasonUnwitnessed revert for the rest. A nil ledger skips journaling
// (the keep/revert decisions still stand — journaling is the audit trail, not the
// gate). Journaling stops and returns on the first ledger error, with the skills
// promoted so far, so a durable-write failure is surfaced, never swallowed.
func PromoteSkills(cands []SkillCandidate, l *CuratorLedger) ([]SkillPromotion, error) {
	promoted := make([]SkillPromotion, 0, len(cands))
	for _, c := range cands {
		p := PromoteSkill(c)
		if p.Promoted {
			promoted = append(promoted, p)
			continue
		}
		if l != nil {
			if _, err := p.Journal(l); err != nil {
				return promoted, err
			}
		}
	}
	return promoted, nil
}
