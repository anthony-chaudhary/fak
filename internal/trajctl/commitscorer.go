package trajctl

// commitscorer.go — issue #2536, spine step 3 of the trajectory-control epic
// (#2533): the first scorer, W3 with zero model calls, so the witness-rung
// doctrine is set from day one.
//
// The witnessed-commit progress scorer measures how much of an objective's
// declared plan is bound to a verified commit: value is the fraction of plan
// phases with at least one candidate commit that resolves to EvidenceVerified. A
// four-phase plan with two witnessed phases scores 0.5. The row is W3 (the
// commit either exists or it does not) and carries a commit EvidenceRef per
// progressed phase so a later audit can re-resolve it — and demote the row to W0
// if the SHA ever goes dangling (FoldVerified).

const (
	// CommitScorerMethod is the stable method id of the witnessed-commit progress
	// scorer. It keys the registry and travels in every row.
	CommitScorerMethod = "witnessed-commit-progress"
	// CommitScorerVersion is this implementation's version.
	CommitScorerVersion = "1"
)

// CommitProgressScorer scores an objective by the fraction of its plan phases
// bound to a verified commit. It is stateless and pure: the only host-dependent
// input is the window's injected resolver.
type CommitProgressScorer struct{}

// Method implements Scorer.
func (CommitProgressScorer) Method() string { return CommitScorerMethod }

// Version implements Scorer.
func (CommitProgressScorer) Version() string { return CommitScorerVersion }

// Score returns one W3 progress row for obj. A phase counts as progressed once
// any of its candidate commits resolves to EvidenceVerified; the first such
// commit is recorded as the phase's evidence pointer. An objective with no plan
// has nothing to score against and yields no row. A nil resolver verifies
// nothing, so an un-resolvable window scores 0 rather than crediting unverified
// work.
func (CommitProgressScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	if len(obj.Plan) == 0 {
		return nil
	}
	progressed := 0
	var evidence []EvidenceRef
	for _, phase := range obj.Plan {
		for _, sha := range win.PhaseCommits[phase.ID] {
			if sha == "" {
				continue
			}
			ref := EvidenceRef{Kind: "commit", Ref: sha, Detail: phase.ID}
			if win.Resolve == nil || win.Resolve(ref) != EvidenceVerified {
				continue
			}
			progressed++
			evidence = append(evidence, ref)
			break // one witnessed commit is enough to progress the phase
		}
	}
	value := float64(progressed) / float64(len(obj.Plan))
	return []ScoreRow{{
		ObjectiveID: obj.ID,
		Value:       value,
		Method:      CommitScorerMethod,
		Version:     CommitScorerVersion,
		Witness:     W3,
		Evidence:    evidence,
		UnixMillis:  win.UnixMillis,
	}}
}
