package modelroute

// Telemetry-to-learned routing optimization engine (#600, epic #595).
//
// THE GAP IT FILLS. SOTA routers (RouteLLM, NotDiamond) fit a black-box
// predictor directly from telemetry feedback and silently mutate routes.
// fak's differentiator is learning routing improvements PER-ASPECT under an
// AUDITABLE, HUMAN-GATED policy:
//
//   - An offline optimization run consumes recorded telemetry (OutcomeJournal
//     or []OutcomeRecord) and a base Manifest.
//   - Evaluates rule alternatives against a measured objective (e.g. maximize
//     quality under cost/latency constraints or a weighted objective).
//   - Evaluates on a held-out split (train vs evaluation/test split) to prevent
//     overfitting to historical runs.
//   - When a candidate rule change strictly improves the objective on the
//     held-out split, it produces a reviewable RuleProposal and ManifestProposal:
//     a valid JSON diff of the manifest, objective score deltas (before vs after),
//     and review-gate metadata (keep=false default, human-review required).
//
// HONESTY AND SAFETY INVARIANTS:
//   - Never silently auto-apply: the review gate defaults to keep=false.
//     Apply() fails closed unless explicitly approved by an operator.
//   - Generalization discipline: improvements must hold on the held-out split,
//     not just the training data.
//   - Transparent diff: changes are rendered as structured and formatted JSON
//     diffs that can be audited before adoption.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// Objective specifies the optimization target and optional operational constraints
// for evaluating routing policies.
type Objective struct {
	// QualityWeight is the weight given to mean answer quality (default 1.0).
	QualityWeight float64 `json:"quality_weight"`
	// CostWeight is the penalty weight applied to mean cost in dollars (default 0.0).
	CostWeight float64 `json:"cost_weight,omitempty"`
	// LatencyWeight is the penalty weight applied to mean latency in seconds (default 0.0).
	LatencyWeight float64 `json:"latency_weight,omitempty"`

	// Constraints: hard operational limits; an evaluation violating any constraint is unfeasible.
	MaxMeanCost    float64       `json:"max_mean_cost,omitempty"`
	MaxMeanLatency time.Duration `json:"max_mean_latency_ns,omitempty"`
	MinMeanQuality float64       `json:"min_mean_quality,omitempty"`
}

func (o Objective) effectiveQualityWeight() float64 {
	if o.QualityWeight == 0 && o.CostWeight == 0 && o.LatencyWeight == 0 {
		return 1.0
	}
	return o.QualityWeight
}

// Score computes the scalar objective score for an evaluation.
func (o Objective) Score(eval EvaluationScore) float64 {
	qw := o.effectiveQualityWeight()
	latencySec := eval.MeanLatency.Seconds()
	return qw*eval.MeanQuality - o.CostWeight*eval.MeanCost - o.LatencyWeight*latencySec
}

// Feasible reports whether the evaluation satisfies all declared constraints.
func (o Objective) Feasible(eval EvaluationScore) bool {
	if eval.Count == 0 {
		return false
	}
	if o.MaxMeanCost > 0 && eval.MeanCost > o.MaxMeanCost {
		return false
	}
	if o.MaxMeanLatency > 0 && eval.MeanLatency > o.MaxMeanLatency {
		return false
	}
	if o.MinMeanQuality > 0 && eval.MeanQuality < o.MinMeanQuality {
		return false
	}
	return true
}

// EvaluationScore holds the rolled-up metrics and scalar score of a manifest
// or rule evaluated over telemetry records.
type EvaluationScore struct {
	Count       int           `json:"count"`
	MeanCost    float64       `json:"mean_cost"`
	MeanLatency time.Duration `json:"mean_latency_ns"`
	MeanQuality float64       `json:"mean_quality"`
	Score       float64       `json:"score"`
	Feasible    bool          `json:"feasible"`
}

// ScoreDelta records the objective score before and after a change.
type ScoreDelta struct {
	Before   EvaluationScore `json:"before"`
	After    EvaluationScore `json:"after"`
	Delta    float64         `json:"delta"`    // After.Score - Before.Score
	Improved bool            `json:"improved"` // Delta > 0 and After is feasible
}

func computeScoreDelta(before, after EvaluationScore) ScoreDelta {
	delta := after.Score - before.Score
	improved := after.Feasible && (delta > 1e-9 || (!before.Feasible && after.Feasible))
	return ScoreDelta{
		Before:   before,
		After:    after,
		Delta:    delta,
		Improved: improved,
	}
}

// RuleProposal describes an evidence-backed rule modification.
type RuleProposal struct {
	RuleName      string     `json:"rule_name"`
	Aspect        Aspect     `json:"aspect,omitempty"`
	CurrentModel  string     `json:"current_model"`
	ProposedModel string     `json:"proposed_model"`
	Current       Plan       `json:"current_target"`
	Proposed      Plan       `json:"proposed_target"`
	TrainScore    ScoreDelta `json:"train_score"`
	HeldOutScore  ScoreDelta `json:"held_out_score"`
	Reason        string     `json:"reason"`
}

// RuleDiff records a change to a single rule in the reviewable diff.
type RuleDiff struct {
	RuleName      string `json:"rule_name"`
	Aspect        Aspect `json:"aspect,omitempty"`
	CurrentModel  string `json:"current_model"`
	ProposedModel string `json:"proposed_model"`
	Current       Plan   `json:"current_target"`
	Proposed      Plan   `json:"proposed_target"`
	Reason        string `json:"reason,omitempty"`
}

// DefaultDiff records a change to the manifest default plan.
type DefaultDiff struct {
	CurrentModel  string `json:"current_model"`
	ProposedModel string `json:"proposed_model"`
	Current       Plan   `json:"current_target"`
	Proposed      Plan   `json:"proposed_target"`
	Reason        string `json:"reason,omitempty"`
}

// ManifestDiff is the structured, reviewable representation of differences
// between the base manifest and the proposed manifest.
type ManifestDiff struct {
	Schema          string       `json:"schema"`
	BaseVersion     string       `json:"base_version"`
	ProposedVersion string       `json:"proposed_version"`
	ChangedRules    []RuleDiff   `json:"changed_rules"`
	DefaultChange   *DefaultDiff `json:"default_change,omitempty"`
	Summary         string       `json:"summary"`
}

// ReviewAudit contains review and governance metadata. By construction,
// Keep defaults to false and RequiresHumanReview is true so no policy
// is ever silently auto-applied without human inspection.
type ReviewAudit struct {
	Keep                bool   `json:"keep"`                  // default false; never silently auto-apply
	RequiresHumanReview bool   `json:"requires_human_review"` // true; human must inspect and approve
	Status              string `json:"status"`                // PENDING_REVIEW | APPROVED | REJECTED
	Reason              string `json:"reason"`
}

// ManifestProposal is the generated proposal for an updated manifest.
type ManifestProposal struct {
	BaseManifest     Manifest       `json:"base_manifest"`
	ProposedManifest Manifest       `json:"proposed_manifest"`
	RuleProposals    []RuleProposal `json:"rule_proposals"`
	Diff             string         `json:"diff"` // reviewable JSON diff
	DiffStruct       *ManifestDiff  `json:"diff_struct,omitempty"`
	HeldOutScore     ScoreDelta     `json:"held_out_score"`
	TrainScore       ScoreDelta     `json:"train_score"`
	ReviewAudit      ReviewAudit    `json:"review_audit"`
}

// Improved reports whether the proposal strictly improved the objective on the held-out split.
func (p *ManifestProposal) Improved() bool {
	return p != nil && p.HeldOutScore.Improved
}

// RuleProposalByName finds the proposed change for the named rule.
func (p *ManifestProposal) RuleProposalByName(name string) (RuleProposal, bool) {
	if p == nil {
		return RuleProposal{}, false
	}
	for _, rp := range p.RuleProposals {
		if rp.RuleName == name {
			return rp, true
		}
	}
	return RuleProposal{}, false
}

// Apply verifies the review gate and returns the proposed manifest.
// It fails closed with an error if the review gate has not been explicitly
// approved (Keep==true).
func (p *ManifestProposal) Apply() (Manifest, error) {
	if p == nil {
		return Manifest{}, errors.New("modelroute: cannot apply nil proposal")
	}
	if !p.ReviewAudit.Keep {
		return Manifest{}, errors.New("modelroute: proposal cannot be silently applied: review gate requires human approval (keep=false)")
	}
	if err := p.ProposedManifest.Validate(); err != nil {
		return Manifest{}, fmt.Errorf("modelroute: proposed manifest validation failed: %w", err)
	}
	return p.ProposedManifest, nil
}

// ProposeOptions configures an offline optimization run.
type ProposeOptions struct {
	// BaseManifest is the starting routing policy to optimize.
	BaseManifest Manifest

	// Journal or Records provides recorded outcomes. If Journal is set and
	// Records is empty, Journal.Records() is used.
	Journal *OutcomeJournal
	Records []OutcomeRecord

	// TrainRecords and HeldOutRecords optionally provide an explicit pre-split
	// dataset. If empty, Records is partitioned using TrainRatio.
	TrainRecords   []OutcomeRecord
	HeldOutRecords []OutcomeRecord

	// TrainRatio is the proportion of records to use for candidate training (default 0.70).
	// The remainder is held out for independent verification.
	TrainRatio float64

	// AlternativeModels lists alternative models to evaluate for rules. If empty,
	// candidate models are discovered from observed telemetry records.
	AlternativeModels []string

	// AlternativeRules provides explicit alternative plans per rule name.
	AlternativeRules map[string][]Plan

	// Objective defines the scoring function and operational constraints.
	Objective Objective

	// MinSamples is the minimum number of samples required to consider an evaluation valid (default 1).
	MinSamples int
}

// SplitRecords deterministically splits recorded outcomes into training and held-out evaluation sets.
// It stratifies by (AspectRuleKey, Model) so both splits receive representative data for every observed model.
func SplitRecords(records []OutcomeRecord, trainRatio float64) ([]OutcomeRecord, []OutcomeRecord) {
	if trainRatio <= 0 || trainRatio >= 1.0 {
		trainRatio = 0.70
	}

	type groupKey struct {
		Aspect Aspect
		Rule   string
		Model  string
	}

	groups := make(map[groupKey][]OutcomeRecord)
	var groupOrder []groupKey

	for _, r := range records {
		gk := groupKey{
			Aspect: r.Key.Aspect,
			Rule:   r.Key.Rule,
			Model:  r.Model,
		}
		if _, exists := groups[gk]; !exists {
			groupOrder = append(groupOrder, gk)
		}
		groups[gk] = append(groups[gk], r)
	}

	var train, heldOut []OutcomeRecord
	for _, gk := range groupOrder {
		recs := groups[gk]
		n := len(recs)
		if n == 1 {
			train = append(train, recs[0])
			continue
		}
		nTrain := int(math.Round(float64(n) * trainRatio))
		if nTrain >= n {
			nTrain = n - 1
		}
		if nTrain <= 0 {
			nTrain = 1
		}
		train = append(train, recs[:nTrain]...)
		heldOut = append(heldOut, recs[nTrain:]...)
	}

	return train, heldOut
}

// recordTarget determines the rule name and chosen model when a record is routed under manifest m.
func recordTarget(m Manifest, rec OutcomeRecord) (matchedRule string, chosenModel string) {
	if rec.Subject.Aspect != "" || rec.Subject.Tool != "" || rec.Subject.Complexity != "" ||
		rec.Subject.Latency != "" || rec.Subject.InputTrigger != "" || len(rec.Subject.Labels) > 0 {
		dec := m.Route(rec.Subject)
		return dec.RuleName, dec.Plan.Primary()
	}

	if rec.Key.Rule != "" {
		for i := range m.Rules {
			if m.Rules[i].Name == rec.Key.Rule {
				return m.Rules[i].Name, m.Rules[i].Plan.Primary()
			}
		}
		// Named rule not found in manifest; falls back to default.
		return "", m.Default.Primary()
	}

	dec := m.Route(Subject{Aspect: rec.Key.Aspect})
	return dec.RuleName, dec.Plan.Primary()
}

// EvaluateManifest measures the performance of manifest m against a slice of OutcomeRecords.
// A record contributes only if it was served by the model that manifest m selects for that route.
func EvaluateManifest(m Manifest, records []OutcomeRecord, obj Objective) (EvaluationScore, error) {
	if len(records) == 0 {
		return EvaluationScore{}, nil
	}

	totalCount := 0
	var totalCost float64
	var totalLatency time.Duration
	var totalQuality float64

	for _, rec := range records {
		_, chosenModel := recordTarget(m, rec)
		recModel := rec.Model
		if recModel == "" || recModel != chosenModel {
			continue
		}

		totalCount++
		totalCost += rec.Outcome.Cost
		totalLatency += rec.Outcome.Latency
		totalQuality += rec.Outcome.Quality
	}

	if totalCount == 0 {
		return EvaluationScore{Count: 0, Feasible: false}, nil
	}

	meanCost := totalCost / float64(totalCount)
	meanLatency := time.Duration(int64(totalLatency) / int64(totalCount))
	meanQuality := totalQuality / float64(totalCount)

	eval := EvaluationScore{
		Count:       totalCount,
		MeanCost:    meanCost,
		MeanLatency: meanLatency,
		MeanQuality: meanQuality,
	}
	eval.Score = obj.Score(eval)
	eval.Feasible = obj.Feasible(eval)

	return eval, nil
}

// evaluateRuleRecords evaluates a specific rule's performance with a chosen model on a slice of records.
func evaluateRuleRecords(ruleName string, aspect Aspect, model string, records []OutcomeRecord, obj Objective) EvaluationScore {
	totalCount := 0
	var totalCost float64
	var totalLatency time.Duration
	var totalQuality float64

	for _, rec := range records {
		if rec.Key.Rule != ruleName {
			continue
		}
		if aspect != "" && rec.Key.Aspect != "" && rec.Key.Aspect != aspect {
			continue
		}
		if rec.Model != model {
			continue
		}

		totalCount++
		totalCost += rec.Outcome.Cost
		totalLatency += rec.Outcome.Latency
		totalQuality += rec.Outcome.Quality
	}

	if totalCount == 0 {
		return EvaluationScore{Count: 0, Feasible: false}
	}

	meanCost := totalCost / float64(totalCount)
	meanLatency := time.Duration(int64(totalLatency) / int64(totalCount))
	meanQuality := totalQuality / float64(totalCount)

	eval := EvaluationScore{
		Count:       totalCount,
		MeanCost:    meanCost,
		MeanLatency: meanLatency,
		MeanQuality: meanQuality,
	}
	eval.Score = obj.Score(eval)
	eval.Feasible = obj.Feasible(eval)
	return eval
}

// Propose runs an offline optimization over recorded outcomes and a base manifest.
// If any rule change strictly improves the objective on the held-out split (and
// satisfies all declared constraints), it returns a ManifestProposal.
func Propose(opts ProposeOptions) (*ManifestProposal, error) {
	if err := opts.BaseManifest.Validate(); err != nil {
		return nil, fmt.Errorf("modelroute: base manifest invalid: %w", err)
	}

	records := opts.Records
	if len(records) == 0 && opts.Journal != nil {
		records = opts.Journal.Records()
	}

	var train, heldOut []OutcomeRecord
	if len(opts.TrainRecords) > 0 && len(opts.HeldOutRecords) > 0 {
		train = opts.TrainRecords
		heldOut = opts.HeldOutRecords
	} else {
		if len(records) == 0 {
			return nil, errors.New("modelroute: no outcome records provided for proposal")
		}
		train, heldOut = SplitRecords(records, opts.TrainRatio)
	}

	if len(train) == 0 || len(heldOut) == 0 {
		return nil, errors.New("modelroute: insufficient records to form train and held-out splits")
	}

	minSamples := opts.MinSamples
	if minSamples <= 0 {
		minSamples = 1
	}

	baseTrainScore, err := EvaluateManifest(opts.BaseManifest, train, opts.Objective)
	if err != nil {
		return nil, fmt.Errorf("modelroute: evaluate base on train: %w", err)
	}
	baseHeldOutScore, err := EvaluateManifest(opts.BaseManifest, heldOut, opts.Objective)
	if err != nil {
		return nil, fmt.Errorf("modelroute: evaluate base on held-out: %w", err)
	}

	if baseHeldOutScore.Count == 0 {
		return nil, errors.New("modelroute: base manifest has no matching outcome records on held-out split")
	}

	type evaluatedImprovement struct {
		ruleIndex    int
		ruleName     string
		aspect       Aspect
		currentModel string
		candModel    string
		base         Plan
		alt          Plan
		trainScore   ScoreDelta
		heldOutScore ScoreDelta
	}

	var bestRuleImprovements []evaluatedImprovement

	// Evaluate alternatives for each rule in the base manifest.
	for rIdx, rule := range opts.BaseManifest.Rules {
		currentModel := rule.Plan.Primary()
		aspect := rule.Match.Aspect

		// Collect candidate models for this rule.
		candidateSet := make(map[string]bool)
		for _, m := range opts.AlternativeModels {
			if m != "" && m != currentModel {
				candidateSet[m] = true
			}
		}
		for _, r := range train {
			if r.Key.Rule == rule.Name && r.Model != "" && r.Model != currentModel {
				candidateSet[r.Model] = true
			}
		}

		var altModels []string
		for m := range candidateSet {
			altModels = append(altModels, m)
		}
		sort.Strings(altModels)

		var bestCand *evaluatedImprovement

		for _, candModel := range altModels {
			// Candidate plan
			alt := rule.Plan
			alt.Members = []Member{{Model: candModel, Role: "primary"}}
			alt.Reason = fmt.Sprintf("learned from feedback: changed model from %s to %s", currentModel, candModel)

			// Construct candidate manifest
			candManifest := opts.BaseManifest
			candManifest.Rules = make([]Rule, len(opts.BaseManifest.Rules))
			copy(candManifest.Rules, opts.BaseManifest.Rules)
			candManifest.Rules[rIdx].Plan = alt

			// Evaluate candidate manifest on train
			candTrainScore, err := EvaluateManifest(candManifest, train, opts.Objective)
			if err != nil || !candTrainScore.Feasible {
				continue
			}

			// Rule-specific train score
			ruleTrainBefore := evaluateRuleRecords(rule.Name, aspect, currentModel, train, opts.Objective)
			ruleTrainAfter := evaluateRuleRecords(rule.Name, aspect, candModel, train, opts.Objective)
			if ruleTrainAfter.Count < minSamples {
				continue
			}
			ruleTrainDelta := computeScoreDelta(ruleTrainBefore, ruleTrainAfter)
			if !ruleTrainDelta.Improved {
				continue
			}

			// Now evaluate candidate manifest on held-out split
			candHeldOutScore, err := EvaluateManifest(candManifest, heldOut, opts.Objective)
			if err != nil || !candHeldOutScore.Feasible {
				continue
			}

			// Rule-specific held-out score
			ruleHeldOutBefore := evaluateRuleRecords(rule.Name, aspect, currentModel, heldOut, opts.Objective)
			ruleHeldOutAfter := evaluateRuleRecords(rule.Name, aspect, candModel, heldOut, opts.Objective)
			if ruleHeldOutAfter.Count < minSamples {
				continue
			}
			ruleHeldOutDelta := computeScoreDelta(ruleHeldOutBefore, ruleHeldOutAfter)
			if !ruleHeldOutDelta.Improved {
				continue
			}

			// Must improve overall held-out score as well
			manifestHeldOutDelta := computeScoreDelta(baseHeldOutScore, candHeldOutScore)
			if !manifestHeldOutDelta.Improved {
				continue
			}

			ci := evaluatedImprovement{
				ruleIndex:    rIdx,
				ruleName:     rule.Name,
				aspect:       aspect,
				currentModel: currentModel,
				candModel:    candModel,
				base:         rule.Plan,
				alt:          alt,
				trainScore:   ruleTrainDelta,
				heldOutScore: ruleHeldOutDelta,
			}

			if bestCand == nil || ci.heldOutScore.Delta > bestCand.heldOutScore.Delta {
				candCopy := ci
				bestCand = &candCopy
			}
		}

		if bestCand != nil {
			bestRuleImprovements = append(bestRuleImprovements, *bestCand)
		}
	}

	if len(bestRuleImprovements) == 0 {
		// No rule change improved on the held-out split.
		return nil, nil
	}

	// Build the proposed manifest with all validated rule improvements.
	proposedManifest := opts.BaseManifest
	proposedManifest.Rules = make([]Rule, len(opts.BaseManifest.Rules))
	copy(proposedManifest.Rules, opts.BaseManifest.Rules)

	var ruleProposals []RuleProposal
	var ruleDiffs []RuleDiff

	for _, imp := range bestRuleImprovements {
		proposedManifest.Rules[imp.ruleIndex].Plan = imp.alt

		rp := RuleProposal{
			RuleName:      imp.ruleName,
			Aspect:        imp.aspect,
			CurrentModel:  imp.currentModel,
			ProposedModel: imp.candModel,
			Current:       imp.base,
			Proposed:      imp.alt,
			TrainScore:    imp.trainScore,
			HeldOutScore:  imp.heldOutScore,
			Reason:        fmt.Sprintf("held-out score improved by +%.4f (quality: %.2f->%.2f, cost: $%.3f->$%.3f)", imp.heldOutScore.Delta, imp.heldOutScore.Before.MeanQuality, imp.heldOutScore.After.MeanQuality, imp.heldOutScore.Before.MeanCost, imp.heldOutScore.After.MeanCost),
		}
		ruleProposals = append(ruleProposals, rp)

		ruleDiffs = append(ruleDiffs, RuleDiff{
			RuleName:      imp.ruleName,
			Aspect:        imp.aspect,
			CurrentModel:  imp.currentModel,
			ProposedModel: imp.candModel,
			Current:       imp.base,
			Proposed:      imp.alt,
			Reason:        rp.Reason,
		})
	}

	if err := proposedManifest.Validate(); err != nil {
		return nil, fmt.Errorf("modelroute: proposed manifest validation: %w", err)
	}

	finalTrainScore, err := EvaluateManifest(proposedManifest, train, opts.Objective)
	if err != nil {
		return nil, fmt.Errorf("modelroute: evaluate proposed manifest on train: %w", err)
	}
	finalHeldOutScore, err := EvaluateManifest(proposedManifest, heldOut, opts.Objective)
	if err != nil {
		return nil, fmt.Errorf("modelroute: evaluate proposed manifest on held-out: %w", err)
	}

	manifestHeldOutDelta := computeScoreDelta(baseHeldOutScore, finalHeldOutScore)
	if !manifestHeldOutDelta.Improved {
		return nil, nil
	}
	manifestTrainDelta := computeScoreDelta(baseTrainScore, finalTrainScore)

	summary := fmt.Sprintf("%d rule(s) improved: held-out score %.4f -> %.4f (delta +%.4f)",
		len(ruleProposals), baseHeldOutScore.Score, finalHeldOutScore.Score, manifestHeldOutDelta.Delta)

	diffStruct := ManifestDiff{
		Schema:          "fak-modelroute-diff/v1",
		BaseVersion:     opts.BaseManifest.Version,
		ProposedVersion: proposedManifest.Version,
		ChangedRules:    ruleDiffs,
		Summary:         summary,
	}

	diffBytes, err := json.MarshalIndent(diffStruct, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("modelroute: marshal manifest diff: %w", err)
	}

	proposal := &ManifestProposal{
		BaseManifest:     opts.BaseManifest,
		ProposedManifest: proposedManifest,
		RuleProposals:    ruleProposals,
		Diff:             string(diffBytes),
		DiffStruct:       &diffStruct,
		HeldOutScore:     manifestHeldOutDelta,
		TrainScore:       manifestTrainDelta,
		ReviewAudit: ReviewAudit{
			Keep:                false,
			RequiresHumanReview: true,
			Status:              "PENDING_REVIEW",
			Reason:              summary,
		},
	}

	return proposal, nil
}

// ProposeManifest is an alias for Propose.
func ProposeManifest(opts ProposeOptions) (*ManifestProposal, error) {
	return Propose(opts)
}
