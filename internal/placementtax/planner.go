package placementtax

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"
)

const (
	// SelectionReceiptSchema is the stable wire identity of a placement decision.
	SelectionReceiptSchema = "fak-native-placement-selection/v1"
	// NativeEngine is the only product engine this planner may select.
	NativeEngine = "fak-native"
)

// ParallelismKind names the decomposition used by a placement candidate.
type ParallelismKind string

const (
	ParallelismTensor   ParallelismKind = "tensor"
	ParallelismPipeline ParallelismKind = "pipeline"
	ParallelismExpert   ParallelismKind = "expert"
	ParallelismData     ParallelismKind = "data"
	ParallelismSequence ParallelismKind = "sequence"
	ParallelismContext  ParallelismKind = "context"
	ParallelismHybrid   ParallelismKind = "hybrid"
)

// ParallelismStrategy keeps every decomposition axis explicit. Degrees are always
// at least one. A non-hybrid strategy may enlarge only its named axis; a hybrid
// enlarges at least two axes.
type ParallelismStrategy struct {
	Kind           ParallelismKind `json:"kind"`
	TensorDegree   int             `json:"tensorDegree"`
	PipelineStages int             `json:"pipelineStages"`
	ExpertDegree   int             `json:"expertDegree"`
	DataReplicas   int             `json:"dataReplicas"`
	SequenceDegree int             `json:"sequenceDegree"`
	ContextDegree  int             `json:"contextDegree"`
}

// TopologyIntent records the locality hierarchy assumed by the candidate. The
// actual nodes and links remain authoritative in Placement and Topology.
type TopologyIntent struct {
	Hierarchy []string `json:"hierarchy"`
	Rationale string   `json:"rationale"`
}

// SLOProjection is a caller-calibrated margin in the SLO's native unit. Positive
// values have headroom, zero is exactly at the boundary, and negative values miss.
type SLOProjection struct {
	Name       string     `json:"name"`
	Unit       string     `json:"unit"`
	Margin     float64    `json:"margin"`
	Modeled    bool       `json:"modeled"`
	Provenance Provenance `json:"provenance,omitempty"`
}

// SLOIdentity gives policy margin values one stable native unit.
type SLOIdentity struct {
	Name string `json:"name"`
	Unit string `json:"unit"`
}

// PlanCandidate binds a typed parallelism strategy to the existing placement-tax
// input. Provenance fields are conservative caller assertions; Analyze output is
// authoritative, and cross-domain provenance may be left empty for derivation.
type PlanCandidate struct {
	ID                    string              `json:"id"`
	Engine                string              `json:"engine"`
	Strategy              ParallelismStrategy `json:"strategy"`
	Placement             Placement           `json:"placement"`
	Topology              TopologyIntent      `json:"topology"`
	Rationale             string              `json:"rationale"`
	Provenance            Provenance          `json:"provenance"`
	CrossDomainProvenance Provenance          `json:"crossDomainProvenance,omitempty"`
	SLO                   SLOProjection       `json:"slo"`
}

// Dimension is one independent operator objective. Duration values are compared
// in seconds; all other values retain the units named by their constants.
type Dimension string

const (
	DimensionLatencySeconds       Dimension = "latency_seconds"
	DimensionThroughput           Dimension = "throughput"
	DimensionMonetaryUSD          Dimension = "monetary_usd"
	DimensionEnergyJoules         Dimension = "energy_joules"
	DimensionMemoryHeadroomBytes  Dimension = "memory_headroom_bytes"
	DimensionComputeHeadroomUnits Dimension = "compute_headroom_units"
	DimensionSLOMargin            Dimension = "slo_margin"
)

// Direction states whether an objective prefers smaller or larger values.
type Direction string

const (
	Minimize Direction = "minimize"
	Maximize Direction = "maximize"
)

// ConstraintOperator is a hard feasibility extension applied before Pareto
// comparison.
type ConstraintOperator string

const (
	AtMost  ConstraintOperator = "at_most"
	AtLeast ConstraintOperator = "at_least"
)

// Constraint is one explicit hard policy boundary.
type Constraint struct {
	Dimension Dimension          `json:"dimension"`
	Operator  ConstraintOperator `json:"operator"`
	Value     float64            `json:"value"`
}

// Objective participates in Pareto dominance without a scalar weight.
type Objective struct {
	Dimension Dimension `json:"dimension"`
	Direction Direction `json:"direction"`
}

// ResolutionMode controls when a frontier becomes one executable plan.
type ResolutionMode string

const (
	// ResolveUnique selects only a singleton Pareto frontier.
	ResolveUnique ResolutionMode = "unique_frontier"
	// ResolveLexicographic applies Objectives in their declared order.
	ResolveLexicographic ResolutionMode = "lexicographic"
)

// TieBreakMode controls equal lexicographic values. Refusing a tie is the default;
// candidate ID ordering is available only when the operator says so explicitly.
type TieBreakMode string

const (
	TieBreakRefuse      TieBreakMode = "refuse"
	TieBreakCandidateID TieBreakMode = "candidate_id"
)

// EstimatedCrossDomainPolicy controls uncalibrated cross-domain candidates.
type EstimatedCrossDomainPolicy string

const (
	RefuseEstimatedCrossDomain EstimatedCrossDomainPolicy = "refuse"
	AllowEstimatedCrossDomain  EstimatedCrossDomainPolicy = "include_labeled_estimated"
)

// OperatorPolicy contains only hard filters, independent objectives, and an
// explicit resolution rule. There are deliberately no weights or aggregate score.
type OperatorPolicy struct {
	Constraints          []Constraint               `json:"constraints,omitempty"`
	Objectives           []Objective                `json:"objectives"`
	SLO                  *SLOIdentity               `json:"slo,omitempty"`
	Resolution           ResolutionMode             `json:"resolution"`
	TieBreak             TieBreakMode               `json:"tieBreak"`
	EstimatedCrossDomain EstimatedCrossDomainPolicy `json:"estimatedCrossDomain"`
}

// PlannerInput is one matched workload, topology, reference, and candidate set.
type PlannerInput struct {
	Workload   Workload        `json:"workload"`
	Topology   Topology        `json:"topology"`
	Reference  Placement       `json:"reference"`
	Candidates []PlanCandidate `json:"candidates"`
	Policy     OperatorPolicy  `json:"policy"`
}

// CapacityHeadroom is the smallest absolute residual capacity among a
// candidate's allocated nodes.
type CapacityHeadroom struct {
	MemoryBytes  uint64  `json:"memoryBytes"`
	ComputeUnits float64 `json:"computeUnits"`
}

// PlanMetrics preserves every decision dimension instead of fusing unlike units.
type PlanMetrics struct {
	Latency        time.Duration    `json:"latency"`
	Cycle          time.Duration    `json:"cycle"`
	Throughput     float64          `json:"throughput"`
	ThroughputUnit string           `json:"throughputUnit"`
	MonetaryUSD    ModeledValue     `json:"monetaryUSD"`
	EnergyJoules   ModeledValue     `json:"energyJoules"`
	Capacity       CapacityHeadroom `json:"capacity"`
	SLO            SLOProjection    `json:"slo"`
}

// CandidateDecision is the policy state assigned to an alternative.
type CandidateDecision string

const (
	DecisionSelected  CandidateDecision = "selected"
	DecisionFrontier  CandidateDecision = "pareto_frontier"
	DecisionDominated CandidateDecision = "dominated"
	DecisionRejected  CandidateDecision = "rejected"
)

// AlternativeReceipt is the auditable record of one candidate.
type AlternativeReceipt struct {
	CandidateID           string              `json:"candidateID"`
	Engine                string              `json:"engine"`
	Strategy              ParallelismStrategy `json:"strategy"`
	Placement             Placement           `json:"placement"`
	Topology              TopologyIntent      `json:"topology"`
	Nodes                 []string            `json:"nodes"`
	Domains               []string            `json:"domains"`
	Links                 []string            `json:"links,omitempty"`
	Rationale             string              `json:"rationale"`
	Provenance            Provenance          `json:"provenance"`
	CrossDomainProvenance Provenance          `json:"crossDomainProvenance,omitempty"`
	CrossDomain           bool                `json:"crossDomain"`
	Estimated             bool                `json:"estimated"`
	ReferenceFeasible     bool                `json:"referenceFeasible"`
	Decision              CandidateDecision   `json:"decision"`
	Reasons               []string            `json:"reasons,omitempty"`
	Metrics               *PlanMetrics        `json:"metrics,omitempty"`
	SLO                   SLOProjection       `json:"slo"`
}

// ReceiptDisposition records whether the policy produced an executable plan.
type ReceiptDisposition string

const (
	DispositionSelected    ReceiptDisposition = "selected"
	DispositionNoSelection ReceiptDisposition = "no_selection"
)

// SelectionReceipt is versioned planner evidence. SelectionID is a content
// binding for future outcome correlation; it is not proof that a plan executed.
type SelectionReceipt struct {
	Schema                       string               `json:"schema"`
	SelectionID                  string               `json:"selectionID"`
	Engine                       string               `json:"engine"`
	WorkloadID                   string               `json:"workloadID"`
	Workload                     Workload             `json:"workload"`
	Topology                     Topology             `json:"topology"`
	Reference                    Placement            `json:"reference"`
	Policy                       OperatorPolicy       `json:"policy"`
	Disposition                  ReceiptDisposition   `json:"disposition"`
	Selected                     *AlternativeReceipt  `json:"selected,omitempty"`
	ParetoFrontier               []string             `json:"paretoFrontier"`
	Alternatives                 []AlternativeReceipt `json:"alternatives"`
	RejectedAlternatives         []AlternativeReceipt `json:"rejectedAlternatives,omitempty"`
	ContainsEstimatedCrossDomain bool                 `json:"containsEstimatedCrossDomain"`
	Fallback                     string               `json:"fallback"`
	NoSelectionReason            string               `json:"noSelectionReason,omitempty"`
}

// PlanResult carries analyzer reports and serializable selection evidence for
// audit, replay, and future actuation correlation.
type PlanResult struct {
	Reports map[string]Report
	Receipt SelectionReceipt
}

type evaluatedPlan struct {
	candidate PlanCandidate
	report    Report
	receipt   AlternativeReceipt
	eligible  bool
}

// Plan evaluates each placement with Analyze, applies hard policy filters, computes
// a deterministic Pareto frontier, and selects only when the declared resolution
// rule resolves it.
func Plan(in PlannerInput) (PlanResult, error) {
	if err := validatePolicy(in.Policy); err != nil {
		return PlanResult{}, err
	}
	if len(in.Candidates) == 0 {
		return PlanResult{}, fmt.Errorf("planner must declare at least one candidate")
	}

	candidates := append([]PlanCandidate(nil), in.Candidates...)
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].ID < candidates[j].ID })
	if err := validateSLOComparability(candidates, in.Policy); err != nil {
		return PlanResult{}, err
	}
	seen := make(map[string]bool, len(candidates))
	plans := make([]evaluatedPlan, 0, len(candidates))
	reports := make(map[string]Report, len(candidates))

	for _, candidate := range candidates {
		if err := validateCandidate(candidate, in.Topology); err != nil {
			return PlanResult{}, fmt.Errorf("candidate %q: %w", candidate.ID, err)
		}
		if seen[candidate.ID] {
			return PlanResult{}, fmt.Errorf("duplicate candidate id %q", candidate.ID)
		}
		seen[candidate.ID] = true

		report, err := Analyze(Comparison{
			Workload: in.Workload, Topology: in.Topology,
			Candidate: candidate.Placement, Reference: in.Reference,
		})
		if err != nil {
			return PlanResult{}, fmt.Errorf("candidate %q analysis: %w", candidate.ID, err)
		}
		reports[candidate.ID] = report
		crossDomain := spansDomains(candidate.Placement, in.Topology)
		effectiveProvenance := evaluationProvenance(report.Candidate)
		if effectiveProvenance == "" {
			// Analyze intentionally emits no ledger for infeasible alternatives.
			// Preserve a conservative estimate instead of dropping provenance or
			// laundering an unwitnessed measured declaration.
			effectiveProvenance = ProvenanceEstimated
		}
		if policyUsesDimension(in.Policy, DimensionSLOMargin) && candidate.SLO.Modeled {
			effectiveProvenance = mergeProvenance(effectiveProvenance, candidate.SLO.Provenance)
		}
		crossDomainProvenance := boundaryProvenance(report.Candidate, in.Topology, crossDomain)
		if report.Candidate.Feasibility.Feasible && candidate.Provenance == ProvenanceMeasured && effectiveProvenance != ProvenanceMeasured {
			return PlanResult{}, fmt.Errorf("candidate %q declares measured provenance but analyzed contributing costs are %s", candidate.ID, effectiveProvenance)
		}
		if report.Candidate.Feasibility.Feasible && candidate.CrossDomainProvenance == ProvenanceMeasured && crossDomainProvenance != ProvenanceMeasured {
			return PlanResult{}, fmt.Errorf("candidate %q declares measured cross-domain provenance but analyzed boundary evidence is %s", candidate.ID, crossDomainProvenance)
		}
		nodes, domains, links := placementTopologyIDs(candidate.Placement, in.Topology)
		alt := AlternativeReceipt{
			CandidateID: candidate.ID, Engine: candidate.Engine,
			Strategy: candidate.Strategy, Placement: clonePlacement(candidate.Placement), Topology: cloneTopologyIntent(candidate.Topology),
			Nodes: nodes, Domains: domains, Links: links,
			Rationale: candidate.Rationale, Provenance: effectiveProvenance,
			CrossDomainProvenance: crossDomainProvenance,
			CrossDomain:           crossDomain,
			Estimated: provenanceIncludesEstimate(effectiveProvenance) ||
				provenanceIncludesEstimate(crossDomainProvenance),
			ReferenceFeasible: report.Reference.Feasibility.Feasible,
			Decision:          DecisionRejected,
			SLO:               candidate.SLO,
		}
		plan := evaluatedPlan{candidate: candidate, report: report, receipt: alt}
		if !report.Candidate.Feasibility.Feasible {
			plan.receipt.Reasons = append(plan.receipt.Reasons, report.Candidate.Feasibility.Reasons...)
			plans = append(plans, plan)
			continue
		}
		metrics := candidateMetrics(report.Candidate, candidate.SLO)
		plan.receipt.Metrics = &metrics
		if crossDomain && provenanceIncludesEstimate(crossDomainProvenance) &&
			in.Policy.EstimatedCrossDomain == RefuseEstimatedCrossDomain {
			plan.receipt.Reasons = append(plan.receipt.Reasons, "estimated cross-domain calibration refused by operator policy")
			plans = append(plans, plan)
			continue
		}
		for _, constraint := range in.Policy.Constraints {
			value, ok := dimensionValue(metrics, constraint.Dimension)
			if !ok {
				plan.receipt.Reasons = append(plan.receipt.Reasons,
					fmt.Sprintf("constraint %s is unmodeled", constraint.Dimension))
				continue
			}
			if !constraintAllows(value, constraint) {
				plan.receipt.Reasons = append(plan.receipt.Reasons,
					fmt.Sprintf("constraint %s %s %.9g rejected value %.9g", constraint.Dimension, constraint.Operator, constraint.Value, value))
			}
		}
		for _, objective := range in.Policy.Objectives {
			if _, ok := dimensionValue(metrics, objective.Dimension); !ok {
				plan.receipt.Reasons = append(plan.receipt.Reasons,
					fmt.Sprintf("objective %s is unmodeled", objective.Dimension))
			}
		}
		if len(plan.receipt.Reasons) == 0 {
			plan.eligible = true
		}
		plans = append(plans, plan)
	}

	frontier := paretoFrontier(plans, in.Policy.Objectives)
	frontierIDs := make([]string, len(frontier))
	for i := range frontier {
		frontierIDs[i] = plans[frontier[i]].candidate.ID
	}
	selectedIndex, reason := resolveFrontier(plans, frontier, in.Policy)
	frontierSet := make(map[int]bool, len(frontier))
	for _, i := range frontier {
		frontierSet[i] = true
	}
	for i := range plans {
		switch {
		case i == selectedIndex:
			plans[i].receipt.Decision = DecisionSelected
		case frontierSet[i] && selectedIndex >= 0:
			plans[i].receipt.Decision = DecisionRejected
			plans[i].receipt.Reasons = append(plans[i].receipt.Reasons, "not selected by the explicit resolution policy")
		case frontierSet[i]:
			plans[i].receipt.Decision = DecisionFrontier
		case plans[i].eligible:
			plans[i].receipt.Decision = DecisionDominated
			plans[i].receipt.Reasons = append(plans[i].receipt.Reasons, "dominated on the declared Pareto objectives")
		}
	}

	receipt := SelectionReceipt{
		Schema: SelectionReceiptSchema, Engine: NativeEngine, WorkloadID: in.Workload.ID,
		Workload: in.Workload, Topology: cloneTopology(in.Topology), Reference: clonePlacement(in.Reference),
		Policy: cloneOperatorPolicy(in.Policy), Disposition: DispositionNoSelection,
		ParetoFrontier: frontierIDs, Fallback: "none", NoSelectionReason: reason,
		Alternatives: make([]AlternativeReceipt, 0, len(plans)),
	}
	for i := range plans {
		alt := cloneAlternativeReceipt(plans[i].receipt)
		receipt.Alternatives = append(receipt.Alternatives, alt)
		if provenanceIncludesEstimate(alt.CrossDomainProvenance) && alt.Decision != DecisionRejected {
			receipt.ContainsEstimatedCrossDomain = true
		}
		if (selectedIndex >= 0 && i != selectedIndex) ||
			(selectedIndex < 0 && (alt.Decision == DecisionRejected || alt.Decision == DecisionDominated)) {
			receipt.RejectedAlternatives = append(receipt.RejectedAlternatives, cloneAlternativeReceipt(alt))
		}
	}
	if selectedIndex >= 0 {
		selected := cloneAlternativeReceipt(plans[selectedIndex].receipt)
		receipt.Selected = &selected
		receipt.Disposition = DispositionSelected
		receipt.NoSelectionReason = ""
	}
	selectionID, err := computeSelectionID(receipt)
	if err != nil {
		return PlanResult{}, err
	}
	receipt.SelectionID = selectionID
	return PlanResult{Reports: reports, Receipt: receipt}, nil
}

func computeSelectionID(receipt SelectionReceipt) (string, error) {
	receipt.SelectionID = ""
	encoded, err := json.Marshal(receipt)
	if err != nil {
		return "", fmt.Errorf("encode selection receipt identity: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return fmt.Sprintf("sha256:%x", digest), nil
}

// VerifySelectionReceipt checks structural selection posture and the content
// binding. It detects mutation but provides no authenticity or signature claim.
func VerifySelectionReceipt(receipt SelectionReceipt) error {
	if receipt.Schema != SelectionReceiptSchema {
		return fmt.Errorf("selection schema %q does not match %q", receipt.Schema, SelectionReceiptSchema)
	}
	if receipt.Engine != NativeEngine {
		return fmt.Errorf("selection engine %q does not match %q", receipt.Engine, NativeEngine)
	}
	if err := validateWorkload(receipt.Workload); err != nil {
		return fmt.Errorf("selection workload: %w", err)
	}
	if receipt.WorkloadID != receipt.Workload.ID {
		return fmt.Errorf("selection workload id %q does not match bound workload %q", receipt.WorkloadID, receipt.Workload.ID)
	}
	if err := validatePolicy(receipt.Policy); err != nil {
		return fmt.Errorf("selection policy: %w", err)
	}
	if receipt.Fallback != "none" {
		return fmt.Errorf("selection fallback posture %q is not %q", receipt.Fallback, "none")
	}
	selectedAlternatives := 0
	selectedOnFrontier := false
	seen := make(map[string]bool, len(receipt.Alternatives))
	for _, alternative := range receipt.Alternatives {
		if err := nonblankID("alternative candidate id", alternative.CandidateID); err != nil {
			return err
		}
		if seen[alternative.CandidateID] {
			return fmt.Errorf("duplicate receipt alternative %q", alternative.CandidateID)
		}
		seen[alternative.CandidateID] = true
		if alternative.Engine != NativeEngine {
			return fmt.Errorf("alternative %q engine %q does not match %q", alternative.CandidateID, alternative.Engine, NativeEngine)
		}
		if alternative.Decision == DecisionSelected {
			selectedAlternatives++
		}
	}
	switch receipt.Disposition {
	case DispositionSelected:
		if receipt.Selected == nil {
			return fmt.Errorf("selected disposition has no selected alternative")
		}
		if receipt.Selected.Engine != NativeEngine || receipt.Selected.Decision != DecisionSelected {
			return fmt.Errorf("selected alternative has inconsistent engine or decision")
		}
		if !seen[receipt.Selected.CandidateID] || selectedAlternatives != 1 {
			return fmt.Errorf("selected candidate %q does not match exactly one selected alternative", receipt.Selected.CandidateID)
		}
		for _, candidateID := range receipt.ParetoFrontier {
			selectedOnFrontier = selectedOnFrontier || candidateID == receipt.Selected.CandidateID
		}
		if !selectedOnFrontier {
			return fmt.Errorf("selected candidate %q is absent from Pareto frontier", receipt.Selected.CandidateID)
		}
		if receipt.NoSelectionReason != "" {
			return fmt.Errorf("selected disposition carries a no-selection reason")
		}
	case DispositionNoSelection:
		if receipt.Selected != nil || selectedAlternatives != 0 {
			return fmt.Errorf("no-selection disposition carries a selected alternative")
		}
		if err := nonblankID("no-selection reason", receipt.NoSelectionReason); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown selection disposition %q", receipt.Disposition)
	}
	expected, err := computeSelectionID(receipt)
	if err != nil {
		return err
	}
	if receipt.SelectionID != expected {
		return fmt.Errorf("selection id mismatch: got %q, want %q", receipt.SelectionID, expected)
	}
	return nil
}

func cloneTopology(topology Topology) Topology {
	topology.Domains = append([]Domain(nil), topology.Domains...)
	topology.Nodes = append([]Node(nil), topology.Nodes...)
	topology.Links = append([]Link(nil), topology.Links...)
	return topology
}

func clonePlacement(placement Placement) Placement {
	placement.Allocations = append([]Allocation(nil), placement.Allocations...)
	placement.Transfers = append([]Transfer(nil), placement.Transfers...)
	return placement
}

func cloneTopologyIntent(intent TopologyIntent) TopologyIntent {
	intent.Hierarchy = append([]string(nil), intent.Hierarchy...)
	return intent
}

func cloneOperatorPolicy(policy OperatorPolicy) OperatorPolicy {
	policy.Constraints = append([]Constraint(nil), policy.Constraints...)
	policy.Objectives = append([]Objective(nil), policy.Objectives...)
	if policy.SLO != nil {
		slo := *policy.SLO
		policy.SLO = &slo
	}
	return policy
}

func cloneAlternativeReceipt(alternative AlternativeReceipt) AlternativeReceipt {
	alternative.Placement = clonePlacement(alternative.Placement)
	alternative.Topology = cloneTopologyIntent(alternative.Topology)
	alternative.Nodes = append([]string(nil), alternative.Nodes...)
	alternative.Domains = append([]string(nil), alternative.Domains...)
	alternative.Links = append([]string(nil), alternative.Links...)
	alternative.Reasons = append([]string(nil), alternative.Reasons...)
	if alternative.Metrics != nil {
		metrics := *alternative.Metrics
		alternative.Metrics = &metrics
	}
	return alternative
}

func validateCandidate(c PlanCandidate, topology Topology) error {
	if err := nonblankID("candidate id", c.ID); err != nil {
		return err
	}
	if c.Engine != NativeEngine {
		return fmt.Errorf("engine must be %q; external engine fallback is not selectable", NativeEngine)
	}
	if err := validateStrategy(c.Strategy); err != nil {
		return err
	}
	if err := nonblankID("candidate rationale", c.Rationale); err != nil {
		return err
	}
	if err := nonblankID("topology rationale", c.Topology.Rationale); err != nil {
		return err
	}
	if len(c.Topology.Hierarchy) == 0 {
		return fmt.Errorf("topology hierarchy must not be empty")
	}
	for i, level := range c.Topology.Hierarchy {
		if err := nonblankID(fmt.Sprintf("topology hierarchy[%d]", i), level); err != nil {
			return err
		}
	}
	if !validInputProvenance(c.Provenance) {
		return fmt.Errorf("provenance must be %q or %q", ProvenanceEstimated, ProvenanceMeasured)
	}
	if err := validateSLO(c.SLO); err != nil {
		return err
	}
	crossDomain := spansDomains(c.Placement, topology)
	if crossDomain && c.CrossDomainProvenance != "" && !validInputProvenance(c.CrossDomainProvenance) {
		return fmt.Errorf("cross-domain provenance must be measured, estimated, or left for analyzer derivation")
	}
	if !crossDomain && c.CrossDomainProvenance != "" {
		return fmt.Errorf("single-domain placement must not declare cross-domain provenance")
	}
	return nil
}

func validateStrategy(s ParallelismStrategy) error {
	degrees := []int{s.TensorDegree, s.PipelineStages, s.ExpertDegree, s.DataReplicas, s.SequenceDegree, s.ContextDegree}
	for _, degree := range degrees {
		if degree < 1 {
			return fmt.Errorf("all parallelism degrees must be at least one")
		}
	}
	kindIndex := map[ParallelismKind]int{
		ParallelismTensor: 0, ParallelismPipeline: 1, ParallelismExpert: 2,
		ParallelismData: 3, ParallelismSequence: 4, ParallelismContext: 5,
	}
	if s.Kind == ParallelismHybrid {
		larger := 0
		for _, degree := range degrees {
			if degree > 1 {
				larger++
			}
		}
		if larger < 2 {
			return fmt.Errorf("hybrid strategy must enlarge at least two parallelism axes")
		}
		return nil
	}
	axis, ok := kindIndex[s.Kind]
	if !ok {
		return fmt.Errorf("unknown parallelism kind %q", s.Kind)
	}
	if degrees[axis] <= 1 {
		return fmt.Errorf("%s strategy must enlarge its named axis", s.Kind)
	}
	for i, degree := range degrees {
		if i != axis && degree != 1 {
			return fmt.Errorf("%s strategy cannot enlarge another axis; use hybrid", s.Kind)
		}
	}
	return nil
}

func validateSLO(s SLOProjection) error {
	if !s.Modeled {
		if s.Name != "" || s.Unit != "" || s.Margin != 0 || s.Provenance != "" {
			return fmt.Errorf("unmodeled SLO projection must be empty")
		}
		return nil
	}
	if err := nonblankID("SLO name", s.Name); err != nil {
		return err
	}
	if err := nonblankID("SLO unit", s.Unit); err != nil {
		return err
	}
	if math.IsNaN(s.Margin) || math.IsInf(s.Margin, 0) {
		return fmt.Errorf("SLO margin must be finite")
	}
	if !validInputProvenance(s.Provenance) {
		return fmt.Errorf("SLO provenance must be measured or estimated")
	}
	return nil
}

func validatePolicy(p OperatorPolicy) error {
	if len(p.Objectives) == 0 {
		return fmt.Errorf("operator policy must declare at least one objective")
	}
	if p.Resolution != ResolveUnique && p.Resolution != ResolveLexicographic {
		return fmt.Errorf("unknown resolution mode %q", p.Resolution)
	}
	if p.TieBreak != TieBreakRefuse && p.TieBreak != TieBreakCandidateID {
		return fmt.Errorf("unknown tie-break mode %q", p.TieBreak)
	}
	if p.Resolution == ResolveUnique && p.TieBreak != TieBreakRefuse {
		return fmt.Errorf("unique-frontier resolution does not use a tie-break")
	}
	if p.EstimatedCrossDomain != RefuseEstimatedCrossDomain && p.EstimatedCrossDomain != AllowEstimatedCrossDomain {
		return fmt.Errorf("unknown estimated cross-domain policy %q", p.EstimatedCrossDomain)
	}
	seen := make(map[Dimension]bool, len(p.Objectives))
	usesSLO := false
	for _, objective := range p.Objectives {
		if !validDimension(objective.Dimension) {
			return fmt.Errorf("unknown objective dimension %q", objective.Dimension)
		}
		if objective.Direction != Minimize && objective.Direction != Maximize {
			return fmt.Errorf("objective %s has unknown direction %q", objective.Dimension, objective.Direction)
		}
		if seen[objective.Dimension] {
			return fmt.Errorf("duplicate objective dimension %q", objective.Dimension)
		}
		seen[objective.Dimension] = true
		usesSLO = usesSLO || objective.Dimension == DimensionSLOMargin
	}
	for _, constraint := range p.Constraints {
		if !validDimension(constraint.Dimension) {
			return fmt.Errorf("unknown constraint dimension %q", constraint.Dimension)
		}
		if constraint.Operator != AtMost && constraint.Operator != AtLeast {
			return fmt.Errorf("constraint %s has unknown operator %q", constraint.Dimension, constraint.Operator)
		}
		if math.IsNaN(constraint.Value) || math.IsInf(constraint.Value, 0) {
			return fmt.Errorf("constraint %s value must be finite", constraint.Dimension)
		}
		usesSLO = usesSLO || constraint.Dimension == DimensionSLOMargin
	}
	if usesSLO && p.SLO == nil {
		return fmt.Errorf("SLO objectives and constraints require an explicit SLO name and unit")
	}
	if p.SLO != nil {
		if err := nonblankID("policy SLO name", p.SLO.Name); err != nil {
			return err
		}
		if err := nonblankID("policy SLO unit", p.SLO.Unit); err != nil {
			return err
		}
	}
	return nil
}

func validateSLOComparability(candidates []PlanCandidate, policy OperatorPolicy) error {
	if policy.SLO == nil {
		return nil
	}
	for _, candidate := range candidates {
		if !candidate.SLO.Modeled {
			continue
		}
		if candidate.SLO.Name != policy.SLO.Name || candidate.SLO.Unit != policy.SLO.Unit {
			return fmt.Errorf("candidate %q SLO %q [%s] does not match policy SLO %q [%s]", candidate.ID,
				candidate.SLO.Name, candidate.SLO.Unit, policy.SLO.Name, policy.SLO.Unit)
		}
	}
	return nil
}

func policyUsesDimension(policy OperatorPolicy, dimension Dimension) bool {
	for _, objective := range policy.Objectives {
		if objective.Dimension == dimension {
			return true
		}
	}
	for _, constraint := range policy.Constraints {
		if constraint.Dimension == dimension {
			return true
		}
	}
	return false
}

func validDimension(d Dimension) bool {
	switch d {
	case DimensionLatencySeconds, DimensionThroughput, DimensionMonetaryUSD,
		DimensionEnergyJoules, DimensionMemoryHeadroomBytes,
		DimensionComputeHeadroomUnits, DimensionSLOMargin:
		return true
	default:
		return false
	}
}

func evaluationProvenance(e Evaluation) Provenance {
	var effective Provenance
	for _, entry := range e.Ledger {
		if componentContributes(entry.Cost) {
			effective = mergeProvenance(effective, entry.Cost.Provenance)
		}
	}
	// Communication provenance remains relevant even when overlap hides all of its
	// latency: bytes, money, energy, and the placement assumption still contribute.
	for _, record := range e.Communication {
		effective = mergeProvenance(effective, record.Provenance)
	}
	return effective
}

func componentContributes(c ComponentCost) bool {
	return c.Latency != 0 || c.Cycle != 0 || c.MonetaryUSD.Value != 0 || c.EnergyJoules.Value != 0
}

func boundaryProvenance(e Evaluation, topology Topology, crossDomain bool) Provenance {
	if !crossDomain {
		return ""
	}
	nodeDomain := make(map[string]string, len(topology.Nodes))
	for _, node := range topology.Nodes {
		nodeDomain[node.ID] = node.DomainID
	}
	var effective Provenance
	for _, record := range e.Communication {
		if nodeDomain[record.FromNode] != nodeDomain[record.ToNode] {
			effective = mergeProvenance(effective, record.Provenance)
		}
	}
	if effective == "" {
		// A cross-domain allocation without explicit boundary evidence is an
		// uncalibrated estimate, never an implicit measurement.
		return ProvenanceEstimated
	}
	return effective
}

func provenanceIncludesEstimate(p Provenance) bool {
	return p == ProvenanceEstimated || p == ProvenanceMixed
}

func spansDomains(p Placement, topology Topology) bool {
	nodeDomain := make(map[string]string, len(topology.Nodes))
	for _, node := range topology.Nodes {
		nodeDomain[node.ID] = node.DomainID
	}
	domains := make(map[string]bool)
	for _, allocation := range p.Allocations {
		if domain := nodeDomain[allocation.NodeID]; domain != "" {
			domains[domain] = true
		}
	}
	return len(domains) > 1
}

func placementTopologyIDs(p Placement, topology Topology) ([]string, []string, []string) {
	nodeDomain := make(map[string]string, len(topology.Nodes))
	for _, node := range topology.Nodes {
		nodeDomain[node.ID] = node.DomainID
	}
	nodes := make([]string, 0, len(p.Allocations))
	domainSet := make(map[string]bool)
	for _, allocation := range p.Allocations {
		nodes = append(nodes, allocation.NodeID)
		if domain := nodeDomain[allocation.NodeID]; domain != "" {
			domainSet[domain] = true
		}
	}
	domains := make([]string, 0, len(domainSet))
	for domain := range domainSet {
		domains = append(domains, domain)
	}
	links := make([]string, 0, len(p.Transfers))
	for _, transfer := range p.Transfers {
		links = append(links, transfer.LinkID)
	}
	sort.Strings(nodes)
	sort.Strings(domains)
	sort.Strings(links)
	return nodes, domains, links
}

func candidateMetrics(e Evaluation, slo SLOProjection) PlanMetrics {
	headroom := CapacityHeadroom{MemoryBytes: math.MaxUint64, ComputeUnits: math.MaxFloat64}
	for _, check := range e.Feasibility.Checks {
		memory := check.Available.MemoryBytes - check.Demand.MemoryBytes
		compute := check.Available.ComputeUnits - check.Demand.ComputeUnits
		if memory < headroom.MemoryBytes {
			headroom.MemoryBytes = memory
		}
		if compute < headroom.ComputeUnits {
			headroom.ComputeUnits = compute
		}
	}
	if len(e.Feasibility.Checks) == 0 {
		headroom = CapacityHeadroom{}
	}
	return PlanMetrics{
		Latency: e.Metrics.Latency, Cycle: e.Metrics.Cycle,
		Throughput: e.Metrics.Throughput, ThroughputUnit: e.Metrics.ThroughputUnit,
		MonetaryUSD: e.Metrics.MonetaryUSD, EnergyJoules: e.Metrics.EnergyJoules,
		Capacity: headroom, SLO: slo,
	}
}

func dimensionValue(metrics PlanMetrics, d Dimension) (float64, bool) {
	switch d {
	case DimensionLatencySeconds:
		return metrics.Latency.Seconds(), true
	case DimensionThroughput:
		return metrics.Throughput, true
	case DimensionMonetaryUSD:
		return metrics.MonetaryUSD.Value, metrics.MonetaryUSD.Modeled
	case DimensionEnergyJoules:
		return metrics.EnergyJoules.Value, metrics.EnergyJoules.Modeled
	case DimensionMemoryHeadroomBytes:
		return float64(metrics.Capacity.MemoryBytes), true
	case DimensionComputeHeadroomUnits:
		return metrics.Capacity.ComputeUnits, true
	case DimensionSLOMargin:
		return metrics.SLO.Margin, metrics.SLO.Modeled
	default:
		return 0, false
	}
}

func constraintAllows(value float64, c Constraint) bool {
	if c.Operator == AtMost {
		return value <= c.Value
	}
	return value >= c.Value
}

func paretoFrontier(plans []evaluatedPlan, objectives []Objective) []int {
	var frontier []int
	for i := range plans {
		if !plans[i].eligible {
			continue
		}
		dominated := false
		for j := range plans {
			if i != j && plans[j].eligible && dominates(plans[j].receipt.Metrics, plans[i].receipt.Metrics, objectives) {
				dominated = true
				break
			}
		}
		if !dominated {
			frontier = append(frontier, i)
		}
	}
	return frontier
}

func dominates(a, b *PlanMetrics, objectives []Objective) bool {
	better := false
	for _, objective := range objectives {
		av, _ := dimensionValue(*a, objective.Dimension)
		bv, _ := dimensionValue(*b, objective.Dimension)
		if objective.Direction == Minimize {
			if av > bv {
				return false
			}
			better = better || av < bv
		} else {
			if av < bv {
				return false
			}
			better = better || av > bv
		}
	}
	return better
}

func resolveFrontier(plans []evaluatedPlan, frontier []int, policy OperatorPolicy) (int, string) {
	if len(frontier) == 0 {
		return -1, "no candidate survived feasibility, calibration, constraints, and modeled-objective filters"
	}
	if policy.Resolution == ResolveUnique {
		if len(frontier) == 1 {
			return frontier[0], ""
		}
		return -1, "operator policy requires a unique Pareto candidate"
	}
	best := frontier[0]
	tied := []int{best}
	for _, candidate := range frontier[1:] {
		comparison := compareLexicographic(plans[candidate].receipt.Metrics, plans[best].receipt.Metrics, policy.Objectives)
		switch {
		case comparison < 0:
			best = candidate
			tied = []int{candidate}
		case comparison == 0:
			tied = append(tied, candidate)
		}
	}
	if len(tied) == 1 {
		return best, ""
	}
	if policy.TieBreak == TieBreakCandidateID {
		// plans and frontier are already candidate-ID ordered.
		return tied[0], ""
	}
	return -1, "lexicographic objectives are tied and operator policy refuses an implicit tie-break"
}

func compareLexicographic(a, b *PlanMetrics, objectives []Objective) int {
	for _, objective := range objectives {
		av, _ := dimensionValue(*a, objective.Dimension)
		bv, _ := dimensionValue(*b, objective.Dimension)
		if av == bv {
			continue
		}
		if objective.Direction == Minimize {
			if av < bv {
				return -1
			}
			return 1
		}
		if av > bv {
			return -1
		}
		return 1
	}
	return 0
}
