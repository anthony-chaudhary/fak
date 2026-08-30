package placementtax

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestPlannerStrategyVocabulary(t *testing.T) {
	strategies := []ParallelismStrategy{
		strategy(ParallelismTensor, 2),
		strategy(ParallelismPipeline, 2),
		strategy(ParallelismExpert, 2),
		strategy(ParallelismData, 2),
		strategy(ParallelismSequence, 2),
		strategy(ParallelismContext, 2),
		strategy(ParallelismHybrid, 2),
	}
	for _, got := range strategies {
		if err := validateStrategy(got); err != nil {
			t.Errorf("strategy %s rejected: %v", got.Kind, err)
		}
	}
}

func TestPlannerComputeBoundTensorParallelWins(t *testing.T) {
	in := plannerFixture()
	tp := sameHostCandidate("tp", strategy(ParallelismTensor, 2), 40*time.Millisecond, 10_000_000)
	tp.Placement.Synchronization = estimatedCost(time.Millisecond)
	data := singleNodeCandidate("data", strategy(ParallelismData, 2), 80*time.Millisecond)
	in.Candidates = []PlanCandidate{data, tp} // deliberately reverse ID order

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, "tp")
	if got, want := result.Reports["tp"].Candidate.Communication[0].RawLatency, 1100*time.Microsecond; got != want {
		t.Fatalf("TP raw communication = %v, want %v", got, want)
	}
	if got := result.Receipt.Alternatives[0].CandidateID; got != "data" {
		t.Fatalf("alternatives are not deterministic: first = %q", got)
	}
	if len(result.Receipt.RejectedAlternatives) != 1 || result.Receipt.RejectedAlternatives[0].CandidateID != "data" {
		t.Fatalf("rejected alternatives = %+v, want data", result.Receipt.RejectedAlternatives)
	}
}

func TestPlannerCommunicationBoundTensorParallelLoses(t *testing.T) {
	in := plannerFixture()
	tp := sameHostCandidate("tp", strategy(ParallelismTensor, 2), 40*time.Millisecond, 10_000_000_000)
	data := singleNodeCandidate("data", strategy(ParallelismData, 2), 80*time.Millisecond)
	in.Candidates = []PlanCandidate{tp, data}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, "data")
	if got := result.Reports["tp"].Candidate.Metrics.Latency; got <= data.Placement.UsefulCompute.Latency {
		t.Fatalf("communication-bound TP latency = %v, want > %v", got, data.Placement.UsefulCompute.Latency)
	}
}

func TestPlannerPipelineBubbleAndImbalance(t *testing.T) {
	in := plannerFixture()
	pipeline := sameHostCandidate("pipeline", strategy(ParallelismPipeline, 2), 35*time.Millisecond, 1_000_000)
	pipeline.Placement.ImbalanceStraggler = ComponentCost{
		Latency: 45 * time.Millisecond, Cycle: 55 * time.Millisecond, Provenance: ProvenanceEstimated,
	}
	pipeline.Rationale = "two stages expose a deterministic bubble and slow-stage imbalance"
	tensor := sameHostCandidate("tensor", strategy(ParallelismTensor, 2), 65*time.Millisecond, 1_000_000)
	in.Candidates = []PlanCandidate{pipeline, tensor}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, "tensor")
	got := ledgerCost(result.Reports["pipeline"].Candidate.Ledger, ComponentImbalanceStraggler)
	if got.Latency != 45*time.Millisecond || got.Cycle != 55*time.Millisecond {
		t.Fatalf("pipeline bubble/imbalance ledger = %+v", got)
	}
}

func TestPlannerExpertAllToAllSkew(t *testing.T) {
	in := plannerFixture()
	expert := sameHostCandidate("expert", strategy(ParallelismExpert, 2), 25*time.Millisecond, 6_000_000_000)
	expert.Placement.Transfers[0].Messages = 32
	expert.Placement.ImbalanceStraggler = estimatedCost(50 * time.Millisecond)
	expert.Rationale = "expert all-to-all includes routing skew in the imbalance ledger"
	data := singleNodeCandidate("data", strategy(ParallelismData, 2), 90*time.Millisecond)
	in.Candidates = []PlanCandidate{expert, data}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, "data")
	comm := result.Reports["expert"].Candidate.Communication[0]
	if comm.Messages != 32 || comm.Bytes != 6_000_000_000 {
		t.Fatalf("expert all-to-all record = %+v", comm)
	}
	if got := ledgerCost(result.Reports["expert"].Candidate.Ledger, ComponentImbalanceStraggler).Latency; got != 50*time.Millisecond {
		t.Fatalf("expert skew = %v, want 50ms", got)
	}
}

func TestPlannerSelectsFeasibleCandidateWhenSingleHostReferenceIsInfeasible(t *testing.T) {
	in := plannerFixture()
	in.Reference.Allocations[0].Demand.MemoryBytes = 300 * gib
	in.Candidates = []PlanCandidate{sameHostCandidate("tp", strategy(ParallelismTensor, 2), 60*time.Millisecond, 1_000_000)}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, "tp")
	if result.Receipt.Selected.ReferenceFeasible {
		t.Fatal("infeasible single-host reference was labeled feasible")
	}
	report := result.Reports["tp"]
	if report.Reference.Feasibility.Feasible || report.Delta != nil || report.Relative != nil {
		t.Fatalf("reference comparison was not refused: %+v", report)
	}
}

func TestPlannerHeterogeneousLinksFavorTopologyAwareHierarchy(t *testing.T) {
	in := plannerFixture()
	in.Policy.EstimatedCrossDomain = AllowEstimatedCrossDomain
	hierarchical := hybridCandidate("hierarchical", 10_000_000)
	hierarchical.CrossDomainProvenance = ProvenanceMeasured
	hierarchical.Topology = TopologyIntent{
		Hierarchy: []string{"accelerator_pair", "host", "rack"},
		Rationale: "tensor traffic stays on the fast local link; only stage output crosses the slow rack link",
	}
	flat := hybridCandidate("flat", 4_000_000_000)
	flat.CrossDomainProvenance = ProvenanceMeasured
	flat.Rationale = "flat decomposition sends tensor traffic over the rack link"
	flat.Topology = TopologyIntent{Hierarchy: []string{"rack"}, Rationale: "flat cross-host path ignores the fast local tier"}
	in.Candidates = []PlanCandidate{flat, hierarchical}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, "hierarchical")
	if got := result.Receipt.Selected.Topology.Hierarchy; len(got) != 3 || got[0] != "accelerator_pair" {
		t.Fatalf("selected topology hierarchy = %v", got)
	}
	fast := result.Reports["hierarchical"].Candidate.Communication[0]
	slow := result.Reports["hierarchical"].Candidate.Communication[1]
	if fast.RawLatency >= slow.RawLatency {
		t.Fatalf("heterogeneous link costs not preserved: fast=%v slow=%v", fast.RawLatency, slow.RawLatency)
	}
}

func TestPlannerEstimatedCrossDomainIsRefusedOrExplicitlyLabeled(t *testing.T) {
	in := plannerFixture()
	candidate := hybridCandidate("estimated-remote", 10_000_000)
	candidate.CrossDomainProvenance = ProvenanceEstimated
	for i := range candidate.Placement.Transfers {
		candidate.Placement.Transfers[i].Provenance = ProvenanceEstimated
	}
	in.Candidates = []PlanCandidate{candidate}

	refused, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	if refused.Receipt.Selected != nil || !strings.Contains(refused.Receipt.Alternatives[0].Reasons[0], "estimated cross-domain") {
		t.Fatalf("estimated cross-domain candidate was not refused: %+v", refused.Receipt)
	}

	in.Policy.EstimatedCrossDomain = AllowEstimatedCrossDomain
	allowed, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, allowed, "estimated-remote")
	if !allowed.Receipt.ContainsEstimatedCrossDomain || !allowed.Receipt.Selected.Estimated {
		t.Fatalf("allowed estimate was not labeled: %+v", allowed.Receipt)
	}
}

func TestPlannerBoundedLocalGPULayersSerializeThroughSelectionContract(t *testing.T) {
	in := plannerFixture()
	local := PlanCandidate{
		ID: "qwen38-bounded-local-gpu-layers", Engine: NativeEngine,
		Strategy: strategy(ParallelismPipeline, 2),
		Placement: Placement{
			Name: "bounded-local-gpu-layers",
			Allocations: []Allocation{
				{NodeID: "cpu", Demand: Capacity{MemoryBytes: 72 * gib, ComputeUnits: 8}},
				{NodeID: "gpu0", Demand: Capacity{MemoryBytes: 20 * gib, ComputeUnits: 20}},
			},
			UsefulCompute: modeledEstimatedCost(55*time.Millisecond, .06, 2.5),
			Transfers:     []Transfer{{LinkID: "cpu-gpu0", Messages: 1, Bytes: 100_000_000, Provenance: ProvenanceEstimated}},
		},
		Topology: TopologyIntent{
			Hierarchy: []string{"layer", "device", "host"},
			Rationale: "a bounded layer range is resident on the local GPU and remaining layers stay on CPU",
		},
		Rationale:  "bounded local-GPU layer placement uses the same candidate and receipt contract as distributed plans",
		Provenance: ProvenanceEstimated,
		SLO:        SLOProjection{Name: "interactive_decode", Unit: "milliseconds", Margin: 12, Modeled: true, Provenance: ProvenanceEstimated},
	}
	in.Candidates = []PlanCandidate{local}
	in.Policy = OperatorPolicy{
		Constraints: []Constraint{
			{Dimension: DimensionLatencySeconds, Operator: AtMost, Value: .2},
			{Dimension: DimensionSLOMargin, Operator: AtLeast, Value: 0},
		},
		Objectives: []Objective{
			{Dimension: DimensionLatencySeconds, Direction: Minimize},
			{Dimension: DimensionThroughput, Direction: Maximize},
			{Dimension: DimensionMonetaryUSD, Direction: Minimize},
			{Dimension: DimensionEnergyJoules, Direction: Minimize},
			{Dimension: DimensionMemoryHeadroomBytes, Direction: Maximize},
			{Dimension: DimensionComputeHeadroomUnits, Direction: Maximize},
			{Dimension: DimensionSLOMargin, Direction: Maximize},
		},
		SLO:        &SLOIdentity{Name: "interactive_decode", Unit: "milliseconds"},
		Resolution: ResolveUnique, TieBreak: TieBreakRefuse,
		EstimatedCrossDomain: RefuseEstimatedCrossDomain,
	}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, local.ID)
	encoded, err := json.Marshal(result.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, want := range []string{
		SelectionReceiptSchema, `"engine":"fak-native"`, local.ID,
		`"kind":"pipeline"`, `"fallback":"none"`, `"provenance":"estimated"`,
		"bounded layer range", `"memory_headroom_bytes"`, `"slo_margin"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("serialized receipt missing %q: %s", want, text)
		}
	}
	if result.Receipt.Selected.Metrics.MonetaryUSD.Value == result.Receipt.Selected.Metrics.EnergyJoules.Value {
		t.Fatal("cost and energy dimensions were conflated")
	}
}

func TestPlannerParetoTieNeedsExplicitResolution(t *testing.T) {
	in := plannerFixture()
	a := singleNodeCandidate("a", strategy(ParallelismData, 2), 50*time.Millisecond)
	b := singleNodeCandidate("b", strategy(ParallelismData, 2), 50*time.Millisecond)
	in.Candidates = []PlanCandidate{b, a}
	in.Policy.Resolution = ResolveLexicographic
	in.Policy.TieBreak = TieBreakRefuse

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	if result.Receipt.Selected != nil || result.Receipt.Disposition != DispositionNoSelection {
		t.Fatalf("implicit tie-break selected a plan: %+v", result.Receipt)
	}
	if got := strings.Join(result.Receipt.ParetoFrontier, ","); got != "a,b" {
		t.Fatalf("frontier = %q, want deterministic a,b", got)
	}

	in.Policy.TieBreak = TieBreakCandidateID
	resolved, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, resolved, "a")
}

func TestPlannerRefusesExternalEngineCandidate(t *testing.T) {
	in := plannerFixture()
	candidate := singleNodeCandidate("external", strategy(ParallelismData, 2), 50*time.Millisecond)
	candidate.Engine = "llama.cpp"
	in.Candidates = []PlanCandidate{candidate}
	_, err := Plan(in)
	if err == nil || !strings.Contains(err.Error(), "external engine fallback") {
		t.Fatalf("Plan() error = %v, want external engine refusal", err)
	}
}

func TestPlannerDerivesEffectiveProvenanceFromAnalyzedCosts(t *testing.T) {
	in := plannerFixture()
	candidate := sameHostCandidate("mixed", strategy(ParallelismTensor, 2), 40*time.Millisecond, 10_000_000)
	candidate.Placement.UsefulCompute = measuredCost(40 * time.Millisecond)
	candidate.Placement.Transfers[0].Provenance = ProvenanceEstimated
	candidate.Provenance = ProvenanceMeasured
	in.Candidates = []PlanCandidate{candidate}

	_, err := Plan(in)
	if err == nil || !strings.Contains(err.Error(), "declares measured provenance") || !strings.Contains(err.Error(), "mixed") {
		t.Fatalf("measured overclaim error = %v, want analyzed mixed provenance refusal", err)
	}

	candidate.Provenance = ProvenanceEstimated
	in.Candidates = []PlanCandidate{candidate}
	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, candidate.ID)
	if result.Receipt.Selected.Provenance != ProvenanceMixed || !result.Receipt.Selected.Estimated {
		t.Fatalf("derived mixed provenance not preserved: %+v", result.Receipt.Selected)
	}

	candidate.Placement.Transfers[0].Provenance = ProvenanceMeasured
	in.Candidates = []PlanCandidate{candidate}
	measured, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	if measured.Receipt.Selected.Provenance != ProvenanceMeasured || measured.Receipt.Selected.Estimated {
		t.Fatalf("caller estimate overrode measured analyzer evidence: %+v", measured.Receipt.Selected)
	}
}

func TestPlannerCrossDomainPolicyUsesAnalyzedBoundaryEvidence(t *testing.T) {
	in := plannerFixture()
	candidate := hybridCandidate("boundary-mixed", 10_000_000)
	candidate.CrossDomainProvenance = ProvenanceMeasured
	candidate.Placement.Transfers[1].Provenance = ProvenanceEstimated
	in.Candidates = []PlanCandidate{candidate}

	_, err := Plan(in)
	if err == nil || !strings.Contains(err.Error(), "declares measured cross-domain provenance") {
		t.Fatalf("cross-domain overclaim error = %v", err)
	}

	candidate.CrossDomainProvenance = ""
	in.Candidates = []PlanCandidate{candidate}
	refused, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	if refused.Receipt.Selected != nil || refused.Receipt.Alternatives[0].CrossDomainProvenance != ProvenanceEstimated {
		t.Fatalf("derived estimated boundary was not refused: %+v", refused.Receipt)
	}
	in.Policy.EstimatedCrossDomain = AllowEstimatedCrossDomain
	allowed, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, allowed, candidate.ID)
	if !allowed.Receipt.ContainsEstimatedCrossDomain || !allowed.Receipt.Selected.Estimated {
		t.Fatalf("allowed boundary estimate not labeled: %+v", allowed.Receipt)
	}
}

func TestPlannerForbidsCrossDomainDeclarationOnLocalCandidate(t *testing.T) {
	in := plannerFixture()
	candidate := sameHostCandidate("local", strategy(ParallelismTensor, 2), 40*time.Millisecond, 1_000_000)
	candidate.CrossDomainProvenance = ProvenanceMeasured
	in.Candidates = []PlanCandidate{candidate}
	_, err := Plan(in)
	if err == nil || !strings.Contains(err.Error(), "single-domain placement must not declare cross-domain provenance") {
		t.Fatalf("Plan() error = %v, want local provenance consistency refusal", err)
	}
}

func TestPlannerSLOIdentityAndNativeUnitMustMatchPolicy(t *testing.T) {
	tests := []struct {
		name string
		slo  SLOProjection
		want string
	}{
		{
			name: "TTFT versus throughput",
			slo:  SLOProjection{Name: "decode_throughput", Unit: "tokens_per_second", Margin: 5, Modeled: true, Provenance: ProvenanceMeasured},
			want: `does not match policy SLO "ttft" [milliseconds]`,
		},
		{
			name: "native unit mismatch",
			slo:  SLOProjection{Name: "ttft", Unit: "seconds", Margin: .1, Modeled: true, Provenance: ProvenanceMeasured},
			want: `does not match policy SLO "ttft" [milliseconds]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := plannerFixture()
			in.Policy.Objectives = []Objective{{Dimension: DimensionSLOMargin, Direction: Maximize}}
			in.Policy.SLO = &SLOIdentity{Name: "ttft", Unit: "milliseconds"}
			a := singleNodeCandidate("a", strategy(ParallelismData, 2), 50*time.Millisecond)
			a.SLO = SLOProjection{Name: "ttft", Unit: "milliseconds", Margin: 10, Modeled: true, Provenance: ProvenanceMeasured}
			b := singleNodeCandidate("b", strategy(ParallelismData, 2), 60*time.Millisecond)
			b.SLO = tt.slo
			in.Candidates = []PlanCandidate{a, b}
			_, err := Plan(in)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Plan() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestPlannerSLOPolicyRequiresStableIdentity(t *testing.T) {
	in := plannerFixture()
	in.Policy.Objectives = []Objective{{Dimension: DimensionSLOMargin, Direction: Maximize}}
	in.Candidates = []PlanCandidate{singleNodeCandidate("a", strategy(ParallelismData, 2), 50*time.Millisecond)}
	_, err := Plan(in)
	if err == nil || !strings.Contains(err.Error(), "explicit SLO name and unit") {
		t.Fatalf("Plan() error = %v, want missing SLO identity refusal", err)
	}
}

func TestPlannerSelectionIDIsOrderIndependentAndContentBound(t *testing.T) {
	fixture := func() PlannerInput {
		in := plannerFixture()
		in.Candidates = []PlanCandidate{
			sameHostCandidate("tp", strategy(ParallelismTensor, 2), 40*time.Millisecond, 10_000_000),
			singleNodeCandidate("data", strategy(ParallelismData, 2), 80*time.Millisecond),
		}
		return in
	}
	baseInput := fixture()
	base, err := Plan(baseInput)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(base.Receipt.SelectionID, "sha256:") || len(base.Receipt.SelectionID) != len("sha256:")+64 {
		t.Fatalf("selection ID = %q, want sha256 content identity", base.Receipt.SelectionID)
	}
	recomputed, err := computeSelectionID(base.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if recomputed != base.Receipt.SelectionID {
		t.Fatalf("selection ID = %q, recomputed %q", base.Receipt.SelectionID, recomputed)
	}
	if err := VerifySelectionReceipt(base.Receipt); err != nil {
		t.Fatalf("verify base selection: %v", err)
	}

	reorderedInput := fixture()
	reorderedInput.Candidates[0], reorderedInput.Candidates[1] = reorderedInput.Candidates[1], reorderedInput.Candidates[0]
	reordered, err := Plan(reorderedInput)
	if err != nil {
		t.Fatal(err)
	}
	if reordered.Receipt.SelectionID != base.Receipt.SelectionID {
		t.Fatalf("candidate input order changed selection ID: %q != %q", reordered.Receipt.SelectionID, base.Receipt.SelectionID)
	}
	if err := VerifySelectionReceipt(reordered.Receipt); err != nil {
		t.Fatalf("verify reordered selection: %v", err)
	}

	mutations := []struct {
		name string
		edit func(*PlannerInput)
	}{
		{"workload", func(in *PlannerInput) { in.Workload.Quality.SequenceLength++ }},
		{"policy", func(in *PlannerInput) {
			in.Policy.Constraints = append(in.Policy.Constraints, Constraint{Dimension: DimensionLatencySeconds, Operator: AtMost, Value: 1})
		}},
		{"placement", func(in *PlannerInput) { in.Candidates[0].Placement.UsefulCompute.Latency += time.Millisecond }},
		{"topology", func(in *PlannerInput) { in.Topology.Links[0].BandwidthBytesPerSecond /= 2 }},
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			in := fixture()
			mutation.edit(&in)
			result, err := Plan(in)
			if err != nil {
				t.Fatal(err)
			}
			if result.Receipt.SelectionID == base.Receipt.SelectionID {
				t.Fatalf("%s mutation did not change selection ID %q", mutation.name, result.Receipt.SelectionID)
			}
		})
	}

	posture := base.Receipt
	posture.Disposition = DispositionNoSelection
	posture.Selected = nil
	posture.NoSelectionReason = "operator left frontier unresolved"
	postureID, err := computeSelectionID(posture)
	if err != nil {
		t.Fatal(err)
	}
	if postureID == base.Receipt.SelectionID {
		t.Fatal("selected/no-selection posture did not change selection ID")
	}
}

func TestPlannerReceiptDeepClonesInput(t *testing.T) {
	in := plannerFixture()
	in.Policy.Constraints = []Constraint{{Dimension: DimensionLatencySeconds, Operator: AtMost, Value: 1}}
	in.Policy.SLO = &SLOIdentity{Name: "ttft", Unit: "milliseconds"}
	in.Candidates = []PlanCandidate{sameHostCandidate("clone", strategy(ParallelismTensor, 2), 40*time.Millisecond, 1_000_000)}
	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(result.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	selectionID := result.Receipt.SelectionID

	in.Topology.Domains[0].ID = "mutated-domain"
	in.Topology.Nodes[0].Capacity.MemoryBytes++
	in.Topology.Links[0].BandwidthBytesPerSecond /= 2
	in.Reference.Allocations[0].Demand.MemoryBytes++
	in.Policy.Constraints[0].Value = .001
	in.Policy.Objectives[0].Direction = Maximize
	in.Policy.SLO.Name = "mutated-slo"
	in.Candidates[0].Placement.Allocations[0].Demand.MemoryBytes++
	in.Candidates[0].Placement.Transfers[0].Bytes++
	in.Candidates[0].Topology.Hierarchy[0] = "mutated-hierarchy"

	after, err := json.Marshal(result.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) || result.Receipt.SelectionID != selectionID {
		t.Fatalf("input mutation changed receipt snapshot:\nbefore=%s\nafter=%s", before, after)
	}
	if err := VerifySelectionReceipt(result.Receipt); err != nil {
		t.Fatalf("verify cloned selection: %v", err)
	}
}

func TestVerifySelectionReceiptRejectsContentMutation(t *testing.T) {
	fixture := func(t *testing.T) SelectionReceipt {
		t.Helper()
		in := plannerFixture()
		in.Candidates = []PlanCandidate{sameHostCandidate("verify", strategy(ParallelismTensor, 2), 40*time.Millisecond, 1_000_000)}
		result, err := Plan(in)
		if err != nil {
			t.Fatal(err)
		}
		return result.Receipt
	}
	tests := []struct {
		name string
		edit func(*SelectionReceipt)
	}{
		{"workload", func(receipt *SelectionReceipt) { receipt.Workload.Quality.SequenceLength++ }},
		{"policy", func(receipt *SelectionReceipt) { receipt.Policy.Objectives[0].Direction = Maximize }},
		{"alternative placement", func(receipt *SelectionReceipt) {
			receipt.Alternatives[0].Placement.Allocations[0].Demand.MemoryBytes++
		}},
		{"selected placement", func(receipt *SelectionReceipt) {
			receipt.Selected.Placement.Transfers[0].Bytes++
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := fixture(t)
			tt.edit(&receipt)
			err := VerifySelectionReceipt(receipt)
			if err == nil || !strings.Contains(err.Error(), "selection id mismatch") {
				t.Fatalf("VerifySelectionReceipt() error = %v, want digest mismatch", err)
			}
		})
	}
}

func TestVerifySelectionReceiptAcceptsSelectedAndNoSelectionPostures(t *testing.T) {
	selectedInput := plannerFixture()
	selectedInput.Candidates = []PlanCandidate{singleNodeCandidate("selected", strategy(ParallelismData, 2), 50*time.Millisecond)}
	selected, err := Plan(selectedInput)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifySelectionReceipt(selected.Receipt); err != nil {
		t.Fatalf("verify selected receipt: %v", err)
	}

	noSelectionInput := plannerFixture()
	noSelectionInput.Candidates = []PlanCandidate{
		singleNodeCandidate("a", strategy(ParallelismData, 2), 50*time.Millisecond),
		singleNodeCandidate("b", strategy(ParallelismData, 2), 50*time.Millisecond),
	}
	noSelection, err := Plan(noSelectionInput)
	if err != nil {
		t.Fatal(err)
	}
	if noSelection.Receipt.Disposition != DispositionNoSelection {
		t.Fatalf("disposition = %q, want no selection", noSelection.Receipt.Disposition)
	}
	if err := VerifySelectionReceipt(noSelection.Receipt); err != nil {
		t.Fatalf("verify no-selection receipt: %v", err)
	}
}

func TestVerifySelectionReceiptRejectsInvalidEnvelopeAndPosture(t *testing.T) {
	fixture := func(t *testing.T) SelectionReceipt {
		t.Helper()
		in := plannerFixture()
		in.Candidates = []PlanCandidate{singleNodeCandidate("selected", strategy(ParallelismData, 2), 50*time.Millisecond)}
		result, err := Plan(in)
		if err != nil {
			t.Fatal(err)
		}
		return result.Receipt
	}
	tests := []struct {
		name string
		edit func(*SelectionReceipt)
		want string
	}{
		{"schema", func(receipt *SelectionReceipt) { receipt.Schema = "other/v1" }, "selection schema"},
		{"engine", func(receipt *SelectionReceipt) { receipt.Engine = "external" }, "selection engine"},
		{"workload id", func(receipt *SelectionReceipt) { receipt.WorkloadID = "other" }, "does not match bound workload"},
		{"selected posture", func(receipt *SelectionReceipt) { receipt.Selected = nil }, "has no selected alternative"},
		{"unknown disposition", func(receipt *SelectionReceipt) { receipt.Disposition = "unknown" }, "unknown selection disposition"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := fixture(t)
			tt.edit(&receipt)
			err := VerifySelectionReceipt(receipt)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("VerifySelectionReceipt() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestPlannerParticipatingSLOProvenanceIsEffectiveEvidence(t *testing.T) {
	in := plannerFixture()
	candidate := sameHostCandidate("slo", strategy(ParallelismTensor, 2), 40*time.Millisecond, 1_000_000)
	candidate.Placement.UsefulCompute = measuredCost(40 * time.Millisecond)
	candidate.Placement.Transfers[0].Provenance = ProvenanceMeasured
	candidate.Provenance = ProvenanceEstimated
	candidate.SLO = SLOProjection{Name: "ttft", Unit: "milliseconds", Margin: 10, Modeled: true, Provenance: ProvenanceEstimated}
	in.Candidates = []PlanCandidate{candidate}
	in.Policy.Objectives = []Objective{{Dimension: DimensionSLOMargin, Direction: Maximize}}
	in.Policy.SLO = &SLOIdentity{Name: "ttft", Unit: "milliseconds"}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	assertSelected(t, result, candidate.ID)
	if result.Receipt.Selected.Provenance != ProvenanceMixed || !result.Receipt.Selected.Estimated {
		t.Fatalf("estimated participating SLO missing from effective provenance: %+v", result.Receipt.Selected)
	}

	in.Policy.Objectives = []Objective{{Dimension: DimensionLatencySeconds, Direction: Minimize}}
	in.Policy.SLO = nil
	nonparticipating, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	if nonparticipating.Receipt.Selected.Provenance != ProvenanceMeasured || nonparticipating.Receipt.Selected.Estimated {
		t.Fatalf("non-participating SLO changed placement provenance: %+v", nonparticipating.Receipt.Selected)
	}
}

func TestPlannerInfeasibleAlternativeKeepsConservativeProvenance(t *testing.T) {
	in := plannerFixture()
	candidate := singleNodeCandidate("infeasible", strategy(ParallelismData, 2), 50*time.Millisecond)
	candidate.Placement.Allocations[0].Demand.MemoryBytes = 300 * gib
	candidate.Provenance = ProvenanceEstimated
	in.Candidates = []PlanCandidate{candidate}

	result, err := Plan(in)
	if err != nil {
		t.Fatal(err)
	}
	if result.Receipt.Selected != nil || len(result.Receipt.Alternatives) != 1 {
		t.Fatalf("infeasible selection posture = %+v", result.Receipt)
	}
	alternative := result.Receipt.Alternatives[0]
	if alternative.Metrics != nil || alternative.Provenance != ProvenanceEstimated || !alternative.Estimated {
		t.Fatalf("infeasible provenance = %+v, want conservative estimate without metrics", alternative)
	}
}

func plannerFixture() PlannerInput {
	return PlannerInput{
		Workload: Workload{
			ID: "qwen38-decode", Units: 1, Unit: "request",
			Quality: QualityEnvelope{ModelID: "Qwen3.8-digest", Precision: "bf16", SequenceLength: 4096, BatchSize: 1, Target: "matched-token-output"},
		},
		Topology: Topology{
			Domains: []Domain{{ID: "host-local"}, {ID: "host-remote"}},
			Nodes: []Node{
				{ID: "cpu", DomainID: "host-local", Capacity: Capacity{MemoryBytes: 256 * gib, ComputeUnits: 64}},
				{ID: "gpu0", DomainID: "host-local", Capacity: Capacity{MemoryBytes: 80 * gib, ComputeUnits: 100}},
				{ID: "gpu1", DomainID: "host-local", Capacity: Capacity{MemoryBytes: 80 * gib, ComputeUnits: 100}},
				{ID: "remote", DomainID: "host-remote", Capacity: Capacity{MemoryBytes: 80 * gib, ComputeUnits: 100}},
			},
			Links: []Link{
				{ID: "gpu0-gpu1", FromNode: "gpu0", ToNode: "gpu1", Latency: time.Millisecond, BandwidthBytesPerSecond: 100_000_000_000, MonetaryUSDPerByte: modeledZero(), EnergyJoulesPerByte: modeledZero()},
				{ID: "gpu1-remote", FromNode: "gpu1", ToNode: "remote", Latency: 5 * time.Millisecond, BandwidthBytesPerSecond: 1_000_000_000, MonetaryUSDPerByte: modeledZero(), EnergyJoulesPerByte: modeledZero()},
				{ID: "cpu-gpu0", FromNode: "cpu", ToNode: "gpu0", Latency: 500 * time.Microsecond, BandwidthBytesPerSecond: 20_000_000_000, MonetaryUSDPerByte: modeledZero(), EnergyJoulesPerByte: modeledZero()},
			},
		},
		Reference: Placement{
			Name:          "single-host-reference",
			Allocations:   []Allocation{{NodeID: "cpu", Demand: Capacity{MemoryBytes: 120 * gib, ComputeUnits: 32}}},
			UsefulCompute: estimatedCost(120 * time.Millisecond),
		},
		Policy: OperatorPolicy{
			Objectives: []Objective{{Dimension: DimensionLatencySeconds, Direction: Minimize}},
			Resolution: ResolveUnique, TieBreak: TieBreakRefuse,
			EstimatedCrossDomain: RefuseEstimatedCrossDomain,
		},
	}
}

func strategy(kind ParallelismKind, degree int) ParallelismStrategy {
	s := ParallelismStrategy{
		Kind: kind, TensorDegree: 1, PipelineStages: 1, ExpertDegree: 1,
		DataReplicas: 1, SequenceDegree: 1, ContextDegree: 1,
	}
	switch kind {
	case ParallelismTensor:
		s.TensorDegree = degree
	case ParallelismPipeline:
		s.PipelineStages = degree
	case ParallelismExpert:
		s.ExpertDegree = degree
	case ParallelismData:
		s.DataReplicas = degree
	case ParallelismSequence:
		s.SequenceDegree = degree
	case ParallelismContext:
		s.ContextDegree = degree
	case ParallelismHybrid:
		// The fixture's hybrid strategy spans two decomposition axes, matching
		// validateStrategy's requirement that hybrid never aliases a single axis.
		s.TensorDegree = degree
		s.PipelineStages = degree
	}
	return s
}

func sameHostCandidate(id string, s ParallelismStrategy, compute time.Duration, transferBytes uint64) PlanCandidate {
	return PlanCandidate{
		ID: id, Engine: NativeEngine, Strategy: s,
		Placement: Placement{
			Name: id + "-placement",
			Allocations: []Allocation{
				{NodeID: "gpu0", Demand: Capacity{MemoryBytes: 40 * gib, ComputeUnits: 40}},
				{NodeID: "gpu1", Demand: Capacity{MemoryBytes: 40 * gib, ComputeUnits: 40}},
			},
			UsefulCompute: estimatedCost(compute),
			Transfers:     []Transfer{{LinkID: "gpu0-gpu1", Messages: 1, Bytes: transferBytes, Provenance: ProvenanceEstimated}},
		},
		Topology:   TopologyIntent{Hierarchy: []string{"accelerator", "host"}, Rationale: "traffic stays within one host"},
		Rationale:  "compare useful compute with explicit same-host communication tax",
		Provenance: ProvenanceEstimated,
	}
}

func singleNodeCandidate(id string, s ParallelismStrategy, compute time.Duration) PlanCandidate {
	return PlanCandidate{
		ID: id, Engine: NativeEngine, Strategy: s,
		Placement: Placement{
			Name:          id + "-placement",
			Allocations:   []Allocation{{NodeID: "gpu0", Demand: Capacity{MemoryBytes: 60 * gib, ComputeUnits: 60}}},
			UsefulCompute: estimatedCost(compute),
		},
		Topology:   TopologyIntent{Hierarchy: []string{"device", "host"}, Rationale: "one device inside one host"},
		Rationale:  "single-device comparison candidate",
		Provenance: ProvenanceEstimated,
	}
}

func hybridCandidate(id string, remoteBytes uint64) PlanCandidate {
	return PlanCandidate{
		ID: id, Engine: NativeEngine,
		Strategy: ParallelismStrategy{Kind: ParallelismHybrid, TensorDegree: 2, PipelineStages: 2, ExpertDegree: 1, DataReplicas: 1, SequenceDegree: 1, ContextDegree: 1},
		Placement: Placement{
			Name: id + "-placement",
			Allocations: []Allocation{
				{NodeID: "gpu0", Demand: Capacity{MemoryBytes: 30 * gib, ComputeUnits: 30}},
				{NodeID: "gpu1", Demand: Capacity{MemoryBytes: 30 * gib, ComputeUnits: 30}},
				{NodeID: "remote", Demand: Capacity{MemoryBytes: 30 * gib, ComputeUnits: 30}},
			},
			UsefulCompute: estimatedCost(35 * time.Millisecond),
			Transfers: []Transfer{
				{LinkID: "gpu0-gpu1", Messages: 1, Bytes: 1_000_000_000, Provenance: ProvenanceMeasured},
				{LinkID: "gpu1-remote", Messages: 1, Bytes: remoteBytes, Provenance: ProvenanceMeasured},
			},
		},
		Topology:   TopologyIntent{Hierarchy: []string{"host", "rack"}, Rationale: "cross-domain hierarchy is explicit"},
		Rationale:  "hybrid tensor and pipeline decomposition",
		Provenance: ProvenanceEstimated,
	}
}

func modeledEstimatedCost(d time.Duration, usd, joules float64) ComponentCost {
	return ComponentCost{
		Latency: d, Cycle: d,
		MonetaryUSD:  ModeledValue{Value: usd, Modeled: true},
		EnergyJoules: ModeledValue{Value: joules, Modeled: true},
		Provenance:   ProvenanceEstimated,
	}
}

func measuredCost(d time.Duration) ComponentCost {
	return ComponentCost{Latency: d, Cycle: d, Provenance: ProvenanceMeasured}
}

func modeledZero() ModeledValue {
	return ModeledValue{Modeled: true}
}

func assertSelected(t *testing.T, result PlanResult, want string) {
	t.Helper()
	if result.Receipt.Schema != SelectionReceiptSchema || result.Receipt.SelectionID == "" || result.Receipt.Engine != NativeEngine || result.Receipt.Fallback != "none" {
		t.Fatalf("invalid receipt identity: %+v", result.Receipt)
	}
	if result.Receipt.Selected == nil || result.Receipt.Selected.CandidateID != want {
		t.Fatalf("selected = %+v, want %q; receipt=%+v", result.Receipt.Selected, want, result.Receipt)
	}
}
