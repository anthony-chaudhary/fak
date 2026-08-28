package newmodel

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCompileNativeObligationGraphQwen38DeterministicWitness(t *testing.T) {
	packet, err := CompileManifest(fixture(t, "qwen38-valid.json"))
	if err != nil {
		t.Fatal(err)
	}
	envelope := nativeLaunchEnvelopeFixture(t)
	first, err := CompileNativeObligationGraph(packet, envelope)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileNativeObligationGraph(packet, envelope)
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, err := MarshalNativeObligationGraph(first)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := MarshalNativeObligationGraph(second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatal("same packet/envelope produced different graph bytes")
	}
	if first.Engine != "fak-native" || first.ExternalRuntimeFallback {
		t.Fatalf("unsafe graph identity: %+v", first)
	}
	var previous NativeObligationGraph
	if err := json.Unmarshal(nativeObligationFixture(t, "qwen38-graph.json"), &previous); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(requiredNativeObligations(first.Nodes), requiredNativeObligations(previous.Nodes)) {
		t.Fatal("launch admission changed required correctness obligations")
	}
	seenFusion := false
	seenAdmittedCandidate := false
	seenRejectedCandidate := false
	for _, node := range first.Nodes {
		seenFusion = seenFusion || node.Class == NativeObligationFusion
		if seenFusion && node.Class == NativeObligationRequired {
			t.Fatalf("required correctness node %q follows optional fusion", node.ID)
		}
		if node.Class == NativeObligationRequired && node.LaunchAdmission != nil {
			t.Fatalf("required correctness node %q acquired an optional launch admission", node.ID)
		}
		if node.Class == NativeObligationFusion && (node.Reason == "" || node.Oracle.ID == "" || node.Oracle.Reason == "" || node.Backend.Engine != "fak-native" || node.Backend.Platform == "" || node.Backend.Backend == "" || node.Backend.Reason == "" || node.MemoryLayout.Reason == "" || node.PromotionWitness.ID == "" || node.PromotionWitness.Reason == "") {
			t.Fatalf("candidate lacks reason-bearing obligations: %+v", node)
		}
		if node.Class == NativeObligationFusion && node.LaunchAdmission == nil {
			t.Fatalf("candidate lacks launch admission: %+v", node)
		}
		if node.Class == NativeObligationFusion && node.Eligible {
			seenAdmittedCandidate = true
			if !node.LaunchAdmission.Admitted || node.LaunchAdmission.Reason != NativeLaunchAdmitted || node.LaunchAdmission.Path != NativeLaunchPathFusion {
				t.Fatalf("eligible candidate has contradictory launch decision: %+v", node)
			}
		}
		if node.Class == NativeObligationFusion && !node.Eligible {
			seenRejectedCandidate = true
			if len(node.Blockers) == 0 || node.LaunchAdmission.Admitted || node.LaunchAdmission.Path != NativeLaunchPathCorrectness || node.LaunchAdmission.Engine != "fak-native" {
				t.Fatalf("rejected candidate lacks a deterministic blocker: %+v", node)
			}
		}
	}
	if !seenAdmittedCandidate {
		t.Fatal("fixture did not witness an admitted launch boundary")
	}
	if !seenRejectedCandidate {
		t.Fatal("fixture did not witness a reason-bearing rejected fusion candidate")
	}
}

func TestNativeFusionLaunchAdmissionMatchesScalarOracle(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*nativeLaunchAdmissionCase)
	}{
		{name: "admitted-boundary"},
		{name: "domain-invalid", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.EmptyBatchPolicy = "unknown" }},
		{name: "empty-batch", mutate: func(c *nativeLaunchAdmissionCase) {
			c.domain.Dimensions[0].Min = 0
			c.domain.Dimensions[0].Value = 0
		}},
		{name: "unknown-dimension", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.Dimensions[1].Known = false }},
		{name: "unbounded-dimension", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.Dimensions[1].Bounded = false }},
		{name: "contradictory-range", mutate: func(c *nativeLaunchAdmissionCase) {
			c.domain.Dimensions[1].Min = 129
			c.domain.Dimensions[1].Max = 128
		}},
		{name: "shape-out-of-range", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.Dimensions[1].Value = 160 }},
		{name: "non-divisible", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.Dimensions[1].Value = 127 }},
		{name: "grid-illegal", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.Grid[0]++ }},
		{name: "block-illegal", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.Block[0]++ }},
		{name: "workspace-unbounded", mutate: func(c *nativeLaunchAdmissionCase) { c.domain.WorkspaceBounded = false }},
		{name: "workspace-exceeds-envelope", mutate: func(c *nativeLaunchAdmissionCase) { c.residentBytes = 900 }},
		{name: "overflow", mutate: func(c *nativeLaunchAdmissionCase) {
			c.domain.Grid = [3]uint64{math.MaxUint64, 2, 1}
			c.limits.MaxGrid = c.domain.Grid
		}},
		{name: "missing-backend-limits", mutate: func(c *nativeLaunchAdmissionCase) { c.limits = nil }},
		{name: "ineligible-optional-fusion", mutate: func(c *nativeLaunchAdmissionCase) { c.optionalFusionEligible = false }},
	}
	covered := make(map[NativeLaunchDecisionReason]bool)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := nativeLaunchAdmissionFixture()
			if test.mutate != nil {
				test.mutate(&candidate)
			}
			got := admitNativeFusionLaunch(
				candidate.optionalFusionEligible,
				candidate.operation,
				candidate.domain,
				candidate.domainCount,
				candidate.limits,
				candidate.platform,
				candidate.backend,
				candidate.residentBytes,
				candidate.memoryEnvelopeBytes,
			)
			wantAdmitted, wantReason := scalarLaunchAdmissionOracle(candidate)
			covered[wantReason] = true
			if got.Admitted != wantAdmitted || got.Reason != wantReason {
				t.Fatalf("production decision = admitted:%t reason:%s detail:%q; scalar oracle = admitted:%t reason:%s", got.Admitted, got.Reason, got.Detail, wantAdmitted, wantReason)
			}
			wantPath := NativeLaunchPathCorrectness
			if wantAdmitted {
				wantPath = NativeLaunchPathFusion
			}
			if got.Engine != "fak-native" || got.Phase != "pre-allocation" || got.Path != wantPath || got.Detail == "" {
				t.Fatalf("decision lost native pre-allocation evidence: %+v", got)
			}
		})
	}
	for _, reason := range []NativeLaunchDecisionReason{
		NativeLaunchAdmitted,
		NativeLaunchDomainInvalid,
		NativeLaunchEmptyBatch,
		NativeLaunchUnknownDimension,
		NativeLaunchUnboundedDimension,
		NativeLaunchContradictoryRange,
		NativeLaunchShapeOutOfRange,
		NativeLaunchNonDivisible,
		NativeLaunchGridIllegal,
		NativeLaunchBlockIllegal,
		NativeLaunchWorkspaceUnbounded,
		NativeLaunchWorkspaceExceedsEnvelope,
		NativeLaunchOverflow,
		NativeLaunchBackendLimitsMissing,
		NativeLaunchOptionalFusionIneligible,
	} {
		if !covered[reason] {
			t.Fatalf("closed launch decision reason %s lacks a scalar-oracle witness", reason)
		}
	}
}

func TestNativeObligationGraphRefusesUnknownAndUnsupportedBeforeAllocation(t *testing.T) {
	packet, err := CompileManifest(fixture(t, "qwen38-valid.json"))
	if err != nil {
		t.Fatal(err)
	}
	base := nativeEnvelopeFixture(t)
	tests := []struct {
		name   string
		mutate func(*Packet, *NativeHardwareEnvelope)
		reason RefusalReason
		axis   string
	}{
		{"operation", func(p *Packet, _ *NativeHardwareEnvelope) {
			p.SemanticDeltas[0] = SemanticDelta{Axis: "attention", Value: "gqa"}
		}, RefusalUnknownNativeOperation, "attention"},
		{"layout", func(_ *Packet, e *NativeHardwareEnvelope) { e.WeightLayout = "mystery-layout" }, RefusalUnknownNativeLayout, "hardware_envelope.layout"},
		{"quantization", func(_ *Packet, e *NativeHardwareEnvelope) { e.Quantization = "fp6" }, RefusalUnknownNativeQuantization, "hardware_envelope.quantization"},
		{"platform-backend", func(_ *Packet, e *NativeHardwareEnvelope) { e.Platform = "darwin/arm64" }, RefusalUnsupportedNativeCombination, "hardware_envelope"},
		{"memory-budget", func(_ *Packet, e *NativeHardwareEnvelope) { e.MemoryBudgetBytes = 1 }, RefusalUnsupportedNativeCombination, "hardware_envelope.memory_budget_bytes"},
		{"known-unsupported", func(_ *Packet, e *NativeHardwareEnvelope) {
			e.Backend, e.Platform, e.WeightLayout, e.StateResidency = "cpu", "linux/amd64", "row-major", "host"
		}, RefusalUnsupportedNativeCombination, "hardware_envelope"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidatePacket := packet
			candidatePacket.SemanticDeltas = append([]SemanticDelta(nil), packet.SemanticDeltas...)
			envelope := base
			test.mutate(&candidatePacket, &envelope)
			_, err := CompileNativeObligationGraph(candidatePacket, envelope)
			var refusal *Refusal
			if !errors.As(err, &refusal) || refusal.Reason != test.reason || refusal.Axis != test.axis || refusal.Phase != "pre-allocation" {
				t.Fatalf("refusal = %#v, err = %v; want %s on %s", refusal, err, test.reason, test.axis)
			}
		})
	}
}

func TestNativeHardwareEnvelopeClosedWorldAndEngineFence(t *testing.T) {
	raw := nativeObligationFixture(t, "qwen38-envelope.json")
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatal(err)
	}
	object["unknown"] = true
	mutated, _ := json.Marshal(object)
	if _, err := ParseNativeHardwareEnvelope(mutated); err == nil {
		t.Fatal("unknown hardware envelope field admitted")
	}
	packet, err := CompileManifest(fixture(t, "qwen38-valid.json"))
	if err != nil {
		t.Fatal(err)
	}
	envelope := nativeEnvelopeFixture(t)
	envelope.Engine = "external"
	_, err = CompileNativeObligationGraph(packet, envelope)
	var refusal *Refusal
	if !errors.As(err, &refusal) || refusal.Reason != RefusalNativeEngineMismatch || refusal.Phase != "pre-allocation" {
		t.Fatalf("engine refusal = %#v, err = %v", refusal, err)
	}
}

func nativeEnvelopeFixture(t *testing.T) NativeHardwareEnvelope {
	t.Helper()
	envelope, err := ParseNativeHardwareEnvelope(nativeObligationFixture(t, "qwen38-envelope.json"))
	if err != nil {
		t.Fatal(err)
	}
	return envelope
}

func nativeLaunchEnvelopeFixture(t *testing.T) NativeHardwareEnvelope {
	t.Helper()
	envelope := nativeEnvelopeFixture(t)
	limits := nativeLaunchLimitsFixture()
	envelope.LaunchLimits = &limits
	envelope.FusionLaunches = []NativeFusionLaunchDomain{
		nativeFusionLaunchFixture("fusion.attention-hybrid"),
		nativeFusionLaunchFixture("fusion.state-hybrid-update"),
	}
	return envelope
}

func requiredNativeObligations(nodes []NativeObligation) []NativeObligation {
	required := make([]NativeObligation, 0, len(nodes))
	for _, node := range nodes {
		if node.Class == NativeObligationRequired {
			required = append(required, node)
		}
	}
	return required
}

type nativeLaunchAdmissionCase struct {
	optionalFusionEligible bool
	operation              string
	domain                 NativeFusionLaunchDomain
	domainCount            int
	limits                 *NativeBackendLaunchLimits
	platform               string
	backend                string
	residentBytes          uint64
	memoryEnvelopeBytes    uint64
}

func nativeLaunchAdmissionFixture() nativeLaunchAdmissionCase {
	limits := nativeLaunchLimitsFixture()
	return nativeLaunchAdmissionCase{
		optionalFusionEligible: true,
		operation:              "fusion.attention-hybrid",
		domain:                 nativeFusionLaunchFixture("fusion.attention-hybrid"),
		domainCount:            1,
		limits:                 &limits,
		platform:               "linux/amd64+nvidia-a100",
		backend:                "cuda",
		residentBytes:          512,
		memoryEnvelopeBytes:    1024,
	}
}

func nativeLaunchLimitsFixture() NativeBackendLaunchLimits {
	return NativeBackendLaunchLimits{
		Platform:           "linux/amd64+nvidia-a100",
		Backend:            "cuda",
		MaxGrid:            [3]uint64{8, 4, 2},
		MaxBlock:           [3]uint64{8, 4, 2},
		MaxThreadsPerBlock: 64,
		MaxWorkspaceBytes:  512 * 1024,
	}
}

func nativeFusionLaunchFixture(operation string) NativeFusionLaunchDomain {
	return NativeFusionLaunchDomain{
		Operation:        operation,
		EmptyBatchPolicy: NativeLaunchEmptyBatchPolicyReject,
		Dimensions: []NativeLaunchDimension{
			{Name: "batch", Known: true, Bounded: true, Value: 4, Min: 1, Max: 4, DivisibleBy: 1},
			{Name: "tokens", Known: true, Bounded: true, Value: 128, Min: 1, Max: 128, DivisibleBy: 32},
			{Name: "hidden", Known: true, Bounded: true, Value: 4096, Min: 4096, Max: 4096, DivisibleBy: 128},
		},
		Grid:               [3]uint64{8, 4, 2},
		Block:              [3]uint64{8, 4, 2},
		WorkspaceBounded:   true,
		PeakWorkspaceBytes: 256,
	}
}

// scalarLaunchAdmissionOracle intentionally restates the admission rules
// without calling production validators or arithmetic helpers.
func scalarLaunchAdmissionOracle(candidate nativeLaunchAdmissionCase) (bool, NativeLaunchDecisionReason) {
	if !candidate.optionalFusionEligible {
		return false, NativeLaunchOptionalFusionIneligible
	}
	if candidate.domainCount != 1 || candidate.operation == "" || candidate.domain.Operation != candidate.operation {
		return false, NativeLaunchDomainInvalid
	}
	limits := candidate.limits
	if limits == nil || limits.Platform == "" || limits.Backend == "" || limits.Platform != candidate.platform || limits.Backend != candidate.backend || limits.MaxThreadsPerBlock == 0 || limits.MaxWorkspaceBytes == 0 || limits.MaxGrid[0] == 0 || limits.MaxGrid[1] == 0 || limits.MaxGrid[2] == 0 || limits.MaxBlock[0] == 0 || limits.MaxBlock[1] == 0 || limits.MaxBlock[2] == 0 {
		return false, NativeLaunchBackendLimitsMissing
	}
	if candidate.domain.EmptyBatchPolicy != NativeLaunchEmptyBatchPolicyReject || len(candidate.domain.Dimensions) == 0 {
		return false, NativeLaunchDomainInvalid
	}
	seen := make(map[string]bool, len(candidate.domain.Dimensions))
	for _, dimension := range candidate.domain.Dimensions {
		if dimension.Name == "" {
			return false, NativeLaunchUnknownDimension
		}
		if seen[dimension.Name] {
			return false, NativeLaunchDomainInvalid
		}
		seen[dimension.Name] = true
		if !dimension.Known {
			return false, NativeLaunchUnknownDimension
		}
		if !dimension.Bounded {
			return false, NativeLaunchUnboundedDimension
		}
		if dimension.Min > dimension.Max {
			return false, NativeLaunchContradictoryRange
		}
		if dimension.Name == "batch" && dimension.Value == 0 {
			return false, NativeLaunchEmptyBatch
		}
		if dimension.Value < dimension.Min || dimension.Value > dimension.Max {
			return false, NativeLaunchShapeOutOfRange
		}
		if dimension.DivisibleBy == 0 || dimension.Value%dimension.DivisibleBy != 0 {
			return false, NativeLaunchNonDivisible
		}
	}
	if !seen["batch"] {
		return false, NativeLaunchUnknownDimension
	}
	gridSize := uint64(1)
	for axis, value := range candidate.domain.Grid {
		if value == 0 || value > limits.MaxGrid[axis] {
			return false, NativeLaunchGridIllegal
		}
		if gridSize > math.MaxUint64/value {
			return false, NativeLaunchOverflow
		}
		gridSize *= value
	}
	blockSize := uint64(1)
	for axis, value := range candidate.domain.Block {
		if value == 0 || value > limits.MaxBlock[axis] {
			return false, NativeLaunchBlockIllegal
		}
		if blockSize > math.MaxUint64/value {
			return false, NativeLaunchOverflow
		}
		blockSize *= value
	}
	if blockSize > limits.MaxThreadsPerBlock {
		return false, NativeLaunchBlockIllegal
	}
	if gridSize > math.MaxUint64/blockSize {
		return false, NativeLaunchOverflow
	}
	if !candidate.domain.WorkspaceBounded {
		return false, NativeLaunchWorkspaceUnbounded
	}
	if candidate.domain.PeakWorkspaceBytes > limits.MaxWorkspaceBytes {
		return false, NativeLaunchWorkspaceExceedsEnvelope
	}
	if candidate.residentBytes > math.MaxUint64-candidate.domain.PeakWorkspaceBytes {
		return false, NativeLaunchOverflow
	}
	if candidate.residentBytes+candidate.domain.PeakWorkspaceBytes > candidate.memoryEnvelopeBytes {
		return false, NativeLaunchWorkspaceExceedsEnvelope
	}
	return true, NativeLaunchAdmitted
}

func nativeObligationFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "native-obligations", name))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
