package newmodel

import (
	"encoding/json"
	"math"
	"testing"
)

func TestFusionLaunchDomainAdmissionMatchesIndependentScalarOracle(t *testing.T) {
	packet, err := CompileManifest(fixture(t, "qwen38-valid.json"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name            string
		operation       string
		backendEligible bool
		mutate          func(*NativeHardwareEnvelope)
		wantAdmitted    bool
		wantReason      string
	}{
		{name: "admitted-boundary", operation: "fusion.attention-hybrid", backendEligible: true, wantAdmitted: true, wantReason: "launch-admitted"},
		{name: "empty-batch", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			launchDimension(e, "batch").Value = 0
		}, wantReason: "empty-batch"},
		{name: "oversized", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			launchDimension(e, "tokens").Value = 65
		}, wantReason: "launch-dimension-oversized"},
		{name: "non-divisible", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			launchDimension(e, "tokens").Value = 63
		}, wantReason: "launch-dimension-non-divisible"},
		{name: "illegal-grid", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.FusionLaunchDomains[0].GridBlocks++
		}, wantReason: "launch-grid-illegal"},
		{name: "illegal-block", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.FusionLaunchDomains[0].BlockThreads++
		}, wantReason: "launch-block-illegal"},
		{name: "workspace-exceeds-limit", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.FusionLaunchDomains[0].PeakWorkspaceBytes++
		}, wantReason: "launch-workspace-exceeds-limit"},
		{name: "workspace-unbounded", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.FusionLaunchDomains[0].WorkspaceBounded = false
		}, wantReason: "launch-workspace-unbounded"},
		{name: "missing-limit", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.LaunchLimits.MaxGridBlocks = 0
		}, wantReason: "launch-limits-missing"},
		{name: "contradictory-range", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			launchDimension(e, "tokens").Min = 65
		}, wantReason: "launch-dimension-contradictory"},
		{name: "unknown-dimension", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			launchDimension(e, "tokens").Name = "mystery"
		}, wantReason: "launch-dimension-unknown"},
		{name: "shape-overflow", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.LaunchLimits.MaxDimensionValue = math.MaxUint64
			dimension := launchDimension(e, "batch")
			dimension.Value, dimension.Min, dimension.Max = math.MaxUint64, math.MaxUint64, math.MaxUint64
		}, wantReason: "launch-shape-overflow"},
		{name: "domain-missing", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.FusionLaunchDomains = nil
		}, wantReason: "launch-domain-missing"},
		{name: "empty-policy-unknown", operation: "fusion.attention-hybrid", backendEligible: true, mutate: func(e *NativeHardwareEnvelope) {
			e.FusionLaunchDomains[0].EmptyBatchPolicy = "launch-zero-grid"
		}, wantReason: "empty-batch-policy-unknown"},
		{name: "fusion-ineligible", operation: "fusion.ffn-swiglu", backendEligible: false, wantReason: "fusion-ineligible"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			envelope := cloneLaunchEnvelope(nativeEnvelopeFixture(t))
			if test.mutate != nil {
				test.mutate(&envelope)
			}
			oracleAdmitted, oracleReason := scalarLaunchDomainOracle(test.operation, test.backendEligible, envelope)
			if oracleAdmitted != test.wantAdmitted || oracleReason != test.wantReason {
				t.Fatalf("scalar oracle = (%t, %q), want (%t, %q)", oracleAdmitted, oracleReason, test.wantAdmitted, test.wantReason)
			}

			graph, err := CompileNativeObligationGraph(packet, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if graph.Engine != "fak-native" || graph.ExternalRuntimeFallback {
				t.Fatalf("unsafe graph identity: %+v", graph)
			}
			candidate := obligationByOperation(t, graph, test.operation)
			if candidate.LaunchAdmission == nil {
				t.Fatal("optional fusion has no launch admission")
			}
			if candidate.Eligible != oracleAdmitted || string(candidate.LaunchAdmission.ReasonCode) != oracleReason {
				t.Fatalf("production admission = (%t, %q), scalar oracle = (%t, %q): %+v", candidate.Eligible, candidate.LaunchAdmission.ReasonCode, oracleAdmitted, oracleReason, candidate.LaunchAdmission)
			}
			if candidate.LaunchAdmission.Reason == "" {
				t.Fatal("launch admission lacks a reason")
			}
			if !candidate.Eligible && (len(candidate.Blockers) != 1 || candidate.Blockers[0] != oracleReason) {
				t.Fatalf("closed blocker = %v, want [%s]", candidate.Blockers, oracleReason)
			}
			for _, node := range graph.Nodes {
				if node.Class == NativeObligationRequired && (!node.Eligible || node.LaunchAdmission != nil) {
					t.Fatalf("launch rejection changed required correctness node: %+v", node)
				}
			}
		})
	}
}

func TestFusionLaunchDomainJSONIsClosed(t *testing.T) {
	var object map[string]any
	if err := json.Unmarshal(nativeObligationFixture(t, "qwen38-envelope.json"), &object); err != nil {
		t.Fatal(err)
	}
	limits := object["launch_limits"].(map[string]any)
	limits["unknown_limit"] = 1
	raw, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseNativeHardwareEnvelope(raw); err == nil {
		t.Fatal("unknown nested launch limit admitted")
	}
}

// scalarLaunchDomainOracle intentionally restates the arithmetic without
// calling the production admission helper or sharing its decision branches.
func scalarLaunchDomainOracle(operation string, backendEligible bool, envelope NativeHardwareEnvelope) (bool, string) {
	if !backendEligible {
		return false, "fusion-ineligible"
	}
	limits := envelope.LaunchLimits
	if limits.Platform == "" || limits.Backend == "" || limits.MaxDimensionValue == 0 || limits.MaxGridBlocks == 0 || limits.MaxBlockThreads == 0 || limits.MaxWorkspaceBytes == 0 {
		return false, "launch-limits-missing"
	}
	if limits.Platform != envelope.Platform || limits.Backend != envelope.Backend {
		return false, "launch-limits-mismatch"
	}
	var domain NativeFusionLaunchDomain
	found := 0
	for _, candidate := range envelope.FusionLaunchDomains {
		if candidate.Operation == operation {
			domain = candidate
			found++
		}
	}
	if found == 0 {
		return false, "launch-domain-missing"
	}
	if found != 1 {
		return false, "launch-domain-contradictory"
	}
	if domain.EmptyBatchPolicy != "native-correctness-path" {
		return false, "empty-batch-policy-unknown"
	}
	if len(domain.Dimensions) == 0 {
		return false, "launch-dimension-unknown"
	}
	seen := map[string]bool{}
	batchSeen, emptyBatch := false, false
	for _, dimension := range domain.Dimensions {
		known := dimension.Name == "batch" || dimension.Name == "channels" || dimension.Name == "tokens"
		if !known || seen[dimension.Name] || dimension.Min == 0 || dimension.Max == 0 || dimension.DivisibleBy == 0 {
			return false, "launch-dimension-unknown"
		}
		seen[dimension.Name] = true
		if dimension.Min > dimension.Max {
			return false, "launch-dimension-contradictory"
		}
		if dimension.Name == "batch" {
			batchSeen, emptyBatch = true, dimension.Value == 0
		} else if dimension.Value == 0 {
			return false, "launch-dimension-unknown"
		}
	}
	if !batchSeen {
		return false, "launch-dimension-unknown"
	}
	if emptyBatch {
		return false, "empty-batch"
	}
	total := uint64(1)
	for _, dimension := range domain.Dimensions {
		if dimension.Max > limits.MaxDimensionValue || dimension.Value < dimension.Min || dimension.Value > dimension.Max {
			return false, "launch-dimension-oversized"
		}
		if dimension.Value%dimension.DivisibleBy != 0 {
			return false, "launch-dimension-non-divisible"
		}
		if total > math.MaxUint64/dimension.Value {
			return false, "launch-shape-overflow"
		}
		total *= dimension.Value
	}
	if domain.BlockThreads == 0 || domain.BlockThreads > limits.MaxBlockThreads {
		return false, "launch-block-illegal"
	}
	if domain.GridBlocks == 0 || domain.GridBlocks > limits.MaxGridBlocks {
		return false, "launch-grid-illegal"
	}
	if total%domain.BlockThreads != 0 {
		return false, "launch-dimension-non-divisible"
	}
	if total/domain.BlockThreads != domain.GridBlocks {
		return false, "launch-grid-illegal"
	}
	if !domain.WorkspaceBounded {
		return false, "launch-workspace-unbounded"
	}
	workspaceLimit := limits.MaxWorkspaceBytes
	if envelope.MemoryBudgetBytes < workspaceLimit {
		workspaceLimit = envelope.MemoryBudgetBytes
	}
	if domain.PeakWorkspaceBytes > workspaceLimit {
		return false, "launch-workspace-exceeds-limit"
	}
	return true, "launch-admitted"
}

func cloneLaunchEnvelope(envelope NativeHardwareEnvelope) NativeHardwareEnvelope {
	clone := envelope
	clone.FusionLaunchDomains = append([]NativeFusionLaunchDomain(nil), envelope.FusionLaunchDomains...)
	for i := range clone.FusionLaunchDomains {
		clone.FusionLaunchDomains[i].Dimensions = append([]NativeLaunchDimension(nil), envelope.FusionLaunchDomains[i].Dimensions...)
	}
	return clone
}

func launchDimension(envelope *NativeHardwareEnvelope, name string) *NativeLaunchDimension {
	for i := range envelope.FusionLaunchDomains[0].Dimensions {
		if envelope.FusionLaunchDomains[0].Dimensions[i].Name == name {
			return &envelope.FusionLaunchDomains[0].Dimensions[i]
		}
	}
	panic("fixture launch dimension not found: " + name)
}

func obligationByOperation(t *testing.T, graph NativeObligationGraph, operation string) NativeObligation {
	t.Helper()
	for _, node := range graph.Nodes {
		if node.Class == NativeObligationFusion && node.Operation == operation {
			return node
		}
	}
	t.Fatalf("optional fusion operation %q not found", operation)
	return NativeObligation{}
}
