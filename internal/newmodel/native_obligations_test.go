package newmodel

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestCompileNativeObligationGraphQwen38DeterministicWitness(t *testing.T) {
	packet, err := CompileManifest(fixture(t, "qwen38-valid.json"))
	if err != nil {
		t.Fatal(err)
	}
	envelope := nativeEnvelopeFixture(t)
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
	want, err := os.ReadFile(filepath.Join("testdata", "native-obligations", "qwen38-graph.json"))
	if err != nil {
		t.Fatal(err)
	}
	// Normalize only checkout transport; graph emission remains strict deterministic LF bytes.
	want = bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n"))
	if !bytes.Equal(firstJSON, want) {
		t.Fatalf("golden graph drifted; got:\n%s", firstJSON)
	}
	if first.Engine != "fak-native" || first.ExternalRuntimeFallback {
		t.Fatalf("unsafe graph identity: %+v", first)
	}
	seenFusion := false
	seenRejectedCandidate := false
	for _, node := range first.Nodes {
		seenFusion = seenFusion || node.Class == NativeObligationFusion
		if seenFusion && node.Class == NativeObligationRequired {
			t.Fatalf("required correctness node %q follows optional fusion", node.ID)
		}
		if node.Class == NativeObligationFusion && (node.Reason == "" || node.Oracle.ID == "" || node.Oracle.Reason == "" || node.Backend.Engine != "fak-native" || node.Backend.Platform == "" || node.Backend.Backend == "" || node.Backend.Reason == "" || node.MemoryLayout.Reason == "" || node.PromotionWitness.ID == "" || node.PromotionWitness.Reason == "") {
			t.Fatalf("candidate lacks reason-bearing obligations: %+v", node)
		}
		if node.Class == NativeObligationFusion && !node.Eligible {
			seenRejectedCandidate = true
			if len(node.Blockers) == 0 {
				t.Fatalf("rejected candidate lacks a deterministic blocker: %+v", node)
			}
		}
	}
	if !seenRejectedCandidate {
		t.Fatal("fixture did not witness a reason-bearing rejected fusion candidate")
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

func nativeObligationFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "native-obligations", name))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
