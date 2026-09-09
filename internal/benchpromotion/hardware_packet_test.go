package benchpromotion

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func validBaselineRequest() HardwarePacketRequest {
	return HardwarePacketRequest{
		Disposition:           DispositionHardware,
		CandidateDigest:       "sha256:1111111111111111111111111111111111111111111111111111111111111111",
		EvidenceDigest:        "sha256:2222222222222222222222222222222222222222222222222222222222222222",
		HardwareClass:         "strix-halo-gfx1151",
		Engine:                "fak-native",
		Model:                 "Qwen/Qwen2.5-Coder-32B-Instruct",
		Workload:              "batch1-ctx8k-gen512",
		QualityGate:           "exact-match-or-better",
		ObjectiveSLO:          "ttft<=40ms,tps>=45",
		AuthorizedBudget:      "$10.00",
		LeaseLane:             "benchpromotion",
		CaptureCommand:        "fak bench run --hardware strix-halo-gfx1151 --workload batch1-ctx8k-gen512",
		ExpectedReceiptSchema: "fak-bench-receipt/1",
	}
}

func TestHardwarePacketDeterministicBytes(t *testing.T) {
	req := validBaselineRequest()

	p1, err := EmitHardwarePacket(req)
	if err != nil {
		t.Fatalf("EmitHardwarePacket failed: %v", err)
	}
	p2, err := EmitHardwarePacket(req)
	if err != nil {
		t.Fatalf("EmitHardwarePacket failed: %v", err)
	}

	if p1.PacketDigest != p2.PacketDigest {
		t.Fatalf("expected identical PacketDigest, got %q vs %q", p1.PacketDigest, p2.PacketDigest)
	}

	b1, err := json.Marshal(p1)
	if err != nil {
		t.Fatalf("json.Marshal(p1) failed: %v", err)
	}
	b2, err := json.Marshal(p2)
	if err != nil {
		t.Fatalf("json.Marshal(p2) failed: %v", err)
	}

	if !bytes.Equal(b1, b2) {
		t.Fatalf("expected byte-identical JSON serialization, got %s vs %s", string(b1), string(b2))
	}

	canon1, err := p1.CanonicalJSON()
	if err != nil {
		t.Fatalf("p1.CanonicalJSON failed: %v", err)
	}
	canon2, err := p2.CanonicalJSON()
	if err != nil {
		t.Fatalf("p2.CanonicalJSON failed: %v", err)
	}
	if !bytes.Equal(canon1, canon2) {
		t.Fatalf("expected byte-identical CanonicalJSON, got %s vs %s", string(canon1), string(canon2))
	}

	if err := VerifyHardwarePacket(p1); err != nil {
		t.Fatalf("VerifyHardwarePacket(p1) failed: %v", err)
	}
	if err := VerifyHardwarePacket(p2); err != nil {
		t.Fatalf("VerifyHardwarePacket(p2) failed: %v", err)
	}
}

func TestHardwarePacketFieldMutationChangesDigest(t *testing.T) {
	baselineReq := validBaselineRequest()
	baselinePacket, err := EmitHardwarePacket(baselineReq)
	if err != nil {
		t.Fatalf("EmitHardwarePacket failed: %v", err)
	}
	if err := VerifyHardwarePacket(baselinePacket); err != nil {
		t.Fatalf("baseline packet failed verification: %v", err)
	}

	fields := []struct {
		name   string
		mutate func(p *HardwareReadyPacket)
	}{
		{"SchemaVersion", func(p *HardwareReadyPacket) { p.SchemaVersion = "fak-hardware-packet/2" }},
		{"CandidateDigest", func(p *HardwareReadyPacket) { p.CandidateDigest += "-mutated" }},
		{"EvidenceDigest", func(p *HardwareReadyPacket) { p.EvidenceDigest += "-mutated" }},
		{"HardwareClass", func(p *HardwareReadyPacket) { p.HardwareClass += "-mutated" }},
		{"Engine", func(p *HardwareReadyPacket) { p.Engine = "fak-native-v2" }},
		{"Model", func(p *HardwareReadyPacket) { p.Model += "-mutated" }},
		{"Workload", func(p *HardwareReadyPacket) { p.Workload += "-mutated" }},
		{"QualityGate", func(p *HardwareReadyPacket) { p.QualityGate += "-mutated" }},
		{"ObjectiveSLO", func(p *HardwareReadyPacket) { p.ObjectiveSLO += "-mutated" }},
		{"AuthorizedBudget", func(p *HardwareReadyPacket) { p.AuthorizedBudget = "$20.00" }},
		{"LeaseLane", func(p *HardwareReadyPacket) { p.LeaseLane += "-mutated" }},
		{"CaptureCommand", func(p *HardwareReadyPacket) { p.CaptureCommand += " --extra" }},
		{"ExpectedReceiptSchema", func(p *HardwareReadyPacket) { p.ExpectedReceiptSchema = "fak-bench-receipt/2" }},
	}

	for _, tc := range fields {
		t.Run(tc.name, func(t *testing.T) {
			mutated := *baselinePacket
			tc.mutate(&mutated)

			newDigest, err := computePacketDigest(&mutated)
			if err != nil {
				t.Fatalf("computePacketDigest failed for mutated field %s: %v", tc.name, err)
			}
			if newDigest == baselinePacket.PacketDigest {
				t.Errorf("mutating %s did not change digest: got %s", tc.name, newDigest)
			}

			if err := VerifyHardwarePacket(&mutated); err == nil {
				t.Errorf("VerifyHardwarePacket unexpectedly succeeded after mutating %s", tc.name)
			}
		})
	}
}

func TestHardwarePacketNonHardwareRefused(t *testing.T) {
	dispositions := []string{
		DispositionAbstain,
		DispositionPruned,
		DispositionSimulationOnly,
		"UNKNOWN",
		"",
	}

	for _, disp := range dispositions {
		t.Run(disp, func(t *testing.T) {
			req := validBaselineRequest()
			req.Disposition = disp

			packet, err := EmitHardwarePacket(req)
			if err == nil {
				t.Fatalf("expected error for disposition %q, got nil error", disp)
			}
			if packet != nil {
				t.Fatalf("expected nil packet for disposition %q, got %+v", disp, packet)
			}
			expectedErrMsg := fmt.Sprintf("benchpromotion: runnable packet refused for non-hardware disposition %q", disp)
			if err.Error() != expectedErrMsg {
				t.Errorf("expected error %q, got %q", expectedErrMsg, err.Error())
			}
		})
	}
}

func TestHardwarePacketRejectsNonFakNative(t *testing.T) {
	engines := []string{
		"llama.cpp",
		"vllm",
		"ollama",
		"onnxruntime",
		"",
	}

	for _, eng := range engines {
		t.Run(eng, func(t *testing.T) {
			req := validBaselineRequest()
			req.Engine = eng

			packet, err := EmitHardwarePacket(req)
			if err == nil {
				t.Fatalf("expected error for engine %q, got nil error", eng)
			}
			if packet != nil {
				t.Fatalf("expected nil packet for engine %q, got %+v", eng, packet)
			}
			expectedErrMsg := fmt.Sprintf("benchpromotion: engine must be fak-native, got %q", eng)
			if err.Error() != expectedErrMsg {
				t.Errorf("expected error %q, got %q", expectedErrMsg, err.Error())
			}
		})
	}
}

func TestHardwarePacketJSONRoundTrip(t *testing.T) {
	req := validBaselineRequest()
	original, err := EmitHardwarePacket(req)
	if err != nil {
		t.Fatalf("EmitHardwarePacket failed: %v", err)
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var decoded HardwareReadyPacket
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if !reflect.DeepEqual(original, &decoded) {
		t.Fatalf("round-trip mismatch:\noriginal: %+v\ndecoded:  %+v", original, &decoded)
	}

	if err := VerifyHardwarePacket(&decoded); err != nil {
		t.Fatalf("VerifyHardwarePacket failed on decoded packet: %v", err)
	}
}

func TestHardwarePacketMissingRequiredFields(t *testing.T) {
	requiredFields := []struct {
		name  string
		clear func(r *HardwarePacketRequest)
	}{
		{"candidate_digest", func(r *HardwarePacketRequest) { r.CandidateDigest = "" }},
		{"evidence_digest", func(r *HardwarePacketRequest) { r.EvidenceDigest = "" }},
		{"hardware_class", func(r *HardwarePacketRequest) { r.HardwareClass = "" }},
		{"model", func(r *HardwarePacketRequest) { r.Model = "" }},
		{"workload", func(r *HardwarePacketRequest) { r.Workload = "" }},
		{"capture_command", func(r *HardwarePacketRequest) { r.CaptureCommand = "" }},
		{"expected_receipt_schema", func(r *HardwarePacketRequest) { r.ExpectedReceiptSchema = "" }},
		{"authorized_budget", func(r *HardwarePacketRequest) { r.AuthorizedBudget = "" }},
		{"lease_lane", func(r *HardwarePacketRequest) { r.LeaseLane = "" }},
	}

	for _, rf := range requiredFields {
		t.Run(rf.name, func(t *testing.T) {
			req := validBaselineRequest()
			rf.clear(&req)

			packet, err := EmitHardwarePacket(req)
			if err == nil {
				t.Fatalf("expected error when %s is missing, got nil", rf.name)
			}
			if packet != nil {
				t.Fatalf("expected nil packet when %s is missing, got %+v", rf.name, packet)
			}
			if !strings.Contains(err.Error(), rf.name) {
				t.Errorf("expected error to mention %q, got %q", rf.name, err.Error())
			}
		})
	}
}

func TestVerifyHardwarePacketValidation(t *testing.T) {
	t.Run("nil packet", func(t *testing.T) {
		if err := VerifyHardwarePacket(nil); err == nil {
			t.Fatal("expected error for nil packet, got nil")
		}
	})

	t.Run("invalid schema version", func(t *testing.T) {
		p, err := EmitHardwarePacket(validBaselineRequest())
		if err != nil {
			t.Fatalf("EmitHardwarePacket failed: %v", err)
		}
		p.SchemaVersion = "invalid-schema"
		p.PacketDigest, _ = computePacketDigest(p)
		if err := VerifyHardwarePacket(p); err == nil {
			t.Fatal("expected error for invalid schema version, got nil")
		}
	})

	t.Run("invalid engine", func(t *testing.T) {
		p, err := EmitHardwarePacket(validBaselineRequest())
		if err != nil {
			t.Fatalf("EmitHardwarePacket failed: %v", err)
		}
		p.Engine = "llama.cpp"
		p.PacketDigest, _ = computePacketDigest(p)
		if err := VerifyHardwarePacket(p); err == nil {
			t.Fatal("expected error for invalid engine, got nil")
		}
	})

	t.Run("missing packet digest", func(t *testing.T) {
		p, err := EmitHardwarePacket(validBaselineRequest())
		if err != nil {
			t.Fatalf("EmitHardwarePacket failed: %v", err)
		}
		p.PacketDigest = ""
		if err := VerifyHardwarePacket(p); err == nil {
			t.Fatal("expected error for missing packet digest, got nil")
		}
	})

	t.Run("corrupted packet digest", func(t *testing.T) {
		p, err := EmitHardwarePacket(validBaselineRequest())
		if err != nil {
			t.Fatalf("EmitHardwarePacket failed: %v", err)
		}
		p.PacketDigest = "0000000000000000000000000000000000000000000000000000000000000000"
		if err := VerifyHardwarePacket(p); err == nil {
			t.Fatal("expected error for corrupted packet digest, got nil")
		}
	})
}
