package issue8846qwen38metalgdn

import (
	"encoding/json"
	"os"
	"testing"
)

type packet struct {
	Schema                         string
	Verdict                        string
	Reason                         string
	ArtifactSHA256                 string `json:"artifact_sha256"`
	PromptTokens                   int    `json:"prompt_tokens"`
	DecodeTokens                   int    `json:"decode_tokens"`
	RepetitionsPerArm              int    `json:"repetitions_per_arm"`
	Machine                        string
	Engine                         string
	Backend                        string
	ForwardPath                    string `json:"forward_path"`
	Fallback                       bool
	ControlDecodeTokensPerSecond   []float64 `json:"control_decode_tokens_per_second"`
	CandidateDecodeTokensPerSecond []float64 `json:"candidate_decode_tokens_per_second"`
}

func TestQwen38MetalGDNWitness(t *testing.T) {
	data, err := os.ReadFile("packet.json")
	if err != nil {
		t.Fatal(err)
	}
	var p packet
	if err := json.Unmarshal(data, &p); err != nil {
		t.Fatal(err)
	}
	if p.Schema != "fak-qwen38-metal-gdn-resident-witness/v1" {
		t.Fatalf("schema=%q", p.Schema)
	}
	if p.ArtifactSHA256 != "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169" {
		t.Fatalf("artifact=%q", p.ArtifactSHA256)
	}
	if p.PromptTokens != 32 || p.DecodeTokens != 64 || p.RepetitionsPerArm != 3 {
		t.Fatalf("envelope=P%d/T%d reps=%d", p.PromptTokens, p.DecodeTokens, p.RepetitionsPerArm)
	}
	if p.Machine != "sanctioned M3 Pro" || p.Engine != "fak-native" || p.Backend != "metal" || p.ForwardPath != "fak-native/metal/qwen35-gdn-resident-decode-v1" || p.Fallback {
		t.Fatalf("identity=%q/%q/%q path=%q fallback=%v", p.Machine, p.Engine, p.Backend, p.ForwardPath, p.Fallback)
	}
	switch p.Verdict {
	case "REJECT":
		if p.Reason == "" {
			t.Fatal("REJECT requires reason")
		}
	case "KEEP":
		if len(p.ControlDecodeTokensPerSecond) != 3 || len(p.CandidateDecodeTokensPerSecond) != 3 {
			t.Fatal("KEEP requires three receipts per arm")
		}
		for i := range p.CandidateDecodeTokensPerSecond {
			if p.CandidateDecodeTokensPerSecond[i] <= 0 || p.ControlDecodeTokensPerSecond[i] <= 0 {
				t.Fatal("non-positive throughput")
			}
		}
	default:
		t.Fatalf("verdict=%q", p.Verdict)
	}
}
