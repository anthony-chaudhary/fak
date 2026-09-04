package issue9525witness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

const (
	expectedSchema   = "fak.issue-9525-qwen38-sequence-prefill-receipt/1"
	expectedArtifact = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	expectedBytes    = int64(17106775008)
	expectedEngine   = "fak-native"
	expectedBackend  = "metal"
	expectedForward  = "metal/qwen35-gdn-preprojected-sequence-v1"
)

var expectedOrder = []string{
	"control-1",
	"candidate-1",
	"candidate-2",
	"control-2",
	"control-3",
	"candidate-3",
}

type receiptArm struct {
	Name             string             `json:"name"`
	Type             string             `json:"type"`
	SequenceSelector bool               `json:"sequence_selector"`
	ProfileSHA256    string             `json:"profile_sha256"`
	Phases           map[string]float64 `json:"phases"`
	Metal            struct {
		CommandBuffers int     `json:"command_buffers"`
		Encoders       int     `json:"encoders"`
		ResidentBytes  uint64  `json:"resident_bytes"`
		WorkingSet     uint64  `json:"working_set_bytes"`
		DispatchMS     float64 `json:"dispatch_milliseconds"`
		WaitMS         float64 `json:"wait_milliseconds"`
	} `json:"metal"`
	Sequence struct {
		Path                  string  `json:"path,omitempty"`
		Available             bool    `json:"available"`
		SelectorState         string  `json:"selector_state"`
		EvidenceState         string  `json:"evidence_state"`
		Tokens                int     `json:"tokens,omitempty"`
		CommandBuffers        int     `json:"command_buffers,omitempty"`
		Encoders              int     `json:"encoders,omitempty"`
		IntermediateWaits     int     `json:"intermediate_waits"`
		IntermediateReadbacks int     `json:"intermediate_readbacks"`
		TerminalWaits         int     `json:"terminal_waits,omitempty"`
		TerminalReadbacks     int     `json:"terminal_readbacks,omitempty"`
		HostUploadBytes       uint64  `json:"host_upload_bytes,omitempty"`
		HostReadbackBytes     uint64  `json:"host_readback_bytes,omitempty"`
		Committed             bool    `json:"committed,omitempty"`
		CompletedWait         bool    `json:"completed_wait,omitempty"`
		TimingAvailable       bool    `json:"timing_available,omitempty"`
		GPUMilliseconds       float64 `json:"gpu_milliseconds,omitempty"`
		WaitMilliseconds      float64 `json:"wait_milliseconds,omitempty"`
	} `json:"sequence"`
	Fallbacks struct {
		PromisedCPUFallbacks int `json:"promised_cpu_fallbacks"`
	} `json:"fallbacks"`
}

type campaignReceipt struct {
	Schema        string   `json:"schema"`
	Issue         int      `json:"issue"`
	ParentIssue   int      `json:"parent_issue"`
	UmbrellaIssue int      `json:"umbrella_issue"`
	Phase         string   `json:"phase"`
	Verdict       string   `json:"verdict"`
	Order         []string `json:"order"`
	Artifact      struct {
		Model        string `json:"model"`
		Repository   string `json:"repository"`
		Revision     string `json:"revision"`
		File         string `json:"file"`
		Quantization string `json:"quantization"`
		Bytes        int64  `json:"bytes"`
		SHA256       string `json:"sha256"`
	} `json:"artifact"`
	Hardware struct {
		SoC         string `json:"soc"`
		GPUCores    int    `json:"gpu_cores"`
		MemoryBytes uint64 `json:"memory_bytes"`
		MemoryGiB   int    `json:"memory_gib"`
		OS          string `json:"os"`
		Arch        string `json:"arch"`
	} `json:"hardware"`
	Execution struct {
		Engine         string `json:"engine"`
		Backend        string `json:"backend"`
		ForwardPath    string `json:"forward_path"`
		PromptTokens   int    `json:"prompt_tokens"`
		DecodeTokens   int    `json:"decode_tokens"`
		TotalFallbacks int    `json:"total_fallbacks"`
	} `json:"execution"`
	Comparison struct {
		PrefillMedianMS struct {
			Control        float64 `json:"control"`
			Candidate      float64 `json:"candidate"`
			ImprovementPct float64 `json:"improvement_pct"`
		} `json:"prefill_median_ms"`
		FirstTokenMedianMS struct {
			Control        float64 `json:"control"`
			Candidate      float64 `json:"candidate"`
			ImprovementPct float64 `json:"improvement_pct"`
		} `json:"first_token_median_ms"`
		PrefillCommandBuffers struct {
			Control   int `json:"control"`
			Candidate int `json:"candidate"`
		} `json:"prefill_command_buffers"`
		TerminalWaits struct {
			Control   int `json:"control"`
			Candidate int `json:"candidate"`
		} `json:"terminal_waits"`
		IntermediateWaits struct {
			Control   int `json:"control"`
			Candidate int `json:"candidate"`
		} `json:"intermediate_waits"`
	} `json:"comparison"`
	Arms []receiptArm `json:"arms"`
}

func TestIssue9525CampaignReceipt(t *testing.T) {
	data, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatalf("read receipt.json: %v", err)
	}

	var r campaignReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatalf("unmarshal receipt.json: %v", err)
	}

	if r.Schema != expectedSchema {
		t.Errorf("schema = %q, want %q", r.Schema, expectedSchema)
	}
	if r.Issue != 9525 || r.ParentIssue != 9230 || r.UmbrellaIssue != 9430 {
		t.Errorf("issue metadata mismatch: issue=%d parent=%d umbrella=%d", r.Issue, r.ParentIssue, r.UmbrellaIssue)
	}
	if r.Phase != "M2" {
		t.Errorf("phase = %q, want M2", r.Phase)
	}
	if r.Verdict != "KEEP" {
		t.Errorf("verdict = %q, want KEEP", r.Verdict)
	}

	if r.Artifact.SHA256 != expectedArtifact || r.Artifact.Bytes != expectedBytes {
		t.Errorf("artifact mismatch: sha=%s bytes=%d", r.Artifact.SHA256, r.Artifact.Bytes)
	}
	if r.Execution.Engine != expectedEngine || r.Execution.Backend != expectedBackend || r.Execution.ForwardPath != expectedForward {
		t.Errorf("execution metadata mismatch: %+v", r.Execution)
	}
	if r.Execution.TotalFallbacks != 0 {
		t.Errorf("total fallbacks = %d, want 0", r.Execution.TotalFallbacks)
	}
	if r.Hardware.GPUCores != 18 || r.Hardware.MemoryGiB != 36 {
		t.Errorf("hardware mismatch: %+v", r.Hardware)
	}

	if len(r.Arms) != 6 {
		t.Fatalf("arms count = %d, want 6", len(r.Arms))
	}
	if len(r.Order) != 6 {
		t.Fatalf("order count = %d, want 6", len(r.Order))
	}

	for i, wantName := range expectedOrder {
		if r.Order[i] != wantName {
			t.Errorf("order[%d] = %q, want %q", i, r.Order[i], wantName)
		}
		arm := r.Arms[i]
		if arm.Name != wantName {
			t.Errorf("arm[%d].Name = %q, want %q", i, arm.Name, wantName)
		}

		profileFile := filepath.Join(".", arm.Name+".json")
		pData, err := os.ReadFile(profileFile)
		if err != nil {
			t.Errorf("missing profile file %s: %v", profileFile, err)
			continue
		}
		h := sha256.Sum256(pData)
		gotSHA := hex.EncodeToString(h[:])
		if gotSHA != arm.ProfileSHA256 {
			t.Errorf("arm %s profile sha mismatch: got %s, want %s", arm.Name, gotSHA, arm.ProfileSHA256)
		}

		if arm.Fallbacks.PromisedCPUFallbacks != 0 {
			t.Errorf("arm %s promised CPU fallbacks = %d, want 0", arm.Name, arm.Fallbacks.PromisedCPUFallbacks)
		}

		if arm.Type == "C" {
			if arm.SequenceSelector {
				t.Errorf("control arm %s has SequenceSelector=true, want false", arm.Name)
			}
			if arm.Sequence.Available {
				t.Errorf("control arm %s has Sequence.Available=true, want false", arm.Name)
			}
		} else if arm.Type == "M" {
			if !arm.SequenceSelector {
				t.Errorf("candidate arm %s has SequenceSelector=false, want true", arm.Name)
			}
			if !arm.Sequence.Available {
				t.Errorf("candidate arm %s has Sequence.Available=false, want true", arm.Name)
			}
			if arm.Sequence.CommandBuffers != 1 {
				t.Errorf("candidate arm %s command_buffers = %d, want 1", arm.Name, arm.Sequence.CommandBuffers)
			}
			if arm.Sequence.TerminalWaits != 1 || arm.Sequence.TerminalReadbacks != 1 {
				t.Errorf("candidate arm %s terminal waits/readbacks = %d/%d, want 1/1", arm.Name, arm.Sequence.TerminalWaits, arm.Sequence.TerminalReadbacks)
			}
			if arm.Sequence.IntermediateWaits != 0 || arm.Sequence.IntermediateReadbacks != 0 {
				t.Errorf("candidate arm %s intermediate waits/readbacks = %d/%d, want 0/0", arm.Name, arm.Sequence.IntermediateWaits, arm.Sequence.IntermediateReadbacks)
			}
		} else {
			t.Errorf("arm %s has invalid type %q", arm.Name, arm.Type)
		}
	}

	if r.Comparison.PrefillMedianMS.ImprovementPct < 15.0 {
		t.Errorf("prefill median improvement = %.1f%%, want >= 15.0%%", r.Comparison.PrefillMedianMS.ImprovementPct)
	}
	if r.Comparison.PrefillCommandBuffers.Candidate != 1 {
		t.Errorf("candidate prefill command buffers = %d, want 1", r.Comparison.PrefillCommandBuffers.Candidate)
	}
}

func TestIssue9525SequenceReceiptValidator(t *testing.T) {
	hash := strings.Repeat("a", 64)
	artifact := qwen38quant.Identity{
		Model: "Qwen/Qwen3.8-27B-Q4_K_M", CheckpointSHA256: hash, ArtifactSHA256: hash,
		TokenizerSHA256: hash, TemplateSHA256: hash, QuantizerRevision: "q4-k-r1",
		RuntimeRevision: strings.Repeat("b", 40), FakModuleRev: "internal/model@r1+gabcdef0",
	}

	controlArm := qwen38quantrun.QwenMetalSequenceArm{
		SelectorEnabled: false,
		Artifact:        artifact,
		Receipt: &model.NativeInferenceReceipt{
			Model:       artifact.Model,
			Engine:      "fak-native",
			Backend:     "metal",
			ForwardPath: "metal/qwen35-hybrid-session-v1",
			Q4K:         true,
			Qwen35MetalForwardSequence: &model.Qwen35MetalForwardSequenceReceipt{
				SelectorState: model.Qwen35MetalSequenceSelectorOff,
				EvidenceState: model.Qwen35MetalSequenceEvidenceNotSelected,
			},
		},
	}

	candidateArm := qwen38quantrun.QwenMetalSequenceArm{
		SelectorEnabled:           true,
		Artifact:                  artifact,
		ExpectedHostUploadBytes:   65536,
		ExpectedHostReadbackBytes: 16384,
		Receipt: &model.NativeInferenceReceipt{
			Model:       artifact.Model,
			Engine:      "fak-native",
			Backend:     "metal",
			ForwardPath: model.Qwen35MetalGDNSequenceForwardPath,
			Q4K:         true,
			Qwen35MetalForwardSequence: &model.Qwen35MetalForwardSequenceReceipt{
				Path:              model.Qwen35MetalGDNSequenceForwardPath,
				Available:         true,
				Tokens:            32,
				SelectorState:     model.Qwen35MetalSequenceSelectorOn,
				EvidenceState:     model.Qwen35MetalSequenceEvidenceExecuted,
				CommandBuffers:    1,
				Encoders:          7,
				TerminalWaits:     1,
				TerminalReadbacks: 1,
				HostUploadBytes:   65536,
				HostReadbackBytes: 16384,
				Committed:         true,
				CompletedWait:     true,
			},
		},
	}

	res := qwen38quantrun.ValidateQwenMetalSequencePair(controlArm, candidateArm)
	if res.Verdict != qwen38quantrun.QwenMetalSequencePASS || len(res.Findings) != 0 {
		t.Fatalf("ValidateQwenMetalSequencePair = %+v, want PASS", res)
	}
}
