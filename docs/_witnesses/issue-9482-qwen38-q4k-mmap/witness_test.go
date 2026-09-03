package issue9482witness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const (
	expectedSchema   = "fak.issue-9482-qwen38-q4k-mmap-receipt/1"
	expectedArtifact = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	expectedBytes    = int64(17106775008)
	expectedEngine   = "fak-native"
	expectedBackend  = "metal"
	expectedForward  = "metal/qwen35-hybrid-session-v1"
	expectedTensors  = 184
	expectedMappedB  = 8328314880
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
	Name                 string             `json:"name"`
	Type                 string             `json:"type"`
	FAKGGUFMMap          string             `json:"fak_gguf_mmap"`
	ProfileSHA256        string             `json:"profile_sha256"`
	ReceiptBindingSHA256 string             `json:"receipt_binding_sha256"`
	Phases               map[string]float64 `json:"phases"`
	Metal                struct {
		CommandBuffers int     `json:"command_buffers"`
		Encoders       int     `json:"encoders"`
		ResidentBytes  uint64  `json:"resident_bytes"`
		WorkingSet     uint64  `json:"working_set_bytes"`
		DispatchMS     float64 `json:"dispatch_milliseconds"`
		WaitMS         float64 `json:"wait_milliseconds"`
	} `json:"metal"`
	Q4KResidency struct {
		Schema        string `json:"schema"`
		FAKGGUFMMap   string `json:"fak_gguf_mmap"`
		MappedSuccess struct {
			Tensors int   `json:"tensors"`
			Bytes   int64 `json:"bytes"`
		} `json:"mapped_success"`
		MappedDecline struct {
			Tensors int   `json:"tensors"`
			Bytes   int64 `json:"bytes"`
		} `json:"mapped_decline_copied_upload"`
		UploadFailure struct {
			Tensors int   `json:"tensors"`
			Bytes   int64 `json:"bytes"`
		} `json:"upload_failure"`
		IntegritySHA256 string `json:"integrity_sha256"`
	} `json:"q4k_residency"`
	Fallbacks struct {
		Schema               string `json:"schema"`
		PromisedCPUFallbacks int    `json:"promised_cpu_fallbacks"`
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
		MappedTensors struct {
			Control   int `json:"control"`
			Candidate int `json:"candidate"`
		} `json:"mapped_tensors"`
		MappedBytes struct {
			Control   int64 `json:"control"`
			Candidate int64 `json:"candidate"`
		} `json:"mapped_bytes"`
	} `json:"comparison"`
	Arms []receiptArm `json:"arms"`
}

func TestIssue9482CampaignReceipt(t *testing.T) {
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
	if r.Issue != 9482 || r.ParentIssue != 8325 || r.UmbrellaIssue != 9430 {
		t.Errorf("issue metadata mismatch: issue=%d parent=%d umbrella=%d", r.Issue, r.ParentIssue, r.UmbrellaIssue)
	}
	if r.Phase != "M1" {
		t.Errorf("phase = %q, want M1", r.Phase)
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
		if arm.Metal.CommandBuffers != 14833 || arm.Metal.Encoders != 23025 {
			t.Errorf("arm %s unexpected command buffers / encoders: cb=%d enc=%d", arm.Name, arm.Metal.CommandBuffers, arm.Metal.Encoders)
		}

		if arm.Type == "C" {
			if arm.FAKGGUFMMap != "0" {
				t.Errorf("control arm %s has FAK_GGUF_MMAP=%q, want 0", arm.Name, arm.FAKGGUFMMap)
			}
			if arm.Q4KResidency.MappedSuccess.Tensors != 0 {
				t.Errorf("control arm %s has %d mapped tensors, want 0", arm.Name, arm.Q4KResidency.MappedSuccess.Tensors)
			}
		} else if arm.Type == "M" {
			if arm.FAKGGUFMMap != "1" {
				t.Errorf("candidate arm %s has FAK_GGUF_MMAP=%q, want 1", arm.Name, arm.FAKGGUFMMap)
			}
			if arm.Q4KResidency.MappedSuccess.Tensors != expectedTensors {
				t.Errorf("candidate arm %s mapped tensors = %d, want %d", arm.Name, arm.Q4KResidency.MappedSuccess.Tensors, expectedTensors)
			}
			if arm.Q4KResidency.MappedSuccess.Bytes != expectedMappedB {
				t.Errorf("candidate arm %s mapped bytes = %d, want %d", arm.Name, arm.Q4KResidency.MappedSuccess.Bytes, expectedMappedB)
			}
			if arm.Q4KResidency.MappedDecline.Tensors != 0 || arm.Q4KResidency.UploadFailure.Tensors != 0 {
				t.Errorf("candidate arm %s had declined=%d failed=%d", arm.Name, arm.Q4KResidency.MappedDecline.Tensors, arm.Q4KResidency.UploadFailure.Tensors)
			}
		} else {
			t.Errorf("arm %s has invalid type %q", arm.Name, arm.Type)
		}
	}

	if r.Comparison.FirstTokenMedianMS.ImprovementPct <= 0 {
		t.Errorf("first-token median improvement = %.1f%%, want > 0", r.Comparison.FirstTokenMedianMS.ImprovementPct)
	}
	if r.Comparison.PrefillMedianMS.ImprovementPct <= 0 {
		t.Errorf("prefill median improvement = %.1f%%, want > 0", r.Comparison.PrefillMedianMS.ImprovementPct)
	}
	if r.Comparison.MappedTensors.Candidate != expectedTensors || r.Comparison.MappedTensors.Control != 0 {
		t.Errorf("mapped tensors comparison mismatch: %+v", r.Comparison.MappedTensors)
	}
}
