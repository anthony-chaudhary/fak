package issue11845witness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

const wantWitnessHash = "2d1e99750e716137a8de74738291e119c1af8f67e109d7cd909f180513f83e8e"

func TestIssue11845WitnessIntegrityAndInvariants(t *testing.T) {
	data, err := os.ReadFile("qwen38_20k_witness.json")
	if err != nil {
		t.Fatalf("failed to read witness artifact: %v", err)
	}

	sum := sha256.Sum256(data)
	gotHash := hex.EncodeToString(sum[:])
	if gotHash != wantWitnessHash {
		t.Fatalf("witness hash mismatch: got %s, want %s", gotHash, wantWitnessHash)
	}

	var rep struct {
		Schema      string `json:"schema"`
		Verdict     string `json:"verdict"`
		Environment struct {
			Engine  string `json:"engine"`
			Backend string `json:"backend"`
			Node    string `json:"node"`
			GPU     struct {
				Model string `json:"model"`
			} `json:"gpu"`
		} `json:"environment"`
		Model struct {
			ID           string `json:"id"`
			Quantization string `json:"quantization"`
		} `json:"model"`
		Measurements struct {
			PromptTokens int `json:"prompt_tokens"`
			Prefill      struct {
				Tokens            int     `json:"tokens"`
				ThroughputTokPSec float64 `json:"throughput_tok_per_sec"`
			} `json:"prefill"`
			GPUSummary struct {
				VRAMPeakDuringMiB int `json:"vram_peak_during_mib"`
			} `json:"gpu_telemetry"`
		} `json:"measurements"`
		Witness struct {
			InkernelChatLog string `json:"inkernel_chat_log"`
		} `json:"witness"`
	}

	if err := json.Unmarshal(data, &rep); err != nil {
		t.Fatalf("failed to parse witness json: %v", err)
	}

	if rep.Verdict != "PASS_20K_NATIVE_CUDA" {
		t.Errorf("unexpected verdict: got %q, want PASS_20K_NATIVE_CUDA", rep.Verdict)
	}
	if rep.Environment.Engine != "fak-native" {
		t.Errorf("unexpected engine: got %q, want fak-native", rep.Environment.Engine)
	}
	if rep.Environment.Backend != "cuda" {
		t.Errorf("unexpected backend: got %q, want cuda", rep.Environment.Backend)
	}
	if rep.Measurements.PromptTokens != 20480 {
		t.Errorf("unexpected prompt tokens: got %d, want 20480", rep.Measurements.PromptTokens)
	}
	if rep.Measurements.Prefill.ThroughputTokPSec < 80.0 {
		t.Errorf("prefill throughput below floor: got %f tok/s, want >= 80.0", rep.Measurements.Prefill.ThroughputTokPSec)
	}
	if !strings.Contains(rep.Witness.InkernelChatLog, "backend=cuda") ||
		!strings.Contains(rep.Witness.InkernelChatLog, "forward_path=cuda/qwen35-gdn-ssm-decode-v1") ||
		!strings.Contains(rep.Witness.InkernelChatLog, "prompt=20480tok") {
		t.Errorf("inkernel_chat_log missing key tokens: %s", rep.Witness.InkernelChatLog)
	}
}
