package issue8820witness

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	pinnedArtifactSHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	pinnedForwardPath = "qwen35-hybrid-sequence-prefill-v1"
)

type receipt struct {
	Schema         string `json:"schema"`
	Role           string `json:"role"`
	EnvelopeID     string `json:"envelope_id"`
	ChangedLeverID string `json:"changed_lever_id"`
	ArtifactSHA256 string `json:"artifact_sha256"`
	Machine        struct {
		Backend string `json:"backend"`
	} `json:"machine"`
	Controls struct {
		PromptTokens int `json:"prompt_tokens"`
		DecodeTokens int `json:"decode_tokens"`
		Repetitions  int `json:"repetitions"`
	} `json:"controls"`
	Repetitions []struct {
		EndToEndMilliseconds float64 `json:"end_to_end_milliseconds"`
		TokensPerSecond      float64 `json:"tokens_per_second"`
		TTFTMilliseconds     float64 `json:"ttft_milliseconds"`
		PrefillMilliseconds  float64 `json:"prefill_milliseconds"`
		DecodeMilliseconds   float64 `json:"decode_milliseconds"`
	} `json:"repetitions"`
	Execution struct {
		Engine        string `json:"engine"`
		ForwardPath   string `json:"forward_path"`
		FallbackCount int    `json:"fallback_count"`
	} `json:"execution"`
	Quality struct {
		Name  string  `json:"name"`
		Score float64 `json:"score"`
	} `json:"quality"`
}

type pairedReport struct {
	Schema string `json:"schema"`
	Arms   map[string]struct {
		TreatmentArm        string  `json:"treatment_arm"`
		BaselineArm         string  `json:"baseline_arm"`
		ApplesToApples      bool    `json:"apples_to_apples"`
		TreatmentPrefillMS  float64 `json:"treatment_prefill_ms"`
		BaselinePrefillMS   float64 `json:"baseline_prefill_ms"`
		TreatmentPrefillTPS float64 `json:"treatment_prefill_tps"`
		BaselinePrefillTPS  float64 `json:"baseline_prefill_tps"`
		SpeedupPrefill      float64 `json:"speedup_prefill"`
		FallbackCount       int     `json:"fallback_count"`
		QualityExact        bool    `json:"quality_exact"`
	} `json:"arms"`
}

func readJSONFile[T any](t *testing.T, filename string) T {
	t.Helper()
	data, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("read %s: %v", filename, err)
	}
	var val T
	if err := json.Unmarshal(data, &val); err != nil {
		t.Fatalf("unmarshal %s: %v", filename, err)
	}
	return val
}

func TestIssue8820ReceiptStrictParityAndThroughputGain(t *testing.T) {
	receiptPath := filepath.Join(".", "receipt.json")
	r := readJSONFile[receipt](t, receiptPath)

	if r.Schema != "fak-native-performance-receipt/v1" && r.Schema != "fak-native-performance-receipt/v2" {
		t.Fatalf("receipt schema = %q, want nativeperf receipt schema", r.Schema)
	}
	if r.Role != "candidate" {
		t.Fatalf("receipt role = %q, want candidate", r.Role)
	}
	if r.ArtifactSHA256 != pinnedArtifactSHA {
		t.Fatalf("artifact sha256 = %q, want exact %q", r.ArtifactSHA256, pinnedArtifactSHA)
	}
	if r.Machine.Backend != "cuda" {
		t.Fatalf("backend = %q, want cuda", r.Machine.Backend)
	}
	if r.Execution.Engine != "fak-native" {
		t.Fatalf("execution engine = %q, want fak-native", r.Execution.Engine)
	}
	if r.Execution.ForwardPath != pinnedForwardPath {
		t.Fatalf("forward path = %q, want %q", r.Execution.ForwardPath, pinnedForwardPath)
	}
	if r.Execution.FallbackCount != 0 {
		t.Fatalf("fallback count = %d, want 0", r.Execution.FallbackCount)
	}
	if r.Quality.Score < 1.0 {
		t.Fatalf("quality score = %f, want strict exact match 1.0", r.Quality.Score)
	}
	if len(r.Repetitions) < 3 {
		t.Fatalf("repetition count = %d, want >= 3", len(r.Repetitions))
	}
	for i, rep := range r.Repetitions {
		if rep.PrefillMilliseconds <= 0 || rep.TTFTMilliseconds <= 0 {
			t.Fatalf("rep %d non-positive timing: prefill=%.2f ttft=%.2f", i, rep.PrefillMilliseconds, rep.TTFTMilliseconds)
		}
		// The serial baseline was 11,830 ms for 22 tokens (1.9 tok/s).
		// The panel prefill executes under 500 ms (speedup >= 20x).
		if rep.PrefillMilliseconds > 1000 {
			t.Fatalf("rep %d prefill latency %.2f ms exceeds 1000 ms ceiling (did not beat serial baseline)", i, rep.PrefillMilliseconds)
		}
	}
}

func TestIssue8820PairedReportPositivePrefillGain(t *testing.T) {
	pairedPath := filepath.Join(".", "paired-report.json")
	report := readJSONFile[pairedReport](t, pairedPath)

	arm, ok := report.Arms["cuda-panel-prefill"]
	if !ok {
		t.Fatal("missing cuda-panel-prefill arm in paired-report.json")
	}
	if !arm.ApplesToApples {
		t.Fatal("cuda-panel-prefill arm must be apples-to-apples")
	}
	if arm.SpeedupPrefill <= 1.0 {
		t.Fatalf("prefill speedup %.2fx must be strictly positive gain (> 1.0x)", arm.SpeedupPrefill)
	}
	if arm.FallbackCount != 0 {
		t.Fatalf("fallback count = %d, want 0", arm.FallbackCount)
	}
	if !arm.QualityExact {
		t.Fatal("quality_exact must be true")
	}
}

func TestIssue8820ReadmeDocumentsVerdict(t *testing.T) {
	readmeData, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	content := string(readmeData)
	if !strings.Contains(content, "PASS_PANEL_PREFILL") {
		t.Fatal("README.md missing PASS_PANEL_PREFILL verdict")
	}
	if !strings.Contains(content, pinnedForwardPath) {
		t.Fatalf("README.md missing forward path %s", pinnedForwardPath)
	}
}
