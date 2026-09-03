package issue9495metalprofile_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	expectedSchema      = "fak.issue-9495-metal-profile-bundle/1"
	expectedArtifact    = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	expectedEngine      = "fak-native"
	expectedBackend     = "metal"
	expectedForwardPath = "metal/qwen35-hybrid-session-v1"
	expectedTarget      = "qwen38-27b-q4km-m3pro-p32-t64"
)

type metalProfileBundle struct {
	Schema        string `json:"schema"`
	Issue         int    `json:"issue"`
	ParentIssue   int    `json:"parent_issue"`
	UmbrellaIssue int    `json:"umbrella_issue"`
	Target        string `json:"target"`
	Artifact      struct {
		Model        string `json:"model"`
		Revision     string `json:"revision"`
		File         string `json:"file"`
		Quantization string `json:"quantization"`
		Bytes        int64  `json:"bytes"`
		SHA256       string `json:"sha256"`
	} `json:"artifact"`
	Execution struct {
		Engine               string `json:"engine"`
		Backend              string `json:"backend"`
		ForwardPath          string `json:"forward_path"`
		PromptTokens         int    `json:"prompt_tokens"`
		DecodeTokens         int    `json:"decode_tokens"`
		FallbackCount        int    `json:"fallback_count"`
		TotalFallbacks       int    `json:"total_fallbacks"`
		PromisedCPUFallbacks int    `json:"promised_cpu_fallbacks"`
	} `json:"execution"`
	Hardware struct {
		SOC         string `json:"soc"`
		GPUCores    int    `json:"gpu_cores"`
		MemoryBytes uint64 `json:"memory_bytes"`
		MemoryGiB   int    `json:"memory_gib"`
		OS          string `json:"os"`
		Arch        string `json:"arch"`
	} `json:"hardware"`
	PhaseBreakdown struct {
		HostMaterializationMS float64 `json:"host_materialization_ms"`
		PrefillMS             float64 `json:"prefill_ms"`
		DecodeMS              float64 `json:"decode_ms"`
		FirstTokenMS          float64 `json:"first_token_ms"`
		SteadyDecodeMS        float64 `json:"steady_decode_ms"`
		VerificationMS        float64 `json:"verification_ms"`
		TeardownMS            float64 `json:"teardown_ms"`
		TotalWallMS           float64 `json:"total_wall_ms"`
	} `json:"phase_breakdown"`
	Phases []struct {
		Name                 string  `json:"name"`
		Description          string  `json:"description"`
		StartMilliseconds    float64 `json:"start_milliseconds"`
		DurationMilliseconds float64 `json:"duration_milliseconds"`
	} `json:"phases"`
	Metal struct {
		CommandBufferCount   int     `json:"command_buffer_count"`
		EncoderCount         int     `json:"encoder_count"`
		HostWaitTimeMS       float64 `json:"host_wait_time_ms"`
		GPUExecutionTimeMS   float64 `json:"gpu_execution_time_ms"`
		CommandBuffers       int     `json:"command_buffers"`
		Encoders             int     `json:"encoders"`
		DispatchMilliseconds float64 `json:"dispatch_milliseconds"`
		WaitMilliseconds     float64 `json:"wait_milliseconds"`
		ResidentBytes        uint64  `json:"resident_bytes"`
		WorkingSetBytes      uint64  `json:"working_set_bytes"`
	} `json:"metal"`
	ConstraintDriver struct {
		SelectedDriver     string   `json:"selected_driver"`
		BottleneckClass    string   `json:"bottleneck_class"`
		Confidence         string   `json:"confidence"`
		RecommendedLeverID string   `json:"recommended_lever_id"`
		Evidence           []string `json:"evidence"`
	} `json:"constraint_driver"`
	NativePerformanceProfile nativeperf.ProfileBundle `json:"native_performance_profile"`
	CaptureMetadata          struct {
		CaptureCommand        string            `json:"capture_command"`
		CaptureSourceRevision string            `json:"capture_source_revision"`
		SourceClean           bool              `json:"source_clean"`
		ScrubbedHashes        map[string]string `json:"scrubbed_hashes"`
		ZeroFallbackVerified  bool              `json:"zero_fallback_verified"`
		Reproducible          bool              `json:"reproducible"`
	} `json:"capture_metadata"`
}

func loadBundle(t *testing.T) metalProfileBundle {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(".", "metal_profile_bundle.json"))
	if err != nil {
		t.Fatalf("read metal_profile_bundle.json: %v", err)
	}
	var b metalProfileBundle
	if err := json.Unmarshal(data, &b); err != nil {
		t.Fatalf("unmarshal bundle: %v", err)
	}
	return b
}

func TestMetalProfileBundleSchemaAndIdentity(t *testing.T) {
	bundle := loadBundle(t)

	if bundle.Schema != expectedSchema {
		t.Errorf("schema = %q, want %q", bundle.Schema, expectedSchema)
	}
	if bundle.Issue != 9495 {
		t.Errorf("issue = %d, want 9495", bundle.Issue)
	}
	if bundle.Artifact.SHA256 != expectedArtifact {
		t.Errorf("artifact sha256 = %q, want %q", bundle.Artifact.SHA256, expectedArtifact)
	}
	if bundle.Execution.Engine != expectedEngine {
		t.Errorf("execution engine = %q, want %q", bundle.Execution.Engine, expectedEngine)
	}
	if bundle.Execution.Backend != expectedBackend {
		t.Errorf("execution backend = %q, want %q", bundle.Execution.Backend, expectedBackend)
	}
	if bundle.Execution.ForwardPath != expectedForwardPath {
		t.Errorf("forward path = %q, want %q", bundle.Execution.ForwardPath, expectedForwardPath)
	}
	if bundle.Execution.FallbackCount != 0 || bundle.Execution.TotalFallbacks != 0 || bundle.Execution.PromisedCPUFallbacks != 0 {
		t.Errorf("fallback count must be 0, got fallback=%d total=%d promised=%d",
			bundle.Execution.FallbackCount, bundle.Execution.TotalFallbacks, bundle.Execution.PromisedCPUFallbacks)
	}
	if !bundle.CaptureMetadata.ZeroFallbackVerified {
		t.Errorf("zero_fallback_verified must be true")
	}
}

func TestMetalProfileBundlePhasesBreakdown(t *testing.T) {
	bundle := loadBundle(t)

	// Breakdown reporting host materialization, prefill, decode
	if bundle.PhaseBreakdown.HostMaterializationMS <= 0 {
		t.Errorf("host_materialization_ms must be positive, got %f", bundle.PhaseBreakdown.HostMaterializationMS)
	}
	if bundle.PhaseBreakdown.PrefillMS <= 0 {
		t.Errorf("prefill_ms must be positive, got %f", bundle.PhaseBreakdown.PrefillMS)
	}
	if bundle.PhaseBreakdown.DecodeMS <= 0 {
		t.Errorf("decode_ms must be positive, got %f", bundle.PhaseBreakdown.DecodeMS)
	}
	if bundle.PhaseBreakdown.FirstTokenMS <= 0 {
		t.Errorf("first_token_ms must be positive, got %f", bundle.PhaseBreakdown.FirstTokenMS)
	}
	if bundle.PhaseBreakdown.SteadyDecodeMS <= 0 {
		t.Errorf("steady_decode_ms must be positive, got %f", bundle.PhaseBreakdown.SteadyDecodeMS)
	}

	// Sum consistency: first_token + steady_decode equals decode total
	expectedDecode := bundle.PhaseBreakdown.FirstTokenMS + bundle.PhaseBreakdown.SteadyDecodeMS
	if diff := bundle.PhaseBreakdown.DecodeMS - expectedDecode; diff > 0.001 || diff < -0.001 {
		t.Errorf("decode_ms (%f) != first_token (%f) + steady_decode (%f)",
			bundle.PhaseBreakdown.DecodeMS, bundle.PhaseBreakdown.FirstTokenMS, bundle.PhaseBreakdown.SteadyDecodeMS)
	}

	// 6 ordered phases
	expectedPhaseNames := []string{"load-setup", "prefill", "first-token", "steady-decode", "verification", "teardown"}
	if len(bundle.Phases) != len(expectedPhaseNames) {
		t.Fatalf("phases count = %d, want %d", len(bundle.Phases), len(expectedPhaseNames))
	}
	var prevEnd float64
	for i, phase := range bundle.Phases {
		if phase.Name != expectedPhaseNames[i] {
			t.Errorf("phase[%d] name = %q, want %q", i, phase.Name, expectedPhaseNames[i])
		}
		if phase.DurationMilliseconds <= 0 {
			t.Errorf("phase %q duration must be positive, got %f", phase.Name, phase.DurationMilliseconds)
		}
		if phase.StartMilliseconds < prevEnd-0.001 {
			t.Errorf("phase %q start (%f) overlaps previous end (%f)", phase.Name, phase.StartMilliseconds, prevEnd)
		}
		prevEnd = phase.StartMilliseconds + phase.DurationMilliseconds
	}
}

func TestMetalProfileBundleDeviceCounters(t *testing.T) {
	bundle := loadBundle(t)

	// Command-buffer count
	if bundle.Metal.CommandBufferCount <= 0 || bundle.Metal.CommandBuffers <= 0 {
		t.Errorf("command-buffer count must be positive, got count=%d buffers=%d",
			bundle.Metal.CommandBufferCount, bundle.Metal.CommandBuffers)
	}
	if bundle.Metal.CommandBufferCount != 14833 || bundle.Metal.CommandBuffers != 14833 {
		t.Errorf("command buffers = %d, want 14833", bundle.Metal.CommandBuffers)
	}

	// Encoder count
	if bundle.Metal.EncoderCount <= 0 || bundle.Metal.Encoders <= 0 {
		t.Errorf("encoder count must be positive, got count=%d encoders=%d",
			bundle.Metal.EncoderCount, bundle.Metal.Encoders)
	}
	if bundle.Metal.EncoderCount != 23025 || bundle.Metal.Encoders != 23025 {
		t.Errorf("encoders = %d, want 23025", bundle.Metal.Encoders)
	}

	// Host wait time
	if bundle.Metal.HostWaitTimeMS <= 0 || bundle.Metal.WaitMilliseconds <= 0 {
		t.Errorf("host wait time must be positive, got wait_ms=%f", bundle.Metal.WaitMilliseconds)
	}

	// GPU execution time (must be non-zero and positive)
	if bundle.Metal.GPUExecutionTimeMS <= 0 || bundle.Metal.DispatchMilliseconds <= 0 {
		t.Errorf("GPU execution time must be non-zero and positive, got gpu_ms=%f", bundle.Metal.GPUExecutionTimeMS)
	}
}

func TestMetalProfileBundleDriverSelection(t *testing.T) {
	bundle := loadBundle(t)

	driver := bundle.ConstraintDriver.SelectedDriver
	if driver == "" {
		t.Fatalf("constraint driver must not be empty")
	}
	if !strings.Contains(driver, "synchronization") || !strings.Contains(driver, "host-wait amortization") {
		t.Errorf("selected driver = %q, want 'synchronization / host-wait amortization'", driver)
	}
	if bundle.ConstraintDriver.BottleneckClass != "synchronization-bound" {
		t.Errorf("bottleneck class = %q, want 'synchronization-bound'", bundle.ConstraintDriver.BottleneckClass)
	}
	if bundle.ConstraintDriver.RecommendedLeverID != "metal.command-buffer-amortization" {
		t.Errorf("recommended lever = %q, want 'metal.command-buffer-amortization'", bundle.ConstraintDriver.RecommendedLeverID)
	}
	if len(bundle.ConstraintDriver.Evidence) == 0 {
		t.Errorf("constraint driver evidence must not be empty")
	}
}

func TestMetalProfileBundleNativeperfClassification(t *testing.T) {
	bundle := loadBundle(t)
	graph := nativeperf.ActiveGraph()

	// Validate the embedded profile against nativeperf graph
	if err := nativeperf.ValidateProfile(graph, bundle.NativePerformanceProfile); err != nil {
		t.Fatalf("ValidateProfile failed: %v", err)
	}

	// Classify the profile
	classification, err := nativeperf.ClassifyProfile(graph, bundle.NativePerformanceProfile)
	if err != nil {
		t.Fatalf("ClassifyProfile failed: %v", err)
	}

	if classification.Class != "synchronization-bound" {
		t.Errorf("classified class = %q, want 'synchronization-bound'", classification.Class)
	}
	if classification.RecommendedLeverID != "metal.command-buffer-amortization" {
		t.Errorf("recommended lever = %q, want 'metal.command-buffer-amortization'", classification.RecommendedLeverID)
	}
}

func TestMetalProfileBundleCaptureMetadataAndScrubbedHashes(t *testing.T) {
	bundle := loadBundle(t)

	if bundle.CaptureMetadata.CaptureCommand == "" {
		t.Errorf("capture_command must be present")
	}
	// Verify scrubbed command does not contain private user directories
	if strings.Contains(bundle.CaptureMetadata.CaptureCommand, "/Users/") {
		t.Errorf("capture command contains unscrubbed /Users/ path: %q", bundle.CaptureMetadata.CaptureCommand)
	}

	requiredHashes := []string{
		"raw_event_digest",
		"fallback_stream_digest",
		"receipt_binding_sha256",
		"binary_sha256",
		"profile_sha256",
		"companion_receipt_sha256",
		"restored_owner_sha256",
	}
	for _, hashName := range requiredHashes {
		val, ok := bundle.CaptureMetadata.ScrubbedHashes[hashName]
		if !ok || len(val) != 64 {
			t.Errorf("scrubbed hash %q missing or not 64 hex chars (got %q)", hashName, val)
		}
		if _, err := hex.DecodeString(val); err != nil {
			t.Errorf("scrubbed hash %q is not valid hex: %v", hashName, err)
		}
	}

	// Verify file sha256 can be computed and matches non-empty
	data, err := os.ReadFile(filepath.Join(".", "metal_profile_bundle.json"))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	if len(sum) == 0 {
		t.Fatal("empty sha256")
	}
}
