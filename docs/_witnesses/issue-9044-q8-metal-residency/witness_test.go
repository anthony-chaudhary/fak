package q8metalresidencywitness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	artifactSHA = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	profileSHA  = "d948c17836343b061e97722b0b74137a677cfaa082aeeea4afe0eb7968e3d68b"
	receiptSHA  = "4a8f3863570ab941857911c1cd2b581438f978e4ae7094722c22cfd845ea07b2"
)

type fileIdentity struct {
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type artifactIdentity struct {
	fileIdentity
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
}

type hostIdentity struct {
	GOOS                 string `json:"goos"`
	GOARCH               string `json:"goarch"`
	CPU                  string `json:"cpu"`
	MetalDevice          string `json:"metal_device"`
	GPUCores             int    `json:"gpu_cores"`
	MemoryBytes          uint64 `json:"memory_bytes"`
	MetalWorkingSetBytes uint64 `json:"metal_working_set_bytes"`
}

type sourceIdentity struct {
	Revision   string `json:"revision"`
	Modified   bool   `json:"modified"`
	DiffBytes  int    `json:"diff_bytes,omitempty"`
	DiffSHA256 string `json:"diff_sha256,omitempty"`
}

type executionEvent struct {
	Operation        string  `json:"operation"`
	CommandBufferID  uint64  `json:"command_buffer_id"`
	Committed        bool    `json:"committed"`
	CompletedWait    bool    `json:"completed_wait"`
	HostReadback     bool    `json:"host_readback"`
	Encoders         int     `json:"encoders"`
	GPUMilliseconds  float64 `json:"gpu_milliseconds"`
	WaitMilliseconds float64 `json:"wait_milliseconds"`
	TimingAvailable  bool    `json:"timing_available"`
}

type executionCounters struct {
	CommandBuffers       int     `json:"command_buffers"`
	Encoders             int     `json:"encoders"`
	DispatchMilliseconds float64 `json:"dispatch_milliseconds"`
	WaitMilliseconds     float64 `json:"wait_milliseconds"`
}

type executionReceipt struct {
	Schema       string            `json:"schema"`
	Events       []executionEvent  `json:"events"`
	EventsSHA256 string            `json:"events_sha256"`
	Counters     executionCounters `json:"counters"`
}

type fallbackEvent struct {
	Sequence        int    `json:"sequence"`
	Route           string `json:"route"`
	Operation       string `json:"operation"`
	PromisedBackend string `json:"promised_backend"`
	ActualBackend   string `json:"actual_backend"`
	Disposition     string `json:"disposition"`
	Promised        bool   `json:"promised"`
	CPUWorkExecuted bool   `json:"cpu_work_executed"`
}

type fallbackReceipt struct {
	Schema               string          `json:"schema"`
	Events               []fallbackEvent `json:"events"`
	EventsSHA256         string          `json:"events_sha256"`
	PromisedCPUFallbacks int             `json:"promised_cpu_fallbacks"`
}

type profileReceipt struct {
	Schema            string            `json:"schema"`
	BindingSHA256     string            `json:"binding_sha256"`
	ProfileSHA256     string            `json:"profile_sha256"`
	EnvelopeID        string            `json:"envelope_id"`
	Artifact          artifactIdentity  `json:"artifact"`
	ModelConfig       map[string]any    `json:"model_config"`
	ModelConfigSHA256 string            `json:"model_config_sha256"`
	Host              hostIdentity      `json:"host"`
	Source            sourceIdentity    `json:"source"`
	Binary            fileIdentity      `json:"binary"`
	Controls          map[string]string `json:"controls"`
	Execution         executionReceipt  `json:"execution"`
	Fallbacks         fallbackReceipt   `json:"fallbacks"`
}

type capture struct {
	Schema       string `json:"schema"`
	ProfileSHA   string `json:"profile_sha256"`
	ReceiptSHA   string `json:"receipt_sha256"`
	ReceiptBind  string `json:"receipt_binding_sha256"`
	FiniteLogits bool   `json:"finite_logits"`
	Q8Runtime    struct {
		Promised     int `json:"promised_projections_per_forward"`
		Prefill      int `json:"prefill_gemm_events"`
		DecodeSingle int `json:"decode_single_gemv_events"`
		DecodeGroup  int `json:"decode_group_gemv_events"`
		GroupWidth   int `json:"decode_group_width"`
		Total        int `json:"total_projection_executions"`
		CPUFallbacks int `json:"promised_cpu_fallbacks"`
	} `json:"q8_runtime"`
	Memory struct {
		OSPeakBytes      uint64  `json:"os_peak_footprint_bytes"`
		OSPeakSource     string  `json:"os_peak_footprint_source"`
		ResidentBytes    uint64  `json:"model_resident_bytes"`
		PhysicalBytes    uint64  `json:"physical_memory_bytes"`
		MetalBytes       uint64  `json:"metal_working_set_bytes"`
		SwapBeforeMiB    float64 `json:"swap_before_mib"`
		SwapAfterMiB     float64 `json:"swap_after_mib"`
		SwapDeltaMiB     float64 `json:"swap_delta_mib"`
		ForcedQ8Override bool    `json:"forced_q8_override"`
		Interpretation   string  `json:"interpretation"`
	} `json:"memory"`
	Restoration struct {
		ExactOwner bool `json:"exact_owner_restored"`
		Health     int  `json:"health_http_status"`
		Models     int  `json:"models_http_status"`
	} `json:"restoration"`
}

func readPinned(t *testing.T, name, want string) []byte {
	t.Helper()
	body, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != want {
		t.Fatalf("%s sha256=%s want %s", name, got, want)
	}
	return body
}

func validateExecution(t *testing.T, receipt executionReceipt) {
	t.Helper()
	if receipt.Schema != "fak-metal-execution-receipt/v1" || len(receipt.Events) == 0 {
		t.Fatalf("execution schema/events=%q/%d", receipt.Schema, len(receipt.Events))
	}
	raw, err := json.Marshal(receipt.Events)
	if err != nil {
		t.Fatal(err)
	}
	if got := sha256.Sum256(raw); hex.EncodeToString(got[:]) != receipt.EventsSHA256 {
		t.Fatal("execution event digest mismatch")
	}
	var got executionCounters
	for i, event := range receipt.Events {
		if !event.Committed || !event.CompletedWait || !event.HostReadback || !event.TimingAvailable ||
			event.Encoders <= 0 || event.GPUMilliseconds <= 0 || event.WaitMilliseconds < 0 ||
			math.IsNaN(event.GPUMilliseconds) || math.IsInf(event.GPUMilliseconds, 0) ||
			math.IsNaN(event.WaitMilliseconds) || math.IsInf(event.WaitMilliseconds, 0) {
			t.Fatalf("event %d lacks complete lifecycle: %+v", i+1, event)
		}
		got.CommandBuffers++
		got.Encoders += event.Encoders
		got.DispatchMilliseconds += event.GPUMilliseconds
		got.WaitMilliseconds += event.WaitMilliseconds
	}
	if got != receipt.Counters {
		t.Fatalf("execution aggregate=%+v want %+v", got, receipt.Counters)
	}
}

func validateFallbacks(t *testing.T, receipt fallbackReceipt) {
	t.Helper()
	if receipt.Schema != "fak-metal-fallback-receipt/v1" {
		t.Fatalf("fallback schema=%q", receipt.Schema)
	}
	raw, err := json.Marshal(receipt.Events)
	if err != nil {
		t.Fatal(err)
	}
	if got := sha256.Sum256(raw); hex.EncodeToString(got[:]) != receipt.EventsSHA256 {
		t.Fatal("fallback event digest mismatch")
	}
	count := 0
	for i, event := range receipt.Events {
		if event.Sequence != i+1 {
			t.Fatalf("fallback sequence %d=%d", i+1, event.Sequence)
		}
		if event.Promised && event.CPUWorkExecuted && event.ActualBackend == "cpu" {
			count++
		}
	}
	if count != receipt.PromisedCPUFallbacks {
		t.Fatalf("fallback aggregate=%d want %d", count, receipt.PromisedCPUFallbacks)
	}
}

func TestQ8MetalResidencyReceiptReadback(t *testing.T) {
	profileBody := readPinned(t, "profile.json", profileSHA)
	receiptBody := readPinned(t, "profile.receipt.json", receiptSHA)

	profile, err := nativeperf.DecodeProfile(profileBody)
	if err != nil {
		t.Fatal(err)
	}
	if err := nativeperf.ValidateProfile(nativeperf.ActiveGraph(), profile); err != nil {
		t.Fatal(err)
	}
	var receipt profileReceipt
	if err := json.Unmarshal(receiptBody, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != "fak-native-metal-profile-receipt/v1" || receipt.EnvelopeID != profile.EnvelopeID {
		t.Fatalf("receipt schema/envelope=%q/%q", receipt.Schema, receipt.EnvelopeID)
	}
	if receipt.Artifact.Bytes != 17106775008 || receipt.Artifact.SHA256 != artifactSHA {
		t.Fatalf("artifact=%d/%s", receipt.Artifact.Bytes, receipt.Artifact.SHA256)
	}
	if got := sha256.Sum256(profileBody); hex.EncodeToString(got[:]) != receipt.ProfileSHA256 {
		t.Fatal("profile digest is not bound by receipt")
	}
	configBody, err := json.Marshal(receipt.ModelConfig)
	if err != nil {
		t.Fatal(err)
	}
	if got := sha256.Sum256(configBody); hex.EncodeToString(got[:]) != receipt.ModelConfigSHA256 {
		t.Fatal("model config digest mismatch")
	}
	wantBinding := receipt.BindingSHA256
	receipt.BindingSHA256 = ""
	bindingBody, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if got := sha256.Sum256(bindingBody); hex.EncodeToString(got[:]) != wantBinding {
		t.Fatal("receipt binding digest mismatch")
	}
	validateExecution(t, receipt.Execution)
	validateFallbacks(t, receipt.Fallbacks)
	if receipt.Execution.Counters.CommandBuffers != profile.Metal.CommandBuffers ||
		receipt.Execution.Counters.Encoders != profile.Metal.Encoders ||
		receipt.Execution.Counters.DispatchMilliseconds != profile.Metal.DispatchMilliseconds ||
		receipt.Execution.Counters.WaitMilliseconds != profile.Metal.WaitMilliseconds {
		t.Fatal("raw event totals do not equal profile counters")
	}
	if profile.Execution.Engine != "fak-native" || profile.Execution.ForwardPath != "metal/qwen35-hybrid-session-v1" ||
		profile.Execution.FallbackCount != 0 || receipt.Fallbacks.PromisedCPUFallbacks != 0 {
		t.Fatalf("execution identity=%+v fallback=%d", profile.Execution, receipt.Fallbacks.PromisedCPUFallbacks)
	}
	counts := map[string]int{}
	for _, event := range receipt.Execution.Events {
		counts[event.Operation]++
	}
	if counts["q8-gemm"] != 272 || counts["q8-gemv"] != 5120 || counts["q8-gemv-group"] != 3072 {
		t.Fatalf("Q8 operation counts=%v", counts)
	}
	if receipt.Controls["FAK_METAL_STREAM_Q4K"] != "1" || receipt.Controls["FAK_Q4K_FREE_CPU"] != "1" ||
		receipt.Controls["FAK_Q4K"] != "1" || receipt.Controls["FAK_METAL_Q8_UPLOAD"] != "<unset>" {
		t.Fatalf("strict controls not preserved: %+v", receipt.Controls)
	}

	var captured capture
	captureBody, err := os.ReadFile("capture.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(captureBody, &captured); err != nil {
		t.Fatal(err)
	}
	if captured.Schema != "fak.issue-9044-q8-metal-residency-capture/1" || captured.ProfileSHA != profileSHA ||
		captured.ReceiptSHA != receiptSHA || captured.ReceiptBind != wantBinding || !captured.FiniteLogits {
		t.Fatalf("capture binding incomplete: %+v", captured)
	}
	wantTotal := captured.Q8Runtime.Prefill + captured.Q8Runtime.DecodeSingle + captured.Q8Runtime.GroupWidth*captured.Q8Runtime.DecodeGroup
	if captured.Q8Runtime.Promised != 272 || wantTotal != 17680 || captured.Q8Runtime.Total != wantTotal || captured.Q8Runtime.CPUFallbacks != 0 {
		t.Fatalf("Q8 projection recomputation=%d capture=%+v", wantTotal, captured.Q8Runtime)
	}
	if captured.Memory.OSPeakSource != "getrusage(RUSAGE_SELF).ru_maxrss" || captured.Memory.OSPeakBytes != profile.Metal.WorkingSetBytes ||
		captured.Memory.ResidentBytes != profile.Metal.ResidentBytes || captured.Memory.PhysicalBytes != receipt.Host.MemoryBytes ||
		captured.Memory.MetalBytes != receipt.Host.MetalWorkingSetBytes || captured.Memory.ForcedQ8Override ||
		math.Abs((captured.Memory.SwapAfterMiB-captured.Memory.SwapBeforeMiB)-captured.Memory.SwapDeltaMiB) > 0.005 ||
		captured.Memory.SwapDeltaMiB != 1773.87 || captured.Memory.Interpretation == "" {
		t.Fatalf("memory capture mismatch: %+v", captured.Memory)
	}
	if captured.Memory.OSPeakBytes >= captured.Memory.PhysicalBytes || captured.Memory.OSPeakBytes >= captured.Memory.MetalBytes {
		t.Fatalf("peak footprint lacks recorded headroom: %+v", captured.Memory)
	}
	if !captured.Restoration.ExactOwner || captured.Restoration.Health != 200 || captured.Restoration.Models != 200 {
		t.Fatalf("service restoration=%+v", captured.Restoration)
	}
}
