package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const (
	nativeCapacityReceiptSchema  = "fak-native-streamed-q4k-capacity/1"
	nativeCapacityReadbackSchema = "fak-native-streamed-q4k-capacity-readback/1"
	nativeCapacityPlanSchema     = "fak-native-streamed-q4k-capacity-plan/1"
	nativeCapacityIssue          = 8971
	nativeCapacityArtifactBytes  = int64(17106775008)
	nativeCapacityArtifactSHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	nativeCapacityHostBytes      = uint64(36 << 30)
)

var nativeCapacityPresentEnvironment = []string{
	"FAK_METAL_STREAM_Q4K=1",
	"FAK_Q4K=1",
}

var nativeCapacityAbsentEnvironment = []string{
	"FAK_BUDGET",
	"FAK_METAL_Q8_UPLOAD",
	"FAK_Q4K_FREE_CPU",
	"FAK_Q4K_MM",
	"FAK_Q8_GEMM_GROUP",
	"FAK_WORKERS",
	"GOMAXPROCS",
}

type nativeCapacityFileIdentity struct {
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type nativeCapacityArtifactIdentity struct {
	nativeCapacityFileIdentity
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
}

type nativeCapacitySourceIdentity struct {
	Revision string `json:"revision"`
	Dirty    bool   `json:"dirty"`
}

type nativeCapacityHostIdentity struct {
	ScrubbedID   string `json:"scrubbed_id"`
	SOC          string `json:"soc"`
	GPUCores     int    `json:"gpu_cores"`
	MemoryBytes  uint64 `json:"memory_bytes"`
	OSKernel     string `json:"os_kernel"`
	Architecture string `json:"architecture"`
}

type nativeCapacityEnvironment struct {
	Present []string `json:"present"`
	Absent  []string `json:"absent"`
}

type nativeCapacityDisplacementSource struct {
	ScrubbedID string `json:"scrubbed_id"`
	Bytes      int64  `json:"bytes"`
	SHA256     string `json:"sha256"`
}

type nativeCapacityCacheDisplacement struct {
	TotalBytes int64                              `json:"total_bytes"`
	Sources    []nativeCapacityDisplacementSource `json:"sources"`
}

type nativeCapacityReadiness struct {
	Endpoint        string `json:"endpoint"`
	HTTPStatus      int    `json:"http_status"`
	ElapsedMillis   int64  `json:"elapsed_millis"`
	DeadlineSeconds int    `json:"deadline_seconds"`
}

type nativeCapacityExecution struct {
	Engine         string `json:"engine"`
	Backend        string `json:"backend"`
	ForwardPath    string `json:"forward_path"`
	FallbackCount  int    `json:"fallback_count"`
	LlamaCPPUsed   bool   `json:"llama_cpp_used"`
	IdentitySource string `json:"identity_source"`
}

type nativeCapacityMemory struct {
	RSSSamplesBytes             []uint64 `json:"rss_samples_bytes"`
	MaxSampledRSSBytes          uint64   `json:"max_sampled_rss_bytes"`
	TimeMaximumResidentSetBytes uint64   `json:"time_maximum_resident_set_bytes"`
	OSFootprintSamplesBytes     []uint64 `json:"os_footprint_samples_bytes"`
	OSPeakMemoryFootprintBytes  uint64   `json:"os_peak_memory_footprint_bytes"`
	SwapUsedBeforeBytes         uint64   `json:"swap_used_before_bytes"`
	SwapUsedPeakBytes           uint64   `json:"swap_used_peak_bytes"`
	SwapUsedAfterBytes          uint64   `json:"swap_used_after_bytes"`
	SwapPeakDeltaBytes          int64    `json:"swap_peak_delta_bytes"`
	SwapDeltaBytes              int64    `json:"swap_delta_bytes"`
}

type nativeCapacityOutcome struct {
	Admission      string `json:"admission"`
	Reason         string `json:"reason"`
	RequiredBytes  int64  `json:"required_bytes"`
	RequiredMethod string `json:"required_method"`
}

type nativeCapacityPlan struct {
	Schema                   string                     `json:"schema"`
	Issue                    int                        `json:"issue"`
	HardwareGate             string                     `json:"hardware_gate"`
	Artifact                 nativeCapacityFileIdentity `json:"artifact"`
	HostMemoryBytes          uint64                     `json:"host_memory_bytes"`
	Environment              nativeCapacityEnvironment  `json:"environment"`
	CacheDisplacementMinimum int64                      `json:"cache_displacement_minimum_bytes"`
	ServeCommand             []string                   `json:"serve_command"`
	ReadinessEndpoint        string                     `json:"readiness_endpoint"`
	ReadinessDeadlineSeconds int                        `json:"readiness_deadline_seconds"`
	WatcherPort              int                        `json:"watcher_port"`
	WatcherSignal            string                     `json:"watcher_signal"`
	WatcherHardDeadline      int                        `json:"watcher_hard_deadline_seconds"`
	RestorationEndpoints     []string                   `json:"restoration_endpoints"`
	RestorationDeadline      int                        `json:"restoration_deadline_seconds"`
	Measurements             []string                   `json:"measurements"`
}

type nativeCapacityRestoration struct {
	Restored           bool   `json:"restored"`
	OwnerCommandSHA256 string `json:"owner_command_sha256"`
	HealthHTTPStatus   int    `json:"health_http_status"`
	ModelsHTTPStatus   int    `json:"models_http_status"`
	ElapsedMillis      int64  `json:"elapsed_millis"`
	DeadlineSeconds    int    `json:"deadline_seconds"`
}

type nativeCapacityWatcher struct {
	Port                int                       `json:"port"`
	OwnerCommandSHA256  string                    `json:"owner_command_sha256"`
	Signal              string                    `json:"signal"`
	MatchedTERMs        int                       `json:"matched_terms"`
	UnmatchedSignals    int                       `json:"unmatched_signals"`
	ReusedPIDSignals    int                       `json:"reused_pid_signals"`
	WatcherStopped      bool                      `json:"watcher_stopped"`
	HardDeadlineSeconds int                       `json:"hard_deadline_seconds"`
	Restoration         nativeCapacityRestoration `json:"restoration"`
}

type nativeCapacityReceipt struct {
	Schema            string                          `json:"schema"`
	Issue             int                             `json:"issue"`
	BindingSHA256     string                          `json:"binding_sha256"`
	Artifact          nativeCapacityArtifactIdentity  `json:"artifact"`
	Source            nativeCapacitySourceIdentity    `json:"source"`
	Binary            nativeCapacityFileIdentity      `json:"binary"`
	Host              nativeCapacityHostIdentity      `json:"host"`
	Environment       nativeCapacityEnvironment       `json:"environment"`
	CacheDisplacement nativeCapacityCacheDisplacement `json:"cache_displacement"`
	Readiness         nativeCapacityReadiness         `json:"readiness"`
	Execution         nativeCapacityExecution         `json:"execution"`
	Memory            nativeCapacityMemory            `json:"memory"`
	Outcome           nativeCapacityOutcome           `json:"outcome"`
	Watcher           nativeCapacityWatcher           `json:"watcher"`
}

type nativeCapacityReadback struct {
	Schema                     string `json:"schema"`
	Valid                      bool   `json:"valid"`
	Issue                      int    `json:"issue"`
	ArtifactSHA256             string `json:"artifact_sha256"`
	SourceRevision             string `json:"source_revision"`
	BinarySHA256               string `json:"binary_sha256"`
	MaxSampledRSSBytes         uint64 `json:"max_sampled_rss_bytes"`
	TimeMaximumResidentBytes   uint64 `json:"time_maximum_resident_set_bytes"`
	OSPeakMemoryFootprintBytes uint64 `json:"os_peak_memory_footprint_bytes"`
	Admission                  string `json:"admission"`
	RequiredBytes              int64  `json:"required_bytes"`
	SwapPeakDeltaBytes         int64  `json:"swap_peak_delta_bytes"`
	SwapDeltaBytes             int64  `json:"swap_delta_bytes"`
	CacheDisplacementBytes     int64  `json:"cache_displacement_bytes"`
	Restored8090               bool   `json:"restored_8090"`
}

func decodeNativeCapacityReceipt(data []byte) (nativeCapacityReceipt, error) {
	var receipt nativeCapacityReceipt
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil {
		return nativeCapacityReceipt{}, fmt.Errorf("decode capacity receipt: %w", err)
	}
	if err := validateNativeCapacityReceipt(receipt); err != nil {
		return nativeCapacityReceipt{}, err
	}
	return receipt, nil
}

func nativeCapacityBinding(receipt nativeCapacityReceipt) (string, error) {
	receipt.BindingSHA256 = ""
	data, err := json.Marshal(receipt)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return fmt.Sprintf("%x", digest), nil
}

func validateNativeCapacityReceipt(receipt nativeCapacityReceipt) error {
	if receipt.Schema != nativeCapacityReceiptSchema || receipt.Issue != nativeCapacityIssue {
		return fmt.Errorf("capacity receipt identity is %q/#%d", receipt.Schema, receipt.Issue)
	}
	if receipt.Artifact.Bytes != nativeCapacityArtifactBytes || receipt.Artifact.SHA256 != nativeCapacityArtifactSHA256 || receipt.Artifact.Model != "qwen38:27b" || receipt.Artifact.ModelRevision == "" {
		return fmt.Errorf("capacity receipt artifact identity is not the exact Qwen3.8 control")
	}
	if !isSHA256(receipt.Binary.SHA256) || receipt.Binary.Bytes <= 0 || len(receipt.Source.Revision) != 40 || receipt.Source.Dirty {
		return fmt.Errorf("capacity receipt source/binary identity is incomplete or dirty")
	}
	if receipt.Host.ScrubbedID == "" || receipt.Host.SOC != "Apple M3 Pro" || receipt.Host.GPUCores != 18 || receipt.Host.MemoryBytes != nativeCapacityHostBytes || receipt.Host.Architecture != "darwin/arm64" || receipt.Host.OSKernel == "" {
		return fmt.Errorf("capacity receipt host is not the exact scrubbed 36 GiB M3 Pro envelope")
	}
	if err := validateNativeCapacityEnvironment(receipt.Environment); err != nil {
		return err
	}
	if err := validateNativeCapacityDisplacement(receipt.CacheDisplacement, receipt.Host.MemoryBytes); err != nil {
		return err
	}
	if receipt.Readiness.Endpoint != "http://127.0.0.1:18971/v1/models" || receipt.Readiness.HTTPStatus != 200 || receipt.Readiness.ElapsedMillis <= 0 || receipt.Readiness.DeadlineSeconds <= 0 || receipt.Readiness.ElapsedMillis >= int64(receipt.Readiness.DeadlineSeconds)*1000 {
		return fmt.Errorf("capacity receipt has incomplete or late readiness evidence")
	}
	if receipt.Execution.Engine != "inkernel" || receipt.Execution.Backend != "metal" || receipt.Execution.ForwardPath != "metal/qwen35-hybrid-session-v1" || receipt.Execution.FallbackCount != 0 || receipt.Execution.LlamaCPPUsed || receipt.Execution.IdentitySource == "" {
		return fmt.Errorf("capacity receipt does not prove fak-native inkernel/Metal with zero fallback")
	}
	if err := validateNativeCapacityMemory(receipt.Memory); err != nil {
		return err
	}
	if err := validateNativeCapacityOutcome(receipt.Outcome, receipt.Host.MemoryBytes, receipt.Memory.SwapPeakDeltaBytes); err != nil {
		return err
	}
	if err := validateNativeCapacityWatcher(receipt.Watcher); err != nil {
		return err
	}
	privacyBytes, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	for _, forbidden := range []string{"/Users/", "anthony", "Hardware UUID", "Serial Number", "credential", "token="} {
		if strings.Contains(string(privacyBytes), forbidden) {
			return fmt.Errorf("capacity receipt contains private marker %q", forbidden)
		}
	}
	binding, err := nativeCapacityBinding(receipt)
	if err != nil || receipt.BindingSHA256 != binding {
		return fmt.Errorf("capacity receipt binding digest mismatch")
	}
	return nil
}

func validateNativeCapacityEnvironment(environment nativeCapacityEnvironment) error {
	present := append([]string(nil), environment.Present...)
	absent := append([]string(nil), environment.Absent...)
	sort.Strings(present)
	sort.Strings(absent)
	wantPresent := append([]string(nil), nativeCapacityPresentEnvironment...)
	wantAbsent := append([]string(nil), nativeCapacityAbsentEnvironment...)
	sort.Strings(wantPresent)
	sort.Strings(wantAbsent)
	if strings.Join(present, "\x00") != strings.Join(wantPresent, "\x00") || strings.Join(absent, "\x00") != strings.Join(wantAbsent, "\x00") {
		return fmt.Errorf("capacity receipt environment is not the literal no-FAK_Q4K_FREE_CPU control")
	}
	return nil
}

func validateNativeCapacityDisplacement(displacement nativeCapacityCacheDisplacement, hostBytes uint64) error {
	var total int64
	seen := map[string]bool{}
	for _, source := range displacement.Sources {
		if source.ScrubbedID == "" || seen[source.ScrubbedID] || source.Bytes <= 0 || !isSHA256(source.SHA256) {
			return fmt.Errorf("capacity receipt cache displacement source is incomplete or duplicated")
		}
		seen[source.ScrubbedID] = true
		total += source.Bytes
	}
	if total != displacement.TotalBytes || displacement.TotalBytes <= int64(hostBytes) || displacement.TotalBytes <= 36<<30 {
		return fmt.Errorf("capacity receipt cache displacement is not a real >36 GiB read")
	}
	return nil
}

func validateNativeCapacityMemory(memory nativeCapacityMemory) error {
	if len(memory.RSSSamplesBytes) == 0 || len(memory.OSFootprintSamplesBytes) == 0 || memory.MaxSampledRSSBytes == 0 || memory.TimeMaximumResidentSetBytes == 0 || memory.OSPeakMemoryFootprintBytes == 0 {
		return fmt.Errorf("capacity receipt lacks RSS, /usr/bin/time -l, or OS footprint evidence")
	}
	if maxUint64(memory.RSSSamplesBytes) != memory.MaxSampledRSSBytes || maxUint64(memory.OSFootprintSamplesBytes) != memory.OSPeakMemoryFootprintBytes {
		return fmt.Errorf("capacity receipt memory maxima do not recompute from raw samples")
	}
	wantPeakDelta := int64(memory.SwapUsedPeakBytes) - int64(memory.SwapUsedBeforeBytes)
	wantDelta := int64(memory.SwapUsedAfterBytes) - int64(memory.SwapUsedBeforeBytes)
	if memory.SwapUsedPeakBytes < memory.SwapUsedAfterBytes || memory.SwapPeakDeltaBytes != wantPeakDelta || memory.SwapDeltaBytes != wantDelta {
		return fmt.Errorf("capacity receipt swap evidence is inconsistent")
	}
	return nil
}

func validateNativeCapacityOutcome(outcome nativeCapacityOutcome, hostBytes uint64, swapPeakDelta int64) error {
	if swapPeakDelta <= 0 || outcome.Admission != "refused" || outcome.Reason != "positive-swap-delta" {
		return fmt.Errorf("capacity receipt outcome must fail closed on positive swap growth")
	}
	want := nativeCapacityRequiredBytes(hostBytes, swapPeakDelta)
	if outcome.RequiredBytes != want || outcome.RequiredMethod != "ceil-gib(host-memory-bytes+peak-swap-delta-bytes)" {
		return fmt.Errorf("capacity receipt required bound does not recompute from measured swap growth")
	}
	return nil
}

func nativeCapacityRequiredBytes(hostBytes uint64, swapPeakDelta int64) int64 {
	const gib = uint64(1 << 30)
	if hostBytes == 0 || swapPeakDelta <= 0 || uint64(swapPeakDelta) > ^uint64(0)-hostBytes {
		return 0
	}
	required := hostBytes + uint64(swapPeakDelta)
	return int64(((required + gib - 1) / gib) * gib)
}

func maxUint64(values []uint64) uint64 {
	var max uint64
	for _, value := range values {
		if value > max {
			max = value
		}
	}
	return max
}

func validateNativeCapacityWatcher(watcher nativeCapacityWatcher) error {
	restored := watcher.Restoration
	if watcher.Port != 8090 || !isSHA256(watcher.OwnerCommandSHA256) || watcher.Signal != "TERM" || watcher.MatchedTERMs < 1 || watcher.UnmatchedSignals != 0 || watcher.ReusedPIDSignals != 0 || !watcher.WatcherStopped || watcher.HardDeadlineSeconds <= 0 {
		return fmt.Errorf("capacity receipt lacks bounded exact-owner TERM-only watcher evidence")
	}
	if !restored.Restored || restored.OwnerCommandSHA256 != watcher.OwnerCommandSHA256 || restored.HealthHTTPStatus != 200 || restored.ModelsHTTPStatus != 200 || restored.ElapsedMillis <= 0 || restored.DeadlineSeconds <= 0 || restored.ElapsedMillis >= int64(restored.DeadlineSeconds)*1000 {
		return fmt.Errorf("capacity receipt lacks exact-owner 8090 restoration evidence")
	}
	return nil
}

func isSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, r := range value {
		if !strings.ContainsRune("0123456789abcdef", r) {
			return false
		}
	}
	return true
}

func nativeCapacityReadbackFor(receipt nativeCapacityReceipt) nativeCapacityReadback {
	return nativeCapacityReadback{
		Schema: nativeCapacityReadbackSchema, Valid: true, Issue: receipt.Issue,
		ArtifactSHA256: receipt.Artifact.SHA256, SourceRevision: receipt.Source.Revision,
		BinarySHA256: receipt.Binary.SHA256, MaxSampledRSSBytes: receipt.Memory.MaxSampledRSSBytes,
		TimeMaximumResidentBytes:   receipt.Memory.TimeMaximumResidentSetBytes,
		OSPeakMemoryFootprintBytes: receipt.Memory.OSPeakMemoryFootprintBytes,
		Admission:                  receipt.Outcome.Admission, RequiredBytes: receipt.Outcome.RequiredBytes,
		SwapPeakDeltaBytes: receipt.Memory.SwapPeakDeltaBytes, SwapDeltaBytes: receipt.Memory.SwapDeltaBytes,
		CacheDisplacementBytes: receipt.CacheDisplacement.TotalBytes,
		Restored8090:           receipt.Watcher.Restoration.Restored,
	}
}

func nativeCapacityCapturePlan() nativeCapacityPlan {
	return nativeCapacityPlan{
		Schema: nativeCapacityPlanSchema, Issue: nativeCapacityIssue,
		HardwareGate:    "blocked-until-issues-9044-and-9020-finish-and-operator-clears-exclusive-Mac-window",
		Artifact:        nativeCapacityFileIdentity{Bytes: nativeCapacityArtifactBytes, SHA256: nativeCapacityArtifactSHA256},
		HostMemoryBytes: nativeCapacityHostBytes,
		Environment: nativeCapacityEnvironment{
			Present: append([]string(nil), nativeCapacityPresentEnvironment...),
			Absent:  append([]string(nil), nativeCapacityAbsentEnvironment...),
		},
		CacheDisplacementMinimum: int64(nativeCapacityHostBytes) + 1,
		ServeCommand: []string{
			"/usr/bin/time", "-l", "<exact-current-source-binary>", "serve", "--addr", "127.0.0.1:18971",
			"--gguf", "<exact-gguf>", "--model", "qwen38:27b", "--metal", "--context-budget-tokens", "4096",
		},
		ReadinessEndpoint: "http://127.0.0.1:18971/v1/models", ReadinessDeadlineSeconds: 420,
		WatcherPort: 8090, WatcherSignal: "TERM", WatcherHardDeadline: 600,
		RestorationEndpoints: []string{"http://127.0.0.1:8090/health", "http://127.0.0.1:8090/v1/models"},
		RestorationDeadline:  120,
		Measurements: []string{
			"artifact/source/binary SHA-256 and bytes", "sequential cache-source SHA-256 and bytes (>36 GiB total)",
			"periodic ps RSS samples", "/usr/bin/time -l maximum resident set size", "periodic footprint(1) samples",
			"vm.swapusage before and after", "inkernel/Metal/fak-native identity with zero fallback",
			"exact-owner TERM-only watcher event counts and post-watcher 8090 restoration probes",
		},
	}
}
