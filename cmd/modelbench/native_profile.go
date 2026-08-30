package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	nativeProfileReceiptSchema = "fak-native-metal-profile-receipt/v1"
	nativeProfileArtifactBytes = int64(17106775008)
	nativeProfileUnset         = "<unset>"
)

var nativeProfileRequiredEnvironment = map[string]string{
	"FAK_METAL_STREAM_Q4K": "1",
	"FAK_Q4K":              "1",
}

const nativeControlGGUFMMap = "FAK_GGUF_MMAP"

// nativeProfileDeniedEnvironment is the complete set of process controls read by the Qwen3.8
// resident-Q4_K forward, its attention/KV helpers, and its worker scheduler. The profile contract
// pins their default behavior by requiring them to be absent. FAK_Q4K_FREE_CPU and
// FAK_METAL_Q8_UPLOAD are deliberately denied: either one changes residency and must never
// masquerade as the documented all-Metal, no-override control run.
var nativeProfileDeniedEnvironment = []string{
	"FAK_ARM_TILE",
	"FAK_BUDGET",
	"FAK_FDOT3_SIMD",
	"FAK_FDOT3_SIMD_MINB",
	"FAK_GDN_BATCHED",
	"FAK_HIDDEN_STEER",
	"FAK_HIDDEN_STEER_ALPHA",
	"FAK_HIDDEN_STEER_LAYER",
	"FAK_HIDDEN_STEER_POS",
	"FAK_HIDDEN_TAP",
	"FAK_HIDDEN_TAP_OPS",
	"FAK_HIDDEN_TAP_POS",
	"FAK_HYBRID_KV",
	"FAK_KQ_INT8",
	"FAK_METAL_DECODE",
	"FAK_METAL_Q8_UPLOAD",
	"FAK_METAL_RESIDENT",
	"FAK_PAGED_KV",
	"FAK_PAGED_KV_BLOCK_TOKENS",
	"FAK_PAR_SHARDS",
	"FAK_PAR_SPIN",
	"FAK_PREFILL_NO_ATTN",
	"FAK_PREFILL_NO_GDN",
	"FAK_Q4K_FREE_CPU",
	"FAK_Q4K_MM",
	"FAK_Q8_GEMM_GROUP",
	"FAK_QATTN_GQA",
	"FAK_QGEMM",
	"FAK_QGEMM_GROUP",
	"FAK_QGEMM_GROUP_MAXP",
	"FAK_QKERNEL",
	"FAK_QPROFILE",
	"FAK_QWEN35_PREFILL_TOKEN_LOOP",
	"FAK_SAXPY3_SIMD_MINB",
	"FAK_SAXPY3_SIMD_MINPOS",
	"FAK_WORKERS",
	"GOMAXPROCS",
}

const (
	nativeControlFlagBudget           = "flag:-budget"
	nativeControlLogicalCPUs          = "runtime:logical_cpus"
	nativeControlGOMAXPROCS           = "runtime:gomaxprocs"
	nativeControlWorkers              = "runtime:matmul_workers"
	nativeControlQ8Workers            = "runtime:q8_decode_workers"
	nativeControlWorkerBudget         = "runtime:worker_budget"
	nativeProfileDecodeHandoffControl = "flag:-native-performance-qwen35-decode-handoff"
)

func nativeProfileControlEnvironment(lookup func(string) (string, bool), environ []string, budget float64, handoffMode model.Qwen35DecodeHandoffMode) (map[string]string, error) {
	if budget != 0 {
		return nil, fmt.Errorf("native performance profile unavailable: -budget must be 0, got %g", budget)
	}
	controls := make(map[string]string, len(nativeProfileRequiredEnvironment)+len(nativeProfileDeniedEnvironment)+8)
	for key, want := range nativeProfileRequiredEnvironment {
		got, ok := lookup(key)
		if !ok || got != want {
			return nil, fmt.Errorf("native performance profile unavailable: %s must equal %s", key, want)
		}
		controls[key] = got
	}
	mmapControl, ok := lookup(nativeControlGGUFMMap)
	if !ok || (mmapControl != "0" && mmapControl != "1") {
		return nil, fmt.Errorf("native performance profile unavailable: %s must equal typed 0 or 1", nativeControlGGUFMMap)
	}
	controls[nativeControlGGUFMMap] = mmapControl
	sequenceSelector, sequenceSelectorSet := lookup(nativeProfileSequenceSelector)
	if sequenceSelectorSet && sequenceSelector != nativeProfileSelectorOff && sequenceSelector != nativeProfileSelectorOn {
		return nil, fmt.Errorf("native performance profile unavailable: %s must equal typed %s or %s", nativeProfileSequenceSelector, nativeProfileSelectorOff, nativeProfileSelectorOn)
	}
	if !sequenceSelectorSet {
		sequenceSelector = nativeProfileUnset
	}
	controls[nativeProfileSequenceSelector] = sequenceSelector
	if handoffMode != model.Qwen35DecodeHandoffAuto && sequenceSelector != nativeProfileSelectorOn {
		return nil, fmt.Errorf("native performance profile unavailable: decode handoff %s requires %s=%s", handoffMode, nativeProfileSequenceSelector, nativeProfileSelectorOn)
	}
	controls[nativeProfileDecodeHandoffControl] = handoffMode.String()
	for _, key := range nativeProfileDeniedEnvironment {
		if got, ok := lookup(key); ok {
			return nil, fmt.Errorf("native performance profile unavailable: %s override is not allowed (got %q)", key, got)
		}
		controls[key] = nativeProfileUnset
	}
	// Refuse unknown FAK_* declarations too: otherwise a newly added process knob could alter a
	// run before the receipt schema learns to bind it. This is intentionally fail-closed.
	for _, declaration := range environ {
		key, _, _ := strings.Cut(declaration, "=")
		if !strings.HasPrefix(key, "FAK_") {
			continue
		}
		if _, ok := nativeProfileRequiredEnvironment[key]; ok {
			continue
		}
		if key == nativeControlGGUFMMap {
			continue
		}
		if key == nativeProfileSequenceSelector {
			continue
		}
		return nil, fmt.Errorf("native performance profile unavailable: unrecognized %s override is not allowed", key)
	}
	controls[nativeControlFlagBudget] = "0"
	controls[nativeControlLogicalCPUs] = strconv.Itoa(runtime.NumCPU())
	controls[nativeControlGOMAXPROCS] = strconv.Itoa(runtime.GOMAXPROCS(0))
	controls[nativeControlWorkers] = strconv.Itoa(model.NumWorkers())
	controls[nativeControlQ8Workers] = strconv.Itoa(model.Q8DecodeWorkers())
	controls[nativeControlWorkerBudget] = model.WorkerBudget()
	if err := validateNativeProfileControls(controls); err != nil {
		return nil, err
	}
	return controls, nil
}

func validateNativeProfileControls(controls map[string]string) error {
	legacyLen := len(nativeProfileRequiredEnvironment) + len(nativeProfileDeniedEnvironment) + 7
	extra := 0
	if _, ok := controls[nativeProfileSequenceSelector]; ok {
		extra++
	}
	if _, ok := controls[nativeProfileDecodeHandoffControl]; ok {
		extra++
	}
	if len(controls) != legacyLen+extra {
		return fmt.Errorf("native performance control receipt has %d fields, want %d for its typed extensions", len(controls), legacyLen+extra)
	}
	for key, want := range nativeProfileRequiredEnvironment {
		if controls[key] != want {
			return fmt.Errorf("required control %s was not captured as %s", key, want)
		}
	}
	if controls[nativeControlGGUFMMap] != "0" && controls[nativeControlGGUFMMap] != "1" {
		return fmt.Errorf("required control %s was not captured as typed 0 or 1", nativeControlGGUFMMap)
	}
	if selector, ok := controls[nativeProfileSequenceSelector]; ok && selector != nativeProfileUnset && selector != nativeProfileSelectorOff && selector != nativeProfileSelectorOn {
		return fmt.Errorf("selector control %s was not captured as typed %s or %s", nativeProfileSequenceSelector, nativeProfileSelectorOff, nativeProfileSelectorOn)
	}
	if value, ok := controls[nativeProfileDecodeHandoffControl]; ok {
		var mode model.Qwen35DecodeHandoffMode
		if err := mode.Set(value); err != nil {
			return fmt.Errorf("decode handoff control: %w", err)
		}
		if mode != model.Qwen35DecodeHandoffAuto && controls[nativeProfileSequenceSelector] != nativeProfileSelectorOn {
			return fmt.Errorf("decode handoff %s requires sequence selector ON", mode)
		}
	}
	for _, key := range nativeProfileDeniedEnvironment {
		if controls[key] != nativeProfileUnset {
			return fmt.Errorf("denied control %s was not captured as unset", key)
		}
	}
	if controls[nativeControlFlagBudget] != "0" || controls[nativeControlWorkerBudget] != "default(GOMAXPROCS)" {
		return fmt.Errorf("worker budget controls are not the default envelope")
	}
	for _, key := range []string{nativeControlLogicalCPUs, nativeControlGOMAXPROCS, nativeControlWorkers, nativeControlQ8Workers} {
		value, err := strconv.Atoi(controls[key])
		if err != nil || value <= 0 {
			return fmt.Errorf("control %s is not a positive integer", key)
		}
	}
	return nil
}

type nativeFileIdentity struct {
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type nativeArtifactIdentity struct {
	nativeFileIdentity
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
}

type nativeHostIdentity struct {
	GOOS                 string `json:"goos"`
	GOARCH               string `json:"goarch"`
	CPU                  string `json:"cpu"`
	MetalDevice          string `json:"metal_device"`
	GPUCores             int    `json:"gpu_cores"`
	MemoryBytes          uint64 `json:"memory_bytes"`
	MetalWorkingSetBytes uint64 `json:"metal_working_set_bytes"`
}

type nativeSourceIdentity struct {
	Revision   string `json:"revision"`
	Modified   bool   `json:"modified"`
	DiffBytes  int    `json:"diff_bytes,omitempty"`
	DiffSHA256 string `json:"diff_sha256,omitempty"`
}

type nativeProfileReceipt struct {
	Schema              string                                 `json:"schema"`
	BindingSHA256       string                                 `json:"binding_sha256"`
	ProfileSHA256       string                                 `json:"profile_sha256"`
	EnvelopeID          string                                 `json:"envelope_id"`
	Artifact            nativeArtifactIdentity                 `json:"artifact"`
	ModelConfig         map[string]any                         `json:"model_config"`
	ModelConfigSHA256   string                                 `json:"model_config_sha256"`
	Host                nativeHostIdentity                     `json:"host"`
	Source              nativeSourceIdentity                   `json:"source"`
	Binary              nativeFileIdentity                     `json:"binary"`
	Controls            map[string]string                      `json:"controls"`
	Execution           metalgemm.ExecutionReceipt             `json:"execution"`
	Fallbacks           model.MetalFallbackReceipt             `json:"fallbacks"`
	Q4KResidency        *model.Q4KResidencyReceipt             `json:"q4k_residency,omitempty"`
	Qwen35DecodeHandoff *model.Qwen35DecodeHandoffReceipt      `json:"qwen35_decode_handoff,omitempty"`
	CachePhaseLatency   *modelperfobs.CachePhaseLatencyReceipt `json:"cache_phase_latency,omitempty"`
}

func sha256JSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum), nil
}

func fileIdentity(path string) (nativeFileIdentity, error) {
	f, err := os.Open(path)
	if err != nil {
		return nativeFileIdentity{}, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nativeFileIdentity{}, err
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nativeFileIdentity{}, err
	}
	return nativeFileIdentity{Bytes: info.Size(), SHA256: fmt.Sprintf("%x", h.Sum(nil))}, nil
}

func profileEnvelopeByID(graph nativeperf.Graph, id string) (nativeperf.Envelope, error) {
	for _, envelope := range graph.Envelopes {
		if envelope.ID == id {
			return envelope, nil
		}
	}
	return nativeperf.Envelope{}, fmt.Errorf("unknown native-performance envelope %q", id)
}

func exactMetalProfileEnvelope(graph nativeperf.Graph, artifact nativeFileIdentity) (nativeperf.Envelope, error) {
	for _, envelope := range graph.Envelopes {
		if envelope.Backend == "metal" && envelope.ArtifactSHA256 == artifact.SHA256 {
			if artifact.Bytes != nativeProfileArtifactBytes {
				return nativeperf.Envelope{}, fmt.Errorf("artifact bytes = %d, want %d", artifact.Bytes, nativeProfileArtifactBytes)
			}
			return envelope, nil
		}
	}
	return nativeperf.Envelope{}, fmt.Errorf("artifact SHA-256 %q is not a pinned Metal envelope", artifact.SHA256)
}

func commandOutput(name string, args ...string) (string, error) {
	b, err := exec.Command(name, args...).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

func captureNativeHost() (nativeHostIdentity, error) {
	host := nativeHostIdentity{GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, MetalDevice: metalgemm.DeviceName()}
	if host.GOOS != "darwin" || host.GOARCH != "arm64" || host.MetalDevice == "" {
		return nativeHostIdentity{}, fmt.Errorf("native Metal host unavailable: %s/%s device=%q", host.GOOS, host.GOARCH, host.MetalDevice)
	}
	var err error
	if host.CPU, err = commandOutput("sysctl", "-n", "machdep.cpu.brand_string"); err != nil || host.CPU == "" {
		return nativeHostIdentity{}, fmt.Errorf("CPU identity: %w", err)
	}
	memory, err := commandOutput("sysctl", "-n", "hw.memsize")
	if err != nil {
		return nativeHostIdentity{}, fmt.Errorf("host memory: %w", err)
	}
	if host.MemoryBytes, err = strconv.ParseUint(memory, 10, 64); err != nil || host.MemoryBytes == 0 {
		return nativeHostIdentity{}, fmt.Errorf("host memory value %q: %w", memory, err)
	}
	var displays struct {
		GPUs []struct {
			Name  string `json:"_name"`
			Model string `json:"sppci_model"`
			Cores string `json:"sppci_cores"`
		} `json:"SPDisplaysDataType"`
	}
	displayJSON, err := exec.Command("system_profiler", "SPDisplaysDataType", "-json").Output()
	if err != nil {
		return nativeHostIdentity{}, fmt.Errorf("GPU identity unavailable: %w", err)
	}
	if err := json.Unmarshal(displayJSON, &displays); err != nil {
		return nativeHostIdentity{}, fmt.Errorf("GPU identity decode: %w", err)
	}
	for _, gpu := range displays.GPUs {
		if gpu.Name == host.MetalDevice || gpu.Model == host.MetalDevice {
			host.GPUCores, err = strconv.Atoi(gpu.Cores)
			break
		}
	}
	if err != nil || host.GPUCores <= 0 {
		return nativeHostIdentity{}, fmt.Errorf("GPU core identity unavailable for %q", host.MetalDevice)
	}
	if host.MetalWorkingSetBytes, _ = metalgemm.DeviceMemoryTotal(); host.MetalWorkingSetBytes == 0 {
		return nativeHostIdentity{}, fmt.Errorf("Metal working-set capacity unavailable")
	}
	return host, nil
}

func captureNativeBuild() (nativeSourceIdentity, nativeFileIdentity, error) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return nativeSourceIdentity{}, nativeFileIdentity{}, fmt.Errorf("Go build identity unavailable")
	}
	var source nativeSourceIdentity
	revisionFound, modifiedFound := false, false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			source.Revision, revisionFound = setting.Value, setting.Value != ""
		case "vcs.modified":
			source.Modified, modifiedFound = setting.Value == "true", true
		}
	}
	if !revisionFound || !modifiedFound {
		return nativeSourceIdentity{}, nativeFileIdentity{}, fmt.Errorf("binary lacks vcs.revision or vcs.modified build settings")
	}
	if source.Modified {
		cmd := exec.Command("git", "diff", "--no-ext-diff", "--binary", source.Revision, "--")
		windowgate.ConfigureBackgroundCommand(cmd)
		diff, err := cmd.Output()
		if err != nil || len(diff) == 0 {
			return nativeSourceIdentity{}, nativeFileIdentity{}, fmt.Errorf("modified source diff identity unavailable: %w", err)
		}
		diffSum := sha256.Sum256(diff)
		source.DiffBytes = len(diff)
		source.DiffSHA256 = fmt.Sprintf("%x", diffSum)
	}
	executable, err := os.Executable()
	if err != nil {
		return nativeSourceIdentity{}, nativeFileIdentity{}, err
	}
	binary, err := fileIdentity(executable)
	return source, binary, err
}

func validateNativeHost(envelope nativeperf.Envelope, host nativeHostIdentity) error {
	wantHardware := fmt.Sprintf("%s, %d-GPU-core", host.CPU, host.GPUCores)
	if host.GOOS != "darwin" || host.GOARCH != "arm64" || host.CPU == "" || host.MetalDevice != host.CPU || host.GPUCores <= 0 || wantHardware != envelope.Hardware {
		return fmt.Errorf("host identity %+v does not match envelope hardware %q", host, envelope.Hardware)
	}
	if host.MemoryBytes != uint64(envelope.MemoryGiB)*(1<<30) || host.MetalWorkingSetBytes == 0 {
		return fmt.Errorf("host memory identity physical=%d Metal=%d does not match %d GiB envelope", host.MemoryBytes, host.MetalWorkingSetBytes, envelope.MemoryGiB)
	}
	return nil
}

func nativeReceiptBinding(receipt nativeProfileReceipt) (string, error) {
	receipt.BindingSHA256 = ""
	return sha256JSON(receipt)
}

func nativeProfileExpectedForwardPath(envelopePath, selector string) (string, error) {
	switch selector {
	case "", nativeProfileUnset, nativeProfileSelectorOff:
		return envelopePath, nil
	case nativeProfileSelectorOn:
		return model.Qwen35MetalGDNSequenceForwardPath, nil
	default:
		return "", fmt.Errorf("selector control %s has untyped value %q", nativeProfileSequenceSelector, selector)
	}
}

// nativeProfileExecutedForwardPath binds a capture label to observed session state. A selector
// declaration alone is not evidence that the candidate route ran, and an OFF capture must prove
// that the opt-in route was not selected.
func nativeProfileExecutedForwardPath(envelopePath, selector string, sequenceExecuted bool) (string, error) {
	expected, err := nativeProfileExpectedForwardPath(envelopePath, selector)
	if err != nil {
		return "", err
	}
	wantsSequence := selector == nativeProfileSelectorOn
	if wantsSequence != sequenceExecuted {
		return "", fmt.Errorf("selector %s=%q execution mismatch: sequence executed=%t", nativeProfileSequenceSelector, selector, sequenceExecuted)
	}
	return expected, nil
}

func validateNativeProfileForControls(profile nativeperf.ProfileBundle, controls map[string]string) error {
	envelope, err := profileEnvelopeByID(nativeperf.ActiveGraph(), profile.EnvelopeID)
	if err != nil {
		return err
	}
	expected, err := nativeProfileExpectedForwardPath(envelope.ForwardPath, controls[nativeProfileSequenceSelector])
	if err != nil {
		return err
	}
	if profile.Execution.ForwardPath != expected {
		return fmt.Errorf("profile forward path %q does not match selector route %q", profile.Execution.ForwardPath, expected)
	}
	// The shared v1 validator pins the historical envelope route. Validate a shallow canonical
	// view after independently admitting exactly the one typed selector route delta above.
	canonical := profile
	canonical.Execution.ForwardPath = envelope.ForwardPath
	return nativeperf.ValidateProfile(nativeperf.ActiveGraph(), canonical)
}

func validateNativeProfileReceipt(profileBytes []byte, profile nativeperf.ProfileBundle, receipt nativeProfileReceipt) error {
	if receipt.Schema != nativeProfileReceiptSchema {
		return fmt.Errorf("unexpected native profile receipt schema %q", receipt.Schema)
	}
	if err := validateNativeProfileForControls(profile, receipt.Controls); err != nil {
		return err
	}
	envelope, err := profileEnvelopeByID(nativeperf.ActiveGraph(), profile.EnvelopeID)
	if err != nil {
		return err
	}
	profileSum := sha256.Sum256(profileBytes)
	if receipt.ProfileSHA256 != fmt.Sprintf("%x", profileSum) || receipt.EnvelopeID != profile.EnvelopeID {
		return fmt.Errorf("profile digest or envelope binding mismatch")
	}
	if receipt.Artifact.SHA256 != envelope.ArtifactSHA256 || receipt.Artifact.Bytes != nativeProfileArtifactBytes || receipt.Artifact.Model != envelope.Model || receipt.Artifact.ModelRevision != envelope.ModelRevision {
		return fmt.Errorf("artifact identity does not match envelope")
	}
	if len(receipt.ModelConfig) == 0 {
		return fmt.Errorf("model config is empty")
	}
	configSHA, err := sha256JSON(receipt.ModelConfig)
	if err != nil || configSHA != receipt.ModelConfigSHA256 {
		return fmt.Errorf("model config digest mismatch")
	}
	if err := validateNativeHost(envelope, receipt.Host); err != nil {
		return err
	}
	if receipt.Source.Revision == "" || receipt.Binary.Bytes <= 0 || len(receipt.Binary.SHA256) != 64 {
		return fmt.Errorf("source revision or binary identity incomplete")
	}
	if receipt.Source.Modified && (receipt.Source.DiffBytes <= 0 || len(receipt.Source.DiffSHA256) != 64) {
		return fmt.Errorf("modified source lacks an exact diff identity")
	}
	if !receipt.Source.Modified && (receipt.Source.DiffBytes != 0 || receipt.Source.DiffSHA256 != "") {
		return fmt.Errorf("clean source carries a contradictory diff identity")
	}
	if err := validateNativeProfileControls(receipt.Controls); err != nil {
		return err
	}
	handoffMode := model.Qwen35DecodeHandoffAuto
	if value, ok := receipt.Controls[nativeProfileDecodeHandoffControl]; ok {
		if err := handoffMode.Set(value); err != nil {
			return err
		}
		if receipt.Qwen35DecodeHandoff == nil {
			return fmt.Errorf("decode handoff control lacks a session receipt")
		}
	}
	if receipt.Qwen35DecodeHandoff != nil {
		if receipt.Qwen35DecodeHandoff.Mode != handoffMode {
			return fmt.Errorf("decode handoff receipt mode %s does not match control %s", receipt.Qwen35DecodeHandoff.Mode, handoffMode)
		}
		if err := model.ValidateQwen35DecodeHandoffReceipt(*receipt.Qwen35DecodeHandoff); err != nil {
			return fmt.Errorf("decode handoff receipt: %w", err)
		}
	}
	if err := metalgemm.ValidateExecutionReceipt(receipt.Execution); err != nil {
		return err
	}
	if err := model.ValidateMetalFallbackReceipt(receipt.Fallbacks); err != nil {
		return err
	}
	if receipt.Q4KResidency != nil {
		if err := model.ValidateQ4KResidencyReceipt(*receipt.Q4KResidency); err != nil {
			return err
		}
		if receipt.Q4KResidency.FAKGGUFMMap != receipt.Controls[nativeControlGGUFMMap] {
			return fmt.Errorf("Q4_K residency control does not match native profile controls")
		}
	}
	if receipt.Fallbacks.PromisedCPUFallbacks != profile.Execution.FallbackCount {
		return fmt.Errorf("raw fallback total does not match v1 execution identity")
	}
	if profile.Metal == nil || receipt.Execution.Counters.CommandBuffers != profile.Metal.CommandBuffers || receipt.Execution.Counters.Encoders != profile.Metal.Encoders || receipt.Execution.Counters.DispatchMilliseconds != profile.Metal.DispatchMilliseconds || receipt.Execution.Counters.WaitMilliseconds != profile.Metal.WaitMilliseconds {
		return fmt.Errorf("raw execution totals do not match v1 Metal counter block")
	}
	binding, err := nativeReceiptBinding(receipt)
	if err != nil || binding != receipt.BindingSHA256 {
		return fmt.Errorf("source/binary/artifact/event binding digest mismatch")
	}
	return nil
}

func nativeReceiptPath(profilePath string) string {
	ext := filepath.Ext(profilePath)
	return strings.TrimSuffix(profilePath, ext) + ".receipt.json"
}

func runNativeProfileReadback(profilePath string) error {
	profileBytes, err := os.ReadFile(profilePath)
	if err != nil {
		return err
	}
	profile, err := nativeperf.DecodeProfile(profileBytes)
	if err != nil {
		return err
	}
	receiptBytes, err := os.ReadFile(nativeReceiptPath(profilePath))
	if err != nil {
		return err
	}
	var receipt nativeProfileReceipt
	if err := json.Unmarshal(receiptBytes, &receipt); err != nil {
		return err
	}
	return validateNativeProfileReceipt(profileBytes, profile, receipt)
}
