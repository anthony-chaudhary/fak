package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

const (
	Qwen38VulkanDecodePacketSchema  = "fak/qwen38-vulkan-decode-packet/v1"
	Qwen38VulkanDecodeReceiptSchema = "fak/qwen38-vulkan-decode-receipt/v2"

	Qwen38VulkanDecodeGGUFSHA256 = "7E78DA5D7E3AE28D178121F58646953305F3E5BD3CB46F4A75584E8B6C6FE169"
	Qwen38VulkanDecodeBackend    = "vulkan"
	Qwen38VulkanDecodeRuntime    = "native"
	Qwen38VulkanDecodeFallback   = "none"
)

var qwen38VulkanGitCommitRE = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)

// Qwen38VulkanDecodePacket freezes the identity and token boundary of one
// deterministic native decode request. PromptTokenIDs are the tokenizer output,
// so replay does not depend on prompt rendering or tokenizer revisions.
type Qwen38VulkanDecodePacket struct {
	Schema              string  `json:"schema"`
	ModelGGUFSHA256     string  `json:"model_gguf_sha256"`
	PromptTokenIDs      []int32 `json:"prompt_token_ids"`
	GeneratedTokenLimit int     `json:"generated_token_limit"`
	Backend             string  `json:"backend"`
	Runtime             string  `json:"runtime"`
	Fallback            string  `json:"fallback"`
}

// NewQwen38VulkanDecodePacket returns the canonical engine and model identity.
// The caller supplies only the exact prompt tokens and positive decode boundary.
func NewQwen38VulkanDecodePacket(promptTokenIDs []int32, generatedTokenLimit int) Qwen38VulkanDecodePacket {
	return Qwen38VulkanDecodePacket{
		Schema:              Qwen38VulkanDecodePacketSchema,
		ModelGGUFSHA256:     Qwen38VulkanDecodeGGUFSHA256,
		PromptTokenIDs:      slices.Clone(promptTokenIDs),
		GeneratedTokenLimit: generatedTokenLimit,
		Backend:             Qwen38VulkanDecodeBackend,
		Runtime:             Qwen38VulkanDecodeRuntime,
		Fallback:            Qwen38VulkanDecodeFallback,
	}
}

func (p Qwen38VulkanDecodePacket) Validate() error {
	if p.Schema != Qwen38VulkanDecodePacketSchema {
		return fmt.Errorf("qwen3.8 Vulkan packet schema %q, want %q", p.Schema, Qwen38VulkanDecodePacketSchema)
	}
	if p.ModelGGUFSHA256 != Qwen38VulkanDecodeGGUFSHA256 {
		return fmt.Errorf("qwen3.8 Vulkan packet model digest %q, want %q", p.ModelGGUFSHA256, Qwen38VulkanDecodeGGUFSHA256)
	}
	if len(p.PromptTokenIDs) == 0 {
		return errors.New("qwen3.8 Vulkan packet requires prompt token IDs")
	}
	if p.GeneratedTokenLimit <= 0 {
		return fmt.Errorf("qwen3.8 Vulkan generated-token limit must be positive, got %d", p.GeneratedTokenLimit)
	}
	if p.Backend != Qwen38VulkanDecodeBackend || p.Runtime != Qwen38VulkanDecodeRuntime || p.Fallback != Qwen38VulkanDecodeFallback {
		return fmt.Errorf("qwen3.8 Vulkan packet requires backend=%s runtime=%s fallback=%s, got backend=%q runtime=%q fallback=%q", Qwen38VulkanDecodeBackend, Qwen38VulkanDecodeRuntime, Qwen38VulkanDecodeFallback, p.Backend, p.Runtime, p.Fallback)
	}
	return nil
}

func (p Qwen38VulkanDecodePacket) Digest() (string, error) {
	if err := p.Validate(); err != nil {
		return "", err
	}
	encoded, err := json.Marshal(p)
	if err != nil {
		return "", fmt.Errorf("marshal qwen3.8 Vulkan packet: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return strings.ToUpper(hex.EncodeToString(sum[:])), nil
}

type Qwen38VulkanTransferCounters struct {
	Count uint64 `json:"count"`
	Bytes uint64 `json:"bytes"`
}

type Qwen38VulkanTensorHomeCounters struct {
	Hits          uint64 `json:"hits"`
	Admissions    uint64 `json:"admissions"`
	Bypasses      uint64 `json:"bypasses"`
	ResidentBytes uint64 `json:"resident_bytes"`
	CopiedBytes   uint64 `json:"copied_bytes"`
}

type Qwen38VulkanDecodeCounters struct {
	ComputeDispatches      uint64                         `json:"compute_dispatches"`
	Q4KMatmulDispatches    uint64                         `json:"q4_k_matmul_dispatches"`
	OtherComputeDispatches uint64                         `json:"other_compute_dispatches"`
	DispatchSubmits        uint64                         `json:"dispatch_submits"`
	H2D                    Qwen38VulkanTransferCounters   `json:"h2d"`
	D2H                    Qwen38VulkanTransferCounters   `json:"d2h"`
	D2D                    Qwen38VulkanTransferCounters   `json:"d2d"`
	Q4KStageCalls          uint64                         `json:"q4_k_stage_calls"`
	Q4KStageBytes          uint64                         `json:"q4_k_stage_bytes"`
	TensorHome             Qwen38VulkanTensorHomeCounters `json:"tensor_home"`
}

// Qwen38VulkanSourceIdentity binds a physical result to the exact source and
// executable bytes that produced it. Dirty is a pointer so an observed clean
// tree is distinguishable from an unobserved tree state.
type Qwen38VulkanSourceIdentity struct {
	GitCommit           string `json:"git_commit"`
	SourceArchiveSHA256 string `json:"source_archive_sha256"`
	BinarySHA256        string `json:"binary_sha256"`
	Dirty               *bool  `json:"dirty"`
	DiffSHA256          string `json:"diff_sha256,omitempty"`
}

type Qwen38VulkanModelIdentity struct {
	Name                  string `json:"name"`
	ArtifactPath          string `json:"artifact_path"`
	ArtifactSHA256        string `json:"artifact_sha256"`
	TensorInventorySHA256 string `json:"tensor_inventory_sha256"`
	TokenizerSHA256       string `json:"tokenizer_sha256"`
	TemplateSHA256        string `json:"template_sha256"`
	Quantization          string `json:"quantization"`
}

type Qwen38VulkanDeviceIdentity struct {
	OS            string `json:"os"`
	Arch          string `json:"arch"`
	Kernel        string `json:"kernel"`
	Name          string `json:"name"`
	MesaVersion   string `json:"mesa_version"`
	VulkanVersion string `json:"vulkan_version"`
	Firmware      string `json:"firmware"`
}

// FallbackCount is a pointer because zero is creditable only when the runner
// actually observed it. A nil value means the current path did not measure it.
type Qwen38VulkanEngineIdentity struct {
	Name          string  `json:"name"`
	Backend       string  `json:"backend"`
	Runtime       string  `json:"runtime"`
	ExecutedPath  string  `json:"executed_path"`
	FallbackCount *uint64 `json:"fallback_count"`
}

type Qwen38VulkanDecodeRun struct {
	Repetition                  int     `json:"repetition"`
	ContextLimit                int     `json:"context_limit"`
	ContextTokens               int     `json:"context_tokens"`
	GeneratedTokenLimit         int     `json:"generated_token_limit"`
	ActualGeneratedTokens       int     `json:"actual_generated_tokens"`
	Sampler                     string  `json:"sampler"`
	SeedPolicy                  string  `json:"seed_policy"`
	IgnoreEOS                   *bool   `json:"ignore_eos"`
	EOSStopped                  *bool   `json:"eos_stopped"`
	OutputTokenIDs              []int32 `json:"output_token_ids"`
	SessionSetupNanoseconds     uint64  `json:"session_setup_nanoseconds"`
	PrefillNanoseconds          uint64  `json:"prefill_nanoseconds"`
	FirstSampleNanoseconds      uint64  `json:"first_sample_nanoseconds"`
	DecodeNanoseconds           uint64  `json:"decode_nanoseconds"`
	TeardownNanoseconds         uint64  `json:"teardown_nanoseconds"`
	CandidateElapsedNanoseconds uint64  `json:"candidate_elapsed_nanoseconds"`
	CPUVerificationNanoseconds  uint64  `json:"cpu_verification_nanoseconds"`
}

// Qwen38VulkanRawDecodeResult is the promotion boundary between a raw runner
// report and a canonical physical receipt. Callers must leave unobserved fields
// empty; BuildQwen38VulkanDecodeReceipt fails closed instead of supplying them.
type Qwen38VulkanRawDecodeResult struct {
	Source                      Qwen38VulkanSourceIdentity  `json:"source"`
	Model                       Qwen38VulkanModelIdentity   `json:"model"`
	Device                      Qwen38VulkanDeviceIdentity  `json:"device"`
	Engine                      Qwen38VulkanEngineIdentity  `json:"engine"`
	CaptureCommand              string                      `json:"capture_command"`
	PromptTokenIDs              []int32                     `json:"prompt_token_ids"`
	GeneratedTokenLimit         int                         `json:"generated_token_limit"`
	OutputTokenIDs              []int32                     `json:"output_token_ids"`
	OutputText                  string                      `json:"output_text"`
	FiniteLogits                *bool                       `json:"finite_logits"`
	CPUModelParity              *bool                       `json:"cpu_model_parity"`
	Runs                        []Qwen38VulkanDecodeRun     `json:"runs"`
	CandidateElapsedNanoseconds uint64                      `json:"candidate_elapsed_nanoseconds"`
	CPUVerificationNanoseconds  uint64                      `json:"cpu_verification_nanoseconds"`
	PeakProcessMemoryBytes      *uint64                     `json:"peak_process_memory_bytes,omitempty"`
	PeakDeviceMemoryBytes       *uint64                     `json:"peak_device_memory_bytes,omitempty"`
	Counters                    *Qwen38VulkanDecodeCounters `json:"counters,omitempty"`
}

// Qwen38VulkanDecodeReceipt captures comparable work and cost for one packet.
type Qwen38VulkanDecodeReceipt struct {
	Schema                      string                     `json:"schema"`
	Source                      Qwen38VulkanSourceIdentity `json:"source"`
	Model                       Qwen38VulkanModelIdentity  `json:"model"`
	Device                      Qwen38VulkanDeviceIdentity `json:"device"`
	Engine                      Qwen38VulkanEngineIdentity `json:"engine"`
	CaptureCommand              string                     `json:"capture_command"`
	Packet                      Qwen38VulkanDecodePacket   `json:"packet"`
	PacketSHA256                string                     `json:"packet_sha256"`
	Backend                     string                     `json:"backend"`
	Runtime                     string                     `json:"runtime"`
	Fallback                    string                     `json:"fallback"`
	OutputTokenIDs              []int32                    `json:"output_token_ids"`
	OutputTokenIDsSHA256        string                     `json:"output_token_ids_sha256"`
	OutputText                  string                     `json:"output_text"`
	GeneratedTokens             int                        `json:"generated_tokens"`
	FiniteLogits                *bool                      `json:"finite_logits"`
	CPUModelParity              *bool                      `json:"cpu_model_parity"`
	Runs                        []Qwen38VulkanDecodeRun    `json:"runs"`
	CandidateElapsedNanoseconds uint64                     `json:"candidate_elapsed_nanoseconds"`
	CPUVerificationNanoseconds  uint64                     `json:"cpu_verification_nanoseconds"`
	PeakProcessMemoryBytes      uint64                     `json:"peak_process_memory_bytes"`
	PeakDeviceMemoryBytes       uint64                     `json:"peak_device_memory_bytes"`
	Counters                    Qwen38VulkanDecodeCounters `json:"counters"`
}

// BuildQwen38VulkanDecodeReceipt promotes a raw result only after the complete
// physical identity and measurement contract validates. On failure it returns
// a zero receipt so partial raw reports cannot be mistaken for canonical ones.
func BuildQwen38VulkanDecodeReceipt(raw Qwen38VulkanRawDecodeResult) (Qwen38VulkanDecodeReceipt, error) {
	packet := NewQwen38VulkanDecodePacket(raw.PromptTokenIDs, raw.GeneratedTokenLimit)
	packetDigest, err := packet.Digest()
	if err != nil {
		return Qwen38VulkanDecodeReceipt{}, fmt.Errorf("raw decode result is not a canonical physical receipt: %w", err)
	}
	var peakProcessMemoryBytes, peakDeviceMemoryBytes uint64
	if raw.PeakProcessMemoryBytes != nil {
		peakProcessMemoryBytes = *raw.PeakProcessMemoryBytes
	}
	if raw.PeakDeviceMemoryBytes != nil {
		peakDeviceMemoryBytes = *raw.PeakDeviceMemoryBytes
	}
	var counters Qwen38VulkanDecodeCounters
	if raw.Counters != nil {
		counters = *raw.Counters
	}
	source := raw.Source
	if raw.Source.Dirty != nil {
		dirty := *raw.Source.Dirty
		source.Dirty = &dirty
	}
	engine := raw.Engine
	if raw.Engine.FallbackCount != nil {
		fallbackCount := *raw.Engine.FallbackCount
		engine.FallbackCount = &fallbackCount
	}
	runs := make([]Qwen38VulkanDecodeRun, len(raw.Runs))
	for i, run := range raw.Runs {
		runs[i] = run
		runs[i].OutputTokenIDs = slices.Clone(run.OutputTokenIDs)
		runs[i].IgnoreEOS = qwen38VulkanBoolCopy(run.IgnoreEOS)
		runs[i].EOSStopped = qwen38VulkanBoolCopy(run.EOSStopped)
	}
	receipt := Qwen38VulkanDecodeReceipt{
		Schema:                      Qwen38VulkanDecodeReceiptSchema,
		Source:                      source,
		Model:                       raw.Model,
		Device:                      raw.Device,
		Engine:                      engine,
		CaptureCommand:              raw.CaptureCommand,
		Packet:                      packet,
		PacketSHA256:                packetDigest,
		Backend:                     raw.Engine.Backend,
		Runtime:                     raw.Engine.Runtime,
		Fallback:                    Qwen38VulkanDecodeFallback,
		OutputTokenIDs:              slices.Clone(raw.OutputTokenIDs),
		OutputTokenIDsSHA256:        Qwen38VulkanTokenIDsSHA256(raw.OutputTokenIDs),
		OutputText:                  raw.OutputText,
		GeneratedTokens:             len(raw.OutputTokenIDs),
		FiniteLogits:                qwen38VulkanBoolCopy(raw.FiniteLogits),
		CPUModelParity:              qwen38VulkanBoolCopy(raw.CPUModelParity),
		Runs:                        runs,
		CandidateElapsedNanoseconds: raw.CandidateElapsedNanoseconds,
		CPUVerificationNanoseconds:  raw.CPUVerificationNanoseconds,
		PeakProcessMemoryBytes:      peakProcessMemoryBytes,
		PeakDeviceMemoryBytes:       peakDeviceMemoryBytes,
		Counters:                    counters,
	}
	if err := receipt.Validate(); err != nil {
		return Qwen38VulkanDecodeReceipt{}, fmt.Errorf("raw decode result is not a canonical physical receipt: %w", err)
	}
	return receipt, nil
}

func qwen38VulkanBoolCopy(value *bool) *bool {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// Qwen38VulkanTokenIDsSHA256 hashes signed token IDs as a count followed by
// big-endian 32-bit values, avoiding architecture- and JSON-dependent hashes.
func Qwen38VulkanTokenIDsSHA256(tokenIDs []int32) string {
	buf := make([]byte, 8+4*len(tokenIDs))
	binary.BigEndian.PutUint64(buf[:8], uint64(len(tokenIDs)))
	for i, tokenID := range tokenIDs {
		binary.BigEndian.PutUint32(buf[8+4*i:], uint32(tokenID))
	}
	sum := sha256.Sum256(buf)
	return strings.ToUpper(hex.EncodeToString(sum[:]))
}

func (r Qwen38VulkanDecodeReceipt) Validate() error {
	if r.Schema != Qwen38VulkanDecodeReceiptSchema {
		return fmt.Errorf("qwen3.8 Vulkan receipt schema %q, want %q", r.Schema, Qwen38VulkanDecodeReceiptSchema)
	}
	if err := r.Packet.Validate(); err != nil {
		return fmt.Errorf("qwen3.8 Vulkan receipt packet: %w", err)
	}
	if err := r.validatePhysicalIdentity(); err != nil {
		return err
	}
	packetDigest, err := r.Packet.Digest()
	if err != nil {
		return err
	}
	if r.PacketSHA256 != packetDigest {
		return fmt.Errorf("qwen3.8 Vulkan packet digest %q, want %q", r.PacketSHA256, packetDigest)
	}
	if r.Backend != r.Packet.Backend || r.Runtime != r.Packet.Runtime || r.Fallback != r.Packet.Fallback {
		return fmt.Errorf("qwen3.8 Vulkan receipt engine identity does not match packet")
	}
	if r.GeneratedTokens != r.Packet.GeneratedTokenLimit || r.GeneratedTokens != len(r.OutputTokenIDs) {
		return fmt.Errorf("qwen3.8 Vulkan work boundary mismatch: limit=%d generated=%d output_ids=%d", r.Packet.GeneratedTokenLimit, r.GeneratedTokens, len(r.OutputTokenIDs))
	}
	if r.OutputTokenIDsSHA256 != Qwen38VulkanTokenIDsSHA256(r.OutputTokenIDs) {
		return fmt.Errorf("qwen3.8 Vulkan output token digest %q does not match token IDs", r.OutputTokenIDsSHA256)
	}
	if r.OutputText == "" {
		return errors.New("qwen3.8 Vulkan receipt requires output text identity")
	}
	if r.FiniteLogits == nil || !*r.FiniteLogits {
		return errors.New("qwen3.8 Vulkan receipt requires observed finite_logits=true")
	}
	if r.CPUModelParity == nil || !*r.CPUModelParity {
		return errors.New("qwen3.8 Vulkan receipt requires observed cpu_model_parity=true")
	}
	if err := r.validateRuns(); err != nil {
		return err
	}
	if r.PeakProcessMemoryBytes == 0 || r.PeakDeviceMemoryBytes == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires positive process and device peak memory")
	}
	if r.PeakDeviceMemoryBytes > r.PeakProcessMemoryBytes {
		return fmt.Errorf("qwen3.8 Vulkan device peak %d exceeds process peak %d", r.PeakDeviceMemoryBytes, r.PeakProcessMemoryBytes)
	}
	return r.Counters.validate()
}

func (r Qwen38VulkanDecodeReceipt) validatePhysicalIdentity() error {
	if !qwen38VulkanGitCommitRE.MatchString(strings.TrimSpace(r.Source.GitCommit)) {
		return errors.New("qwen3.8 Vulkan receipt requires full source commit identity")
	}
	for _, identity := range []struct{ name, value string }{
		{"source archive", r.Source.SourceArchiveSHA256},
		{"binary", r.Source.BinarySHA256},
	} {
		if !qwen38VulkanSHA256(identity.value) {
			return fmt.Errorf("qwen3.8 Vulkan receipt requires %s SHA-256 identity", identity.name)
		}
	}
	if r.Source.Dirty == nil {
		return errors.New("qwen3.8 Vulkan receipt requires observed source clean/dirty state")
	}
	if *r.Source.Dirty {
		if !qwen38VulkanSHA256(r.Source.DiffSHA256) {
			return errors.New("qwen3.8 Vulkan dirty source requires diff SHA-256 identity")
		}
	} else if strings.TrimSpace(r.Source.DiffSHA256) != "" {
		return errors.New("qwen3.8 Vulkan clean source cannot include a diff SHA-256")
	}
	if strings.TrimSpace(r.Model.Name) == "" {
		return errors.New("qwen3.8 Vulkan receipt requires model identity")
	}
	if strings.TrimSpace(r.Model.ArtifactPath) == "" || !qwen38VulkanSHA256(r.Model.ArtifactSHA256) {
		return errors.New("qwen3.8 Vulkan receipt requires artifact path and SHA-256 identity")
	}
	if !strings.EqualFold(qwen38VulkanBareSHA256(r.Model.ArtifactSHA256), r.Packet.ModelGGUFSHA256) {
		return errors.New("qwen3.8 Vulkan artifact SHA-256 does not match packet model identity")
	}
	for _, identity := range []struct{ name, value string }{
		{"tensor inventory", r.Model.TensorInventorySHA256},
		{"tokenizer", r.Model.TokenizerSHA256},
		{"template", r.Model.TemplateSHA256},
	} {
		if !qwen38VulkanSHA256(identity.value) {
			return fmt.Errorf("qwen3.8 Vulkan receipt requires %s SHA-256 identity", identity.name)
		}
	}
	if strings.TrimSpace(r.Model.Quantization) == "" {
		return errors.New("qwen3.8 Vulkan receipt requires quantization identity")
	}
	for _, identity := range []struct{ name, value string }{
		{"OS", r.Device.OS},
		{"architecture", r.Device.Arch},
		{"kernel", r.Device.Kernel},
		{"device", r.Device.Name},
		{"Mesa", r.Device.MesaVersion},
		{"Vulkan", r.Device.VulkanVersion},
		{"firmware", r.Device.Firmware},
		{"executed path", r.Engine.ExecutedPath},
		{"capture command", r.CaptureCommand},
	} {
		if strings.TrimSpace(identity.value) == "" {
			return fmt.Errorf("qwen3.8 Vulkan receipt requires %s identity", identity.name)
		}
	}
	if r.Engine.Name != "fak-native" || r.Engine.Backend != Qwen38VulkanDecodeBackend || r.Engine.Runtime != Qwen38VulkanDecodeRuntime {
		return fmt.Errorf("qwen3.8 Vulkan receipt requires engine=fak-native backend=%s runtime=%s", Qwen38VulkanDecodeBackend, Qwen38VulkanDecodeRuntime)
	}
	if r.Engine.FallbackCount == nil || *r.Engine.FallbackCount != 0 {
		return errors.New("qwen3.8 Vulkan receipt requires observed fallback_count=0")
	}
	return nil
}

func (r Qwen38VulkanDecodeReceipt) validateRuns() error {
	if len(r.Runs) == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires per-repetition timing evidence")
	}
	for i, run := range r.Runs {
		if run.Repetition != i+1 {
			return fmt.Errorf("qwen3.8 Vulkan repetition index %d, want %d", run.Repetition, i+1)
		}
		if run.ContextLimit <= 0 || run.ContextTokens != len(r.Packet.PromptTokenIDs)+run.ActualGeneratedTokens || run.ContextTokens > run.ContextLimit {
			return fmt.Errorf("qwen3.8 Vulkan repetition %d has invalid context boundary", run.Repetition)
		}
		if run.GeneratedTokenLimit != r.Packet.GeneratedTokenLimit || run.ActualGeneratedTokens != run.GeneratedTokenLimit || run.ActualGeneratedTokens != len(run.OutputTokenIDs) {
			return fmt.Errorf("qwen3.8 Vulkan repetition %d has invalid generated-token boundary", run.Repetition)
		}
		if !slices.Equal(run.OutputTokenIDs, r.OutputTokenIDs) {
			return fmt.Errorf("qwen3.8 Vulkan repetition %d output token identity mismatch", run.Repetition)
		}
		if run.Sampler != "greedy" || run.SeedPolicy != "not_applicable_greedy" {
			return fmt.Errorf("qwen3.8 Vulkan repetition %d requires sampler=greedy seed_policy=not_applicable_greedy", run.Repetition)
		}
		if run.IgnoreEOS == nil || run.EOSStopped == nil {
			return fmt.Errorf("qwen3.8 Vulkan repetition %d requires observed EOS policy and outcome", run.Repetition)
		}
		candidate, ok := qwen38VulkanDurationSum(run.SessionSetupNanoseconds, run.PrefillNanoseconds, run.FirstSampleNanoseconds, run.DecodeNanoseconds, run.TeardownNanoseconds)
		if !ok || candidate == 0 || run.CandidateElapsedNanoseconds != candidate {
			return fmt.Errorf("qwen3.8 Vulkan repetition %d has invalid candidate timing", run.Repetition)
		}
	}
	if r.CandidateElapsedNanoseconds != r.Runs[0].CandidateElapsedNanoseconds {
		return errors.New("qwen3.8 Vulkan receipt candidate elapsed time does not match repetition zero")
	}
	if r.CPUVerificationNanoseconds != r.Runs[0].CPUVerificationNanoseconds {
		return errors.New("qwen3.8 Vulkan receipt CPU verification time does not match repetition zero")
	}
	return nil
}

func qwen38VulkanDurationSum(values ...uint64) (uint64, bool) {
	var total uint64
	for _, value := range values {
		if ^uint64(0)-total < value {
			return 0, false
		}
		total += value
	}
	return total, true
}

func qwen38VulkanBareSHA256(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > len("sha256:") && strings.EqualFold(value[:len("sha256:")], "sha256:") {
		return value[len("sha256:"):]
	}
	return value
}

func qwen38VulkanSHA256(value string) bool {
	bare := qwen38VulkanBareSHA256(value)
	if len(bare) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(bare)
	return err == nil
}

func (c Qwen38VulkanDecodeCounters) validate() error {
	if c.ComputeDispatches == 0 || c.DispatchSubmits == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires compute dispatches and dispatch submits")
	}
	if c.ComputeDispatches != c.Q4KMatmulDispatches+c.OtherComputeDispatches {
		return fmt.Errorf("qwen3.8 Vulkan compute dispatch total %d != q4_k %d + other %d", c.ComputeDispatches, c.Q4KMatmulDispatches, c.OtherComputeDispatches)
	}
	for name, transfer := range map[string]Qwen38VulkanTransferCounters{"h2d": c.H2D, "d2h": c.D2H, "d2d": c.D2D} {
		if (transfer.Count == 0) != (transfer.Bytes == 0) {
			return fmt.Errorf("qwen3.8 Vulkan %s count/bytes presence mismatch", name)
		}
	}
	if (c.Q4KStageCalls == 0) != (c.Q4KStageBytes == 0) {
		return errors.New("qwen3.8 Vulkan q4_k stage calls/bytes presence mismatch")
	}
	if c.TensorHome.Admissions > c.D2D.Count || c.TensorHome.CopiedBytes > c.D2D.Bytes {
		return errors.New("qwen3.8 Vulkan tensor-home admissions/copies exceed d2d work")
	}
	if c.TensorHome.ResidentBytes > c.TensorHome.CopiedBytes {
		return errors.New("qwen3.8 Vulkan tensor-home resident bytes exceed copied bytes")
	}
	if c.TensorHome.Hits+c.TensorHome.Admissions+c.TensorHome.Bypasses == 0 {
		return errors.New("qwen3.8 Vulkan receipt requires tensor-home accounting")
	}
	return nil
}

// CompareQwen38VulkanDecodeReceipts rejects unequal request, output, or token
// work boundaries while intentionally allowing performance counters to differ.
func CompareQwen38VulkanDecodeReceipts(parent, candidate Qwen38VulkanDecodeReceipt) error {
	if err := parent.Validate(); err != nil {
		return fmt.Errorf("parent receipt: %w", err)
	}
	if err := candidate.Validate(); err != nil {
		return fmt.Errorf("candidate receipt: %w", err)
	}
	if parent.PacketSHA256 != candidate.PacketSHA256 || !equalQwen38VulkanPackets(parent.Packet, candidate.Packet) {
		return errors.New("parent/candidate qwen3.8 Vulkan packet mismatch")
	}
	if parent.GeneratedTokens != candidate.GeneratedTokens {
		return errors.New("parent/candidate qwen3.8 Vulkan generated-token boundary mismatch")
	}
	if err := compareQwen38VulkanRunRequests(parent.Runs, candidate.Runs); err != nil {
		return err
	}
	if parent.OutputTokenIDsSHA256 != candidate.OutputTokenIDsSHA256 || !slices.Equal(parent.OutputTokenIDs, candidate.OutputTokenIDs) {
		return errors.New("parent/candidate qwen3.8 Vulkan output mismatch")
	}
	if parent.OutputText != candidate.OutputText {
		return errors.New("parent/candidate qwen3.8 Vulkan output text mismatch")
	}
	return nil
}

func compareQwen38VulkanRunRequests(parent, candidate []Qwen38VulkanDecodeRun) error {
	if len(parent) != len(candidate) {
		return fmt.Errorf("parent/candidate qwen3.8 Vulkan repetition count mismatch: %d != %d", len(parent), len(candidate))
	}
	for i := range parent {
		p, c := parent[i], candidate[i]
		if p.Repetition != c.Repetition || p.ContextTokens != c.ContextTokens || p.GeneratedTokenLimit != c.GeneratedTokenLimit || p.ActualGeneratedTokens != c.ActualGeneratedTokens {
			return fmt.Errorf("parent/candidate qwen3.8 Vulkan repetition %d request shape mismatch", i+1)
		}
		if p.ContextLimit != c.ContextLimit {
			return fmt.Errorf("parent/candidate qwen3.8 Vulkan repetition %d context limit mismatch", i+1)
		}
		if p.Sampler != c.Sampler || p.SeedPolicy != c.SeedPolicy {
			return fmt.Errorf("parent/candidate qwen3.8 Vulkan repetition %d sampler/seed policy mismatch", i+1)
		}
		if *p.IgnoreEOS != *c.IgnoreEOS {
			return fmt.Errorf("parent/candidate qwen3.8 Vulkan repetition %d EOS policy mismatch", i+1)
		}
		if *p.EOSStopped != *c.EOSStopped {
			return fmt.Errorf("parent/candidate qwen3.8 Vulkan repetition %d EOS outcome mismatch", i+1)
		}
	}
	return nil
}

func equalQwen38VulkanPackets(a, b Qwen38VulkanDecodePacket) bool {
	return a.Schema == b.Schema &&
		a.ModelGGUFSHA256 == b.ModelGGUFSHA256 &&
		slices.Equal(a.PromptTokenIDs, b.PromptTokenIDs) &&
		a.GeneratedTokenLimit == b.GeneratedTokenLimit &&
		a.Backend == b.Backend && a.Runtime == b.Runtime && a.Fallback == b.Fallback
}

// ExecuteQwen38VulkanDecodeContract keeps lifecycle cleanup independent of the
// live runtime. The deferred call runs exactly once on return or panic.
func ExecuteQwen38VulkanDecodeContract[T any](cleanup func(), run func() (T, error)) (T, error) {
	if cleanup == nil {
		var zero T
		return zero, errors.New("qwen3.8 Vulkan decode cleanup is required")
	}
	defer cleanup()
	if run == nil {
		var zero T
		return zero, errors.New("qwen3.8 Vulkan decode runner is required")
	}
	return run()
}
