package qwen38quantrun

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
)

const (
	StrixComparisonCellManifestSchema = "fak.qwen38.strix-comparison-cell.v1"
	strixComparisonCellDigestDomain   = "fak.qwen38.strix-comparison-cell.digest.v1"

	StrixComparisonSourceRowID                 = "nabe-qwen38-27b-vulkan-mtp-c1-c4-c8"
	StrixComparisonSourceDate                  = "2026-08-30"
	StrixComparisonSourceRevision              = "d2049a8892577a2e8caef51d71b5f74dfbd7abad"
	StrixComparisonArtifactSHA256              = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	StrixComparisonTokenizerSHA256             = "839c662c4a47759df9150bd939b382767b1e36bc2372740ad6d02495a63fa5a0"
	StrixComparisonTemplateSHA256              = "87049d017c4eee304541572ddfee78756784389fac4808d02039215f062d31d1"
	StrixComparisonRenderedPromptSHA256        = "ecb27635e62e758f65a349c34ef2f873d9f43e1e78357d3ab72691eb255f05ef"
	StrixComparisonLlamaSourceRevision         = "70adb1b4cea5ee39f867792c78dc59320921eda7"
	StrixComparisonLlamaTreeSHA256             = "174f4186cb52cad7783040246a7f40b7a7cd5329"
	StrixComparisonContextTokens               = 32768
	StrixComparisonOutputTokens                = 128
	StrixComparisonWarmups                     = 3
	StrixComparisonMinimumMeasuredPairs        = 5
	StrixComparisonUMAClassBytes        uint64 = 64 << 30
)

var (
	strixComparisonPlatformObservationNames = []string{
		"physical_ram", "pci_device", "boot_session", "kernel", "mesa", "radv",
		"vulkan_loader", "vulkan_icd", "power_policy", "clock_policy", "thermal_policy",
		"throttle_policy", "gpu_lease",
	}
	strixComparisonCaptureBindings = []string{
		"trial_native_timing_reconciliation", "accepted_token_ids", "accepted_token_logprobs",
		"eos_observation", "resource_observation",
	}
)

// StrixComparisonCellManifest is a preregistered integrity envelope. Its digest
// proves byte identity only; it cannot prove execution, hardware, or performance.
type StrixComparisonCellManifest struct {
	Schema    string                        `json:"schema"`
	Challenge StrixComparisonChallenge      `json:"challenge"`
	Platform  StrixComparisonPlatform       `json:"platform"`
	Workload  StrixComparisonWorkload       `json:"workload"`
	Memory    StrixComparisonMemoryEnvelope `json:"memory"`
	Candidate StrixComparisonCandidatePin   `json:"candidate"`
	Reference StrixComparisonReferencePin   `json:"reference"`
	Capture   StrixComparisonCapturePlan    `json:"capture"`
	Digest    string                        `json:"digest,omitempty"`
}

type StrixComparisonChallenge struct {
	SourceRowID        string   `json:"source_row_id"`
	SourceDate         string   `json:"source_date"`
	SourceRevision     string   `json:"source_revision"`
	Concurrency        int      `json:"concurrency"`
	AcceptedOutputTPS  float64  `json:"accepted_output_tps_bar"`
	ConfidenceRule     string   `json:"confidence_rule"`
	PairedRatioRule    string   `json:"paired_ratio_rule"`
	SamplingAssumption string   `json:"sampling_assumption"`
	Warmups            int      `json:"warmups"`
	MeasuredPairs      int      `json:"measured_pairs"`
	AlternatingOrder   []string `json:"alternating_order"`
	PairIDs            []string `json:"pair_ids"`
}

type StrixComparisonObservation struct {
	Name        string `json:"name"`
	Value       string `json:"value"`
	ValueSHA256 string `json:"value_sha256"`
}

type StrixComparisonPlatform struct {
	ApplianceID      string                       `json:"appliance_id"`
	CPU              string                       `json:"cpu"`
	GPU              string                       `json:"gpu"`
	GPUArchitecture  string                       `json:"gpu_architecture"`
	ComputeUnits     int                          `json:"compute_units"`
	UMAClassBytes    uint64                       `json:"uma_class_bytes"`
	PhysicalRAMBytes uint64                       `json:"physical_ram_bytes"`
	LeaseIdentity    string                       `json:"lease_identity"`
	Observations     []StrixComparisonObservation `json:"observations"`
}

type StrixComparisonWorkload struct {
	Model                string `json:"model"`
	Quantization         string `json:"quantization"`
	ArtifactSHA256       string `json:"artifact_sha256"`
	PromptPacketBytes    []byte `json:"prompt_packet_bytes"`
	PromptPacketDigest   string `json:"prompt_packet_digest"`
	TokenizerSHA256      string `json:"tokenizer_sha256"`
	TemplateSHA256       string `json:"template_sha256"`
	RenderedPromptSHA256 string `json:"rendered_prompt_sha256"`
	PromptTokenIDs       []int  `json:"prompt_token_ids"`
	ContextTokens        int    `json:"context_tokens"`
	AcceptedOutputTokens int    `json:"accepted_output_tokens"`
	ReasoningEnabled     bool   `json:"reasoning_enabled"`
	SpeculationEnabled   bool   `json:"speculation_enabled"`
}

type StrixComparisonMemoryEnvelope struct {
	ContextBudgetBytes     uint64 `json:"context_budget_bytes"`
	KVTypeK                string `json:"kv_type_k"`
	KVTypeV                string `json:"kv_type_v"`
	KVOffload              string `json:"kv_offload"`
	FlashAttention         bool   `json:"flash_attention"`
	GPUUMABudgetBytes      uint64 `json:"gpu_uma_budget_bytes"`
	CandidateBudgetBytes   uint64 `json:"candidate_budget_bytes"`
	ReferenceBudgetBytes   uint64 `json:"reference_budget_bytes"`
	HostSpillPolicy        string `json:"host_spill_policy"`
	PrimaryCacheState      string `json:"primary_cache_state"`
	MemoryAccountingPolicy string `json:"memory_accounting_policy"`
	ResidentModelBytes     uint64 `json:"resident_model_bytes"`
	CandidatePeakMethod    string `json:"candidate_peak_method"`
	ReferencePeakMethod    string `json:"reference_peak_method"`
}

type StrixComparisonCandidatePin struct {
	CampaignClass       string `json:"campaign_class"`
	Runtime             string `json:"runtime"`
	Owner               string `json:"owner"`
	Planner             string `json:"planner"`
	Backend             string `json:"backend"`
	Fallback            bool   `json:"fallback"`
	FallbackCount       int    `json:"fallback_count"`
	SourceRevision      string `json:"source_revision"`
	SourceArchiveSHA256 string `json:"source_archive_sha256"`
	BuildManifestSHA256 string `json:"build_manifest_sha256"`
	ToolchainSHA256     string `json:"toolchain_sha256"`
	ExecutableSHA256    string `json:"executable_sha256"`
	ShaderBundleSHA256  string `json:"shader_bundle_sha256"`
	ModelSHA256         string `json:"model_sha256"`
	ForwardPath         string `json:"forward_path"`
}

type StrixComparisonReferencePin struct {
	CampaignClass       string `json:"campaign_class"`
	SourceRevision      string `json:"source_revision"`
	SourceTreeSHA256    string `json:"source_tree_sha256"`
	BuildType           string `json:"build_type"`
	GGMLVulkan          bool   `json:"ggml_vulkan"`
	SourceArchiveSHA256 string `json:"source_archive_sha256"`
	BuildManifestSHA256 string `json:"build_manifest_sha256"`
	ToolchainSHA256     string `json:"toolchain_sha256"`
	ServerBinarySHA256  string `json:"server_binary_sha256"`
	BenchBinarySHA256   string `json:"bench_binary_sha256"`
	LoaderSHA256        string `json:"loader_sha256"`
	DependencySHA256    string `json:"dependency_sha256"`
	RADVDeviceIdentity  string `json:"radv_device_identity"`
	ModelSHA256         string `json:"model_sha256"`
	PromptPacketDigest  string `json:"prompt_packet_digest"`
}

type StrixComparisonCapturePlan struct {
	CellNonce           string   `json:"cell_nonce"`
	AuthoritySchema     string   `json:"authority_schema"`
	ArmRoles            []string `json:"arm_roles"`
	PairIDs             []string `json:"pair_ids"`
	AlternatingOrder    []string `json:"alternating_order"`
	TrialCount          int      `json:"trial_count"`
	MonotonicClockID    string   `json:"monotonic_clock_id"`
	SessionIdentity     string   `json:"session_identity"`
	ReplayKey           string   `json:"replay_key"`
	ObservationBindings []string `json:"observation_bindings"`
}

// SealStrixComparisonCellManifest validates and seals a pre-warmup manifest.
func SealStrixComparisonCellManifest(m StrixComparisonCellManifest) (StrixComparisonCellManifest, error) {
	sealed, err := cloneStrixComparisonCellManifest(m)
	if err != nil {
		return StrixComparisonCellManifest{}, err
	}
	sealed.Digest = ""
	if err := validateStrixComparisonCellManifest(sealed); err != nil {
		return StrixComparisonCellManifest{}, err
	}
	digest, err := computeStrixComparisonCellManifestDigest(sealed)
	if err != nil {
		return StrixComparisonCellManifest{}, err
	}
	sealed.Digest = digest
	return sealed, nil
}

// VerifyStrixComparisonCellManifest verifies integrity and the exact digest
// independently approved before warmup. A newly recomputed digest is not approval.
func VerifyStrixComparisonCellManifest(m StrixComparisonCellManifest, approvedDigest string) error {
	if !nonEmptyManifestSHA256(approvedDigest) {
		return errors.New("strix comparison cell approved digest is required")
	}
	if err := validateStrixComparisonCellManifest(m); err != nil {
		return err
	}
	want, err := computeStrixComparisonCellManifestDigest(m)
	if err != nil {
		return err
	}
	if m.Digest != want {
		return fmt.Errorf("strix comparison cell digest mismatch: got %q want %q", m.Digest, want)
	}
	if m.Digest != approvedDigest {
		return fmt.Errorf("strix comparison cell identity %q is not the approved challenge %q", m.Digest, approvedDigest)
	}
	return nil
}

// ImportStrixComparisonCellManifest rejects duplicate and unknown JSON fields.
func ImportStrixComparisonCellManifest(raw []byte, approvedDigest string) (StrixComparisonCellManifest, error) {
	if err := rejectDuplicateJSONFields(raw); err != nil {
		return StrixComparisonCellManifest{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var m StrixComparisonCellManifest
	if err := dec.Decode(&m); err != nil {
		return StrixComparisonCellManifest{}, fmt.Errorf("decode strix comparison cell: %w", err)
	}
	if err := ensureJSONEOF(dec); err != nil {
		return StrixComparisonCellManifest{}, err
	}
	if err := VerifyStrixComparisonCellManifest(m, approvedDigest); err != nil {
		return StrixComparisonCellManifest{}, err
	}
	return m, nil
}

func ExportStrixComparisonCellManifest(m StrixComparisonCellManifest, approvedDigest string) ([]byte, error) {
	if err := VerifyStrixComparisonCellManifest(m, approvedDigest); err != nil {
		return nil, err
	}
	return json.MarshalIndent(m, "", "  ")
}

func computeStrixComparisonCellManifestDigest(m StrixComparisonCellManifest) (string, error) {
	m.Digest = ""
	payload, err := json.Marshal(m)
	if err != nil {
		return "", fmt.Errorf("encode strix comparison cell: %w", err)
	}
	var canonical bytes.Buffer
	writeManifestBytes(&canonical, []byte(strixComparisonCellDigestDomain))
	writeManifestBytes(&canonical, payload)
	sum := sha256.Sum256(canonical.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

func cloneStrixComparisonCellManifest(m StrixComparisonCellManifest) (StrixComparisonCellManifest, error) {
	raw, err := json.Marshal(m)
	if err != nil {
		return StrixComparisonCellManifest{}, fmt.Errorf("clone strix comparison cell: %w", err)
	}
	var clone StrixComparisonCellManifest
	if err := json.Unmarshal(raw, &clone); err != nil {
		return StrixComparisonCellManifest{}, fmt.Errorf("clone strix comparison cell: %w", err)
	}
	return clone, nil
}

func writeManifestBytes(dst *bytes.Buffer, value []byte) {
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(value)))
	dst.Write(n[:])
	dst.Write(value)
}

func validateStrixComparisonCellManifest(m StrixComparisonCellManifest) error {
	if m.Schema != StrixComparisonCellManifestSchema {
		return fmt.Errorf("unsupported strix comparison cell schema %q", m.Schema)
	}
	if err := validateStrixChallenge(m.Challenge); err != nil {
		return err
	}
	if err := validateStrixPlatform(m.Platform); err != nil {
		return err
	}
	packet, err := validateStrixWorkload(m.Workload)
	if err != nil {
		return err
	}
	if err := validateStrixMemory(m.Memory); err != nil {
		return err
	}
	if packet.ContextBudget.ContextBudgetBytes != m.Memory.ContextBudgetBytes || m.Platform.UMAClassBytes > m.Platform.PhysicalRAMBytes || m.Memory.GPUUMABudgetBytes > m.Platform.UMAClassBytes {
		return errors.New("strix comparison cell packet/platform/memory budgets are contradictory")
	}
	if err := validateCandidatePin(m.Candidate, m.Workload); err != nil {
		return err
	}
	if err := validateReferencePin(m.Reference, m.Workload); err != nil {
		return err
	}
	return validateCapturePlan(m.Capture, m.Challenge)
}

func validateStrixChallenge(c StrixComparisonChallenge) error {
	if c.SourceRowID != StrixComparisonSourceRowID || c.SourceDate != StrixComparisonSourceDate || c.SourceRevision != StrixComparisonSourceRevision {
		return errors.New("strix comparison cell source row/date is not canonical")
	}
	bars := map[int]float64{1: 16.65, 4: 24.13, 8: 29.42}
	bar, ok := bars[c.Concurrency]
	if !ok || c.AcceptedOutputTPS != bar || math.IsNaN(c.AcceptedOutputTPS) || math.IsInf(c.AcceptedOutputTPS, 0) {
		return errors.New("strix comparison cell concurrency/bar pair is not canonical")
	}
	if c.ConfidenceRule != "one-sided-95-percent" || c.PairedRatioRule != "paired-ratio-lcb95>1" || c.SamplingAssumption != "iid-approximately-normal-paired-ratios" {
		return errors.New("strix comparison cell statistical contract is not canonical")
	}
	if c.Warmups != StrixComparisonWarmups || c.MeasuredPairs != StrixComparisonMinimumMeasuredPairs {
		return errors.New("strix comparison cell requires 3 warmups and exactly 5 measured pairs")
	}
	if len(c.PairIDs) != c.MeasuredPairs || len(c.AlternatingOrder) != 2*c.MeasuredPairs {
		return errors.New("strix comparison cell pair/order cardinality mismatch")
	}
	if err := uniqueRequiredStrings("pair_ids", c.PairIDs); err != nil {
		return err
	}
	for i := range c.MeasuredPairs {
		pairID, first, second, _, _ := strixComparisonPairOrder(i)
		if c.PairIDs[i] != pairID || c.AlternatingOrder[2*i] != first || c.AlternatingOrder[2*i+1] != second {
			return errors.New("strix comparison cell measured order is not the canonical candidate/reference pair alternation")
		}
	}
	return nil
}

func strixComparisonPairOrder(pairIndex int) (pairID, first, second string, candidateSequence, referenceSequence int) {
	pairID = fmt.Sprintf("pair-%02d", pairIndex+1)
	candidateSequence, referenceSequence = 2*pairIndex+1, 2*pairIndex+2
	first, second = "candidate:"+pairID, "reference:"+pairID
	if pairIndex%2 != 0 {
		candidateSequence, referenceSequence = referenceSequence, candidateSequence
		first, second = second, first
	}
	return
}

func validateStrixPlatform(p StrixComparisonPlatform) error {
	if p.ApplianceID != "strix1" || p.CPU != "AMD Ryzen AI MAX+ 395" || p.GPU != "Radeon 8060S" || p.GPUArchitecture != "gfx1151" || p.ComputeUnits != 40 || p.UMAClassBytes != StrixComparisonUMAClassBytes {
		return errors.New("strix comparison cell platform constants are not canonical")
	}
	if p.PhysicalRAMBytes == 0 || invalidManifestString(p.LeaseIdentity) {
		return errors.New("strix comparison cell observed RAM and canonical lease identity are required")
	}
	if len(p.Observations) != len(strixComparisonPlatformObservationNames) {
		return errors.New("strix comparison cell platform observation set is incomplete")
	}
	for i, want := range strixComparisonPlatformObservationNames {
		got := p.Observations[i]
		valueSum := sha256.Sum256([]byte(got.Value))
		if got.Name != want || invalidManifestString(got.Value) || got.ValueSHA256 != hex.EncodeToString(valueSum[:]) {
			return fmt.Errorf("strix comparison cell platform observation %q is invalid", want)
		}
	}
	return nil
}

func validateStrixWorkload(w StrixComparisonWorkload) (PromptTokenPacket, error) {
	if w.Model != "Qwen3.8-27B" || w.Quantization != "Q4_K_M" || w.ArtifactSHA256 != StrixComparisonArtifactSHA256 || w.TokenizerSHA256 != StrixComparisonTokenizerSHA256 || w.TemplateSHA256 != StrixComparisonTemplateSHA256 || w.RenderedPromptSHA256 != StrixComparisonRenderedPromptSHA256 {
		return PromptTokenPacket{}, errors.New("strix comparison cell workload pins are not canonical")
	}
	if w.ContextTokens != StrixComparisonContextTokens || w.AcceptedOutputTokens != StrixComparisonOutputTokens || w.ReasoningEnabled || w.SpeculationEnabled {
		return PromptTokenPacket{}, errors.New("strix comparison cell workload controls are not canonical")
	}
	if len(w.PromptPacketBytes) == 0 || !nonEmptyManifestSHA256(w.PromptPacketDigest) {
		return PromptTokenPacket{}, errors.New("strix comparison cell exact prompt packet bytes/digest are required")
	}
	if err := rejectDuplicateJSONFields(w.PromptPacketBytes); err != nil {
		return PromptTokenPacket{}, fmt.Errorf("strix comparison cell prompt packet: %w", err)
	}
	packet, err := ImportPromptPacket(w.PromptPacketBytes)
	if err != nil {
		return PromptTokenPacket{}, fmt.Errorf("strix comparison cell prompt packet: %w", err)
	}
	if packet.PacketDigest != w.PromptPacketDigest || packet.ArtifactSHA256 != w.ArtifactSHA256 || packet.TokenizerDigest != w.TokenizerSHA256 || packet.TemplateDigest != w.TemplateSHA256 {
		return PromptTokenPacket{}, errors.New("strix comparison cell prompt packet identity mismatch")
	}
	if packet.TokenizerIdentity != GGUFTokenizerIdentity || len(w.PromptTokenIDs) != 26 || !slices.Equal(packet.PromptTokenIDs, w.PromptTokenIDs) || packet.ContextBudget.ContextTokens != w.ContextTokens || packet.ContextBudget.ContextBudgetBytes == 0 || packet.GenerationControls.MaxOutputTokens != w.AcceptedOutputTokens || !packet.GenerationControls.IgnoreEOS || packet.GenerationControls.Temperature != 0 || packet.GenerationControls.TopP != 1 || packet.GenerationControls.TopK != 1 || len(packet.StopTokens) != 0 || len(packet.StopTokenIDs) != 0 || len(packet.GenerationControls.StopTokens) != 0 || len(packet.GenerationControls.StopTokenIDs) != 0 {
		return PromptTokenPacket{}, errors.New("strix comparison cell packet must bind the exact 26 IDs, GGUF tokenizer, context 32768, output 128, deterministic controls, ignore EOS, and no stops")
	}
	return packet, nil
}

func validateStrixMemory(m StrixComparisonMemoryEnvelope) error {
	if m.ContextBudgetBytes == 0 || m.GPUUMABudgetBytes == 0 || m.CandidateBudgetBytes == 0 || m.ReferenceBudgetBytes == 0 || m.CandidateBudgetBytes > m.GPUUMABudgetBytes || m.ReferenceBudgetBytes > m.GPUUMABudgetBytes || m.ResidentModelBytes == 0 || m.ResidentModelBytes > m.CandidateBudgetBytes || m.ResidentModelBytes > m.ReferenceBudgetBytes {
		return errors.New("strix comparison cell memory byte budgets are invalid")
	}
	for name, value := range map[string]string{
		"kv_type_k": m.KVTypeK, "kv_type_v": m.KVTypeV, "kv_offload": m.KVOffload,
		"host_spill_policy": m.HostSpillPolicy, "primary_cache_state": m.PrimaryCacheState,
		"memory_accounting_policy": m.MemoryAccountingPolicy, "candidate_peak_method": m.CandidatePeakMethod,
		"reference_peak_method": m.ReferencePeakMethod,
	} {
		if invalidManifestString(value) {
			return fmt.Errorf("strix comparison cell %s is required", name)
		}
	}
	if !m.FlashAttention || m.PrimaryCacheState != "cold-no-prefix" || m.MemoryAccountingPolicy != "uma-overlap-not-summed" {
		return errors.New("strix comparison cell cache/attention/accounting policy is not canonical")
	}
	return nil
}

func validateCandidatePin(p StrixComparisonCandidatePin, w StrixComparisonWorkload) error {
	if p.CampaignClass != "fak-native" || p.Runtime != "native" || p.Owner != "fak" || p.Planner != "inkernel" || p.Backend != "vulkan" || p.Fallback || p.FallbackCount != 0 || p.ModelSHA256 != w.ArtifactSHA256 {
		return errors.New("strix comparison cell candidate class/backend/fallback/model pin is invalid")
	}
	for name, value := range map[string]string{"source_revision": p.SourceRevision, "forward_path": p.ForwardPath} {
		if invalidManifestString(value) {
			return fmt.Errorf("strix comparison cell candidate %s is required", name)
		}
	}
	return requireManifestHashes("candidate", map[string]string{
		"source_archive": p.SourceArchiveSHA256, "build_manifest": p.BuildManifestSHA256,
		"toolchain": p.ToolchainSHA256, "executable": p.ExecutableSHA256, "shader_bundle": p.ShaderBundleSHA256,
	})
}

func validateReferencePin(p StrixComparisonReferencePin, w StrixComparisonWorkload) error {
	if p.CampaignClass != "llama.cpp-comparator-only" || p.SourceRevision != StrixComparisonLlamaSourceRevision || p.SourceTreeSHA256 != StrixComparisonLlamaTreeSHA256 || p.BuildType != "Release" || !p.GGMLVulkan || p.ModelSHA256 != w.ArtifactSHA256 || p.PromptPacketDigest != w.PromptPacketDigest || invalidManifestString(p.RADVDeviceIdentity) {
		return errors.New("strix comparison cell llama.cpp comparator pin is invalid")
	}
	return requireManifestHashes("reference", map[string]string{
		"source_archive": p.SourceArchiveSHA256, "build_manifest": p.BuildManifestSHA256,
		"toolchain": p.ToolchainSHA256, "server_binary": p.ServerBinarySHA256,
		"bench_binary": p.BenchBinarySHA256, "loader": p.LoaderSHA256, "dependency": p.DependencySHA256,
	})
}

func validateCapturePlan(p StrixComparisonCapturePlan, c StrixComparisonChallenge) error {
	for name, value := range map[string]string{"cell_nonce": p.CellNonce, "authority_schema": p.AuthoritySchema, "monotonic_clock_id": p.MonotonicClockID, "session_identity": p.SessionIdentity, "replay_key": p.ReplayKey} {
		if invalidManifestString(value) {
			return fmt.Errorf("strix comparison cell capture %s is required", name)
		}
	}
	if !slices.Equal(p.ArmRoles, []string{"candidate", "reference"}) || !slices.Equal(p.PairIDs, c.PairIDs) || !slices.Equal(p.AlternatingOrder, c.AlternatingOrder) || p.TrialCount != 2*c.MeasuredPairs || !slices.Equal(p.ObservationBindings, strixComparisonCaptureBindings) {
		return errors.New("strix comparison cell capture pair/order/trial/observation binding is invalid")
	}
	return nil
}

func requireManifestHashes(prefix string, values map[string]string) error {
	for name, value := range values {
		if !nonEmptyManifestSHA256(value) {
			return fmt.Errorf("strix comparison cell %s %s SHA-256 is required", prefix, name)
		}
	}
	return nil
}

func nonEmptyManifestSHA256(value string) bool {
	return validCanonicalSHA256(value) && value != emptySHA256
}

func invalidManifestString(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return true
	}
	switch strings.ToLower(value) {
	case "unknown", "tbd", "todo", "placeholder":
		return true
	default:
		return false
	}
}

func uniqueRequiredStrings(name string, values []string) error {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if invalidManifestString(value) {
			return fmt.Errorf("strix comparison cell %s contains an empty/placeholder value", name)
		}
		if _, ok := seen[value]; ok {
			return fmt.Errorf("strix comparison cell %s contains duplicate %q", name, value)
		}
		seen[value] = struct{}{}
	}
	return nil
}

func rejectDuplicateJSONFields(raw []byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	var walk func() error
	walk = func() error {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]struct{}{}
			for dec.More() {
				keyToken, err := dec.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok {
					return errors.New("JSON object key is not a string")
				}
				if !canonicalManifestJSONKey(key) {
					return fmt.Errorf("non-canonical JSON field %q", key)
				}
				if _, duplicate := seen[key]; duplicate {
					return fmt.Errorf("duplicate JSON field %q", key)
				}
				seen[key] = struct{}{}
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		case '[':
			for dec.More() {
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delim)
		}
	}
	if err := walk(); err != nil {
		return fmt.Errorf("inspect strix comparison cell JSON: %w", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

func ensureJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

func canonicalManifestJSONKey(key string) bool {
	if key == "" {
		return false
	}
	for _, c := range []byte(key) {
		if c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' {
			continue
		}
		return false
	}
	return true
}
