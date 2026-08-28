package newmodel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modeldescriptor"
)

const (
	NativeHardwareEnvelopeSchema = "fak.new-model-native-hardware-envelope/1"
	NativeObligationGraphSchema  = "fak.new-model-native-obligation-graph/1"

	NativeObligationRequired = "required-correctness"
	NativeObligationFusion   = "optional-fusion"

	NativeLaunchAdmitted                 NativeLaunchDecisionReason = "ADMITTED"
	NativeLaunchDomainInvalid            NativeLaunchDecisionReason = "LAUNCH_DOMAIN_INVALID"
	NativeLaunchEmptyBatch               NativeLaunchDecisionReason = "EMPTY_BATCH_DISALLOWED"
	NativeLaunchUnknownDimension         NativeLaunchDecisionReason = "UNKNOWN_DIMENSION"
	NativeLaunchUnboundedDimension       NativeLaunchDecisionReason = "UNBOUNDED_DIMENSION"
	NativeLaunchContradictoryRange       NativeLaunchDecisionReason = "CONTRADICTORY_DIMENSION_RANGE"
	NativeLaunchShapeOutOfRange          NativeLaunchDecisionReason = "DIMENSION_OUT_OF_RANGE"
	NativeLaunchNonDivisible             NativeLaunchDecisionReason = "DIMENSION_NOT_DIVISIBLE"
	NativeLaunchGridIllegal              NativeLaunchDecisionReason = "GRID_ILLEGAL"
	NativeLaunchBlockIllegal             NativeLaunchDecisionReason = "BLOCK_ILLEGAL"
	NativeLaunchWorkspaceUnbounded       NativeLaunchDecisionReason = "WORKSPACE_UNBOUNDED"
	NativeLaunchWorkspaceExceedsEnvelope NativeLaunchDecisionReason = "WORKSPACE_EXCEEDS_ENVELOPE"
	NativeLaunchOverflow                 NativeLaunchDecisionReason = "LAUNCH_ARITHMETIC_OVERFLOW"
	NativeLaunchBackendLimitsMissing     NativeLaunchDecisionReason = "BACKEND_LIMITS_MISSING"
	NativeLaunchOptionalFusionIneligible NativeLaunchDecisionReason = "OPTIONAL_FUSION_INELIGIBLE"
	NativeLaunchPathFusion               string                     = "optional-fusion"
	NativeLaunchPathCorrectness          string                     = "native-correctness-path"
	NativeLaunchEmptyBatchPolicyReject   string                     = "reject"
)

// NativeLaunchDecisionReason is the closed launch-domain admission vocabulary.
// Every non-admitted result preserves the fak-native correctness path.
type NativeLaunchDecisionReason string

// NativeHardwareEnvelope is one checkpoint-bound planning envelope. It is not
// an execution or performance receipt and cannot select a foreign runtime.
type NativeHardwareEnvelope struct {
	Schema                  string                     `json:"schema"`
	ID                      string                     `json:"id"`
	Engine                  string                     `json:"engine"`
	Platform                string                     `json:"platform"`
	Backend                 string                     `json:"backend"`
	Quantization            string                     `json:"quantization"`
	QuantizationAuthority   string                     `json:"quantization_authority"`
	ArtifactSHA256          string                     `json:"artifact_sha256"`
	WeightLayout            string                     `json:"weight_layout"`
	StateLayout             string                     `json:"state_layout"`
	StateResidency          string                     `json:"state_residency"`
	MemoryBudgetBytes       uint64                     `json:"memory_budget_bytes"`
	ExternalRuntimeFallback bool                       `json:"external_runtime_fallback"`
	LaunchLimits            *NativeBackendLaunchLimits `json:"launch_limits,omitempty"`
	FusionLaunches          []NativeFusionLaunchDomain `json:"fusion_launches,omitempty"`
}

// NativeBackendLaunchLimits are the platform/backend limits admitted by the
// hardware envelope. Zero-valued or mismatched limits are not inferred.
type NativeBackendLaunchLimits struct {
	Platform           string    `json:"platform"`
	Backend            string    `json:"backend"`
	MaxGrid            [3]uint64 `json:"max_grid"`
	MaxBlock           [3]uint64 `json:"max_block"`
	MaxThreadsPerBlock uint64    `json:"max_threads_per_block"`
	MaxWorkspaceBytes  uint64    `json:"max_workspace_bytes"`
}

// NativeFusionLaunchDomain is one concrete launch inside explicitly bounded
// dynamic dimensions. It is evidence for planning, not a runtime allocation.
type NativeFusionLaunchDomain struct {
	Operation          string                  `json:"operation"`
	EmptyBatchPolicy   string                  `json:"empty_batch_policy"`
	Dimensions         []NativeLaunchDimension `json:"dimensions"`
	Grid               [3]uint64               `json:"grid"`
	Block              [3]uint64               `json:"block"`
	WorkspaceBounded   bool                    `json:"workspace_bounded"`
	PeakWorkspaceBytes uint64                  `json:"peak_workspace_bytes"`
}

type NativeLaunchDimension struct {
	Name        string `json:"name"`
	Known       bool   `json:"known"`
	Bounded     bool   `json:"bounded"`
	Value       uint64 `json:"value"`
	Min         uint64 `json:"min"`
	Max         uint64 `json:"max"`
	DivisibleBy uint64 `json:"divisible_by"`
}

type NativeLaunchAdmission struct {
	Engine              string                     `json:"engine"`
	Phase               string                     `json:"phase"`
	Path                string                     `json:"path"`
	Admitted            bool                       `json:"admitted"`
	Reason              NativeLaunchDecisionReason `json:"reason"`
	Detail              string                     `json:"detail"`
	Domain              NativeFusionLaunchDomain   `json:"domain"`
	Limits              *NativeBackendLaunchLimits `json:"limits,omitempty"`
	ResidentBytes       uint64                     `json:"resident_bytes"`
	MemoryEnvelopeBytes uint64                     `json:"memory_envelope_bytes"`
}

type NativeObligationGraph struct {
	Schema                  string             `json:"schema"`
	ManifestDigest          string             `json:"manifest_digest"`
	DescriptorDigest        string             `json:"descriptor_digest"`
	EnvelopeDigest          string             `json:"envelope_digest"`
	Engine                  string             `json:"engine"`
	ExternalRuntimeFallback bool               `json:"external_runtime_fallback"`
	Nodes                   []NativeObligation `json:"nodes"`
}

type NativeObligation struct {
	ID               string                       `json:"id"`
	Class            string                       `json:"class"`
	Reason           string                       `json:"reason"`
	Operation        string                       `json:"operation"`
	DependencyIDs    []string                     `json:"dependency_ids"`
	Oracle           NativeOracleObligation       `json:"oracle"`
	Backend          NativeBackendObligation      `json:"backend"`
	MemoryLayout     NativeMemoryLayoutObligation `json:"memory_layout"`
	PromotionWitness NativePromotionWitness       `json:"promotion_witness"`
	Eligible         bool                         `json:"eligible"`
	Blockers         []string                     `json:"blockers"`
	LaunchAdmission  *NativeLaunchAdmission       `json:"launch_admission,omitempty"`
}

type NativeOracleObligation struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

type NativeBackendObligation struct {
	Engine    string `json:"engine"`
	Platform  string `json:"platform"`
	Backend   string `json:"backend"`
	Operation string `json:"operation"`
	Reason    string `json:"reason"`
}

type NativeMemoryLayoutObligation struct {
	Quantization          string   `json:"quantization"`
	QuantizationAuthority string   `json:"quantization_authority"`
	WeightLayout          string   `json:"weight_layout"`
	StateLayout           string   `json:"state_layout"`
	StateResidency        string   `json:"state_residency"`
	RequiredStateKinds    []string `json:"required_state_kinds"`
	MaxBytes              uint64   `json:"max_bytes"`
	Reason                string   `json:"reason"`
}

type NativePromotionWitness struct {
	ID     string `json:"id"`
	Kind   string `json:"kind"`
	Reason string `json:"reason"`
}

type nativeOperationRule struct {
	Axis, Value              string
	RequiredID, RequiredOp   string
	CandidateID, CandidateOp string
	CandidateBackends        []string
}

// This catalog names operation contracts, not model-family aliases. In
// particular, broad upstream labels never silently bind to GLM DSA or Qwen GDN.
var nativeOperationRules = []nativeOperationRule{
	{Axis: "attention", Value: "hybrid", RequiredID: "correctness.attention-hybrid", RequiredOp: "attention.hybrid-dispatch", CandidateID: "fusion.attention-hybrid", CandidateOp: "fusion.attention-hybrid", CandidateBackends: []string{"cuda"}},
	{Axis: "ffn", Value: "swiglu", RequiredID: "correctness.ffn-swiglu", RequiredOp: "ffn.swiglu", CandidateID: "fusion.ffn-swiglu", CandidateOp: "fusion.ffn-swiglu"},
	{Axis: "normalization", Value: "rmsnorm", RequiredID: "correctness.normalization-rmsnorm", RequiredOp: "normalization.rmsnorm"},
	{Axis: "position", Value: "rope", RequiredID: "correctness.position-rope", RequiredOp: "position.rope"},
	{Axis: "routing", Value: "moe-topk", RequiredID: "correctness.routing-moe-topk", RequiredOp: "routing.moe-topk", CandidateID: "fusion.routing-moe-topk", CandidateOp: "fusion.routing-moe-topk"},
	{Axis: "state", Value: "hybrid", RequiredID: "correctness.state-hybrid", RequiredOp: "state.hybrid-lifecycle", CandidateID: "fusion.state-hybrid", CandidateOp: "fusion.state-hybrid-update", CandidateBackends: []string{"cuda"}},
}

func ParseNativeHardwareEnvelope(raw []byte) (NativeHardwareEnvelope, error) {
	var envelope NativeHardwareEnvelope
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&envelope); err != nil {
		return NativeHardwareEnvelope{}, refuse(RefusalHardwareEnvelopeInvalid, "hardware_envelope.json", err.Error())
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return NativeHardwareEnvelope{}, refuse(RefusalHardwareEnvelopeInvalid, "hardware_envelope.json", "multiple JSON values")
	}
	return envelope, nil
}

// CompileNativeObligationGraph is a pure pre-allocation compiler. It plans
// correctness and possible fusion work without importing or invoking a runtime.
func CompileNativeObligationGraph(packet Packet, envelope NativeHardwareEnvelope) (NativeObligationGraph, error) {
	if packet.Schema != PacketSchema || packet.ManifestDigest == "" {
		return NativeObligationGraph{}, refuse(RefusalManifestInvalid, "packet", "a validated onboarding packet is required")
	}
	if packet.Engine != "fak-native" || packet.Descriptor.Engine != "fak-native" || packet.ExternalRuntimeFallback {
		return NativeObligationGraph{}, refuse(RefusalNativeEngineMismatch, "engine", "packet must remain fak-native with external runtime fallback disabled")
	}
	envelope = normalizeNativeLaunchEnvelope(envelope)
	descriptor := packet.Descriptor.ModelDescriptor()
	if err := modeldescriptor.Validate(descriptor); err != nil {
		return NativeObligationGraph{}, refuse(RefusalDescriptorInvalid, "descriptor", err.Error())
	}
	descriptorDigest, err := modeldescriptor.Digest(descriptor)
	if err != nil {
		return NativeObligationGraph{}, refuse(RefusalDescriptorInvalid, "descriptor", err.Error())
	}
	if descriptorDigest != packet.Coupling.DescriptorDigest {
		return NativeObligationGraph{}, refuse(RefusalDescriptorInvalid, "packet.coupling.descriptor_digest", "packet descriptor digest no longer matches its validated coupling report")
	}
	if err := validateNativeEnvelope(packet, envelope); err != nil {
		return NativeObligationGraph{}, err
	}
	stateKinds, stateBytes, err := nativeStateRequirement(packet.Descriptor.State)
	if err != nil {
		return NativeObligationGraph{}, err
	}
	if stateBytes > envelope.MemoryBudgetBytes {
		return NativeObligationGraph{}, refuse(RefusalUnsupportedNativeCombination, "hardware_envelope.memory_budget_bytes", fmt.Sprintf("descriptor state requires %d bytes, envelope permits %d", stateBytes, envelope.MemoryBudgetBytes))
	}

	rules := make([]nativeOperationRule, 0, len(packet.SemanticDeltas))
	for _, delta := range packet.SemanticDeltas {
		rule, ok := nativeOperationRuleFor(delta.Axis, delta.Value)
		if !ok {
			return NativeObligationGraph{}, refuse(RefusalUnknownNativeOperation, delta.Axis, fmt.Sprintf("no native operation contract for %q", delta.Value))
		}
		rules = append(rules, rule)
	}
	sort.Slice(rules, func(i, j int) bool { return rules[i].RequiredID < rules[j].RequiredID })

	constraint := nativeMemoryLayout(envelope, stateKinds)
	oracleID := packet.Descriptor.Oracles[0]
	nodes := []NativeObligation{
		nativeNode("correctness.semantic-verification", NativeObligationRequired, "verify.semantic-contract", nil, oracleID, envelope, constraint, "oracle-test", "verify-upstream-semantics", "The upstream semantic contract must be witnessed before any native binding."),
		nativeNode("correctness.tensor-weight-reachability", NativeObligationRequired, "checkpoint.tensor-weight-reachability", []string{"correctness.semantic-verification"}, "checkpoint-tensor-index", envelope, constraint, "conformance-test", "prove-tensor-weight-reachability", "Every required tensor and checkpoint-authoritative quantized weight must reach its native consumer."),
	}
	for _, rule := range rules {
		required := nativeNode(rule.RequiredID, NativeObligationRequired, rule.RequiredOp, []string{"correctness.tensor-weight-reachability"}, oracleID, envelope, constraint, "parity-test", "prove-"+strings.ReplaceAll(rule.RequiredOp, ".", "-"), "Correctness for the admitted semantic operation must precede fusion.")
		nodes = append(nodes, required)
		if rule.CandidateID != "" {
			candidate := nativeNode(rule.CandidateID, NativeObligationFusion, rule.CandidateOp, []string{rule.RequiredID}, oracleID, envelope, constraint, "promotion-test", "promote-"+strings.ReplaceAll(rule.CandidateOp, ".", "-"), "Fusion is optional and may be promoted only after its unfused correctness dependency passes.")
			domain, domainCount := nativeFusionLaunchFor(envelope.FusionLaunches, rule.CandidateOp)
			admission := admitNativeFusionLaunch(
				contains(rule.CandidateBackends, envelope.Backend),
				rule.CandidateOp,
				domain,
				domainCount,
				envelope.LaunchLimits,
				envelope.Platform,
				envelope.Backend,
				stateBytes,
				envelope.MemoryBudgetBytes,
			)
			candidate.LaunchAdmission = &admission
			candidate.Eligible = admission.Admitted
			if !admission.Admitted {
				candidate.Blockers = []string{"launch:" + string(admission.Reason) + ":" + admission.Detail}
			}
			nodes = append(nodes, candidate)
		}
	}
	nodes, err = orderNativeObligations(nodes)
	if err != nil {
		return NativeObligationGraph{}, err
	}
	envelopeDigest, err := digestNativeJSON(envelope)
	if err != nil {
		return NativeObligationGraph{}, refuse(RefusalHardwareEnvelopeInvalid, "hardware_envelope", err.Error())
	}
	return NativeObligationGraph{
		Schema: NativeObligationGraphSchema, ManifestDigest: packet.ManifestDigest,
		DescriptorDigest: descriptorDigest, EnvelopeDigest: envelopeDigest,
		Engine: "fak-native", ExternalRuntimeFallback: false, Nodes: nodes,
	}, nil
}

func MarshalNativeObligationGraph(graph NativeObligationGraph) ([]byte, error) {
	raw, err := json.MarshalIndent(graph, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func validateNativeEnvelope(packet Packet, envelope NativeHardwareEnvelope) error {
	if envelope.Schema != NativeHardwareEnvelopeSchema || envelope.ID == "" || envelope.Platform == "" || envelope.MemoryBudgetBytes == 0 {
		return refuse(RefusalHardwareEnvelopeInvalid, "hardware_envelope", "schema, id, platform, and a positive memory budget are required")
	}
	if envelope.Engine != "fak-native" || envelope.ExternalRuntimeFallback {
		return refuse(RefusalNativeEngineMismatch, "hardware_envelope.engine", "envelope must remain fak-native with external runtime fallback disabled")
	}
	if !knownNativeQuantization(envelope.Quantization) {
		return refuse(RefusalUnknownNativeQuantization, "hardware_envelope.quantization", fmt.Sprintf("unknown quantization %q", envelope.Quantization))
	}
	if !knownNativeLayout(envelope.WeightLayout, envelope.StateLayout, envelope.StateResidency) {
		return refuse(RefusalUnknownNativeLayout, "hardware_envelope.layout", fmt.Sprintf("unknown weight/state layout %q/%q/%q", envelope.WeightLayout, envelope.StateLayout, envelope.StateResidency))
	}
	if !contains([]string{"cuda", "metal", "cpu"}, envelope.Backend) {
		return refuse(RefusalUnsupportedNativeCombination, "hardware_envelope.backend", fmt.Sprintf("unsupported native backend %q", envelope.Backend))
	}
	if envelope.QuantizationAuthority != "checkpoint" || envelope.ArtifactSHA256 != packet.Artifact.SHA256 {
		return refuse(RefusalUnsupportedNativeCombination, "hardware_envelope.quantization_authority", "quantization must be checkpoint-authoritative and bound to the packet artifact digest")
	}
	if !contains(packet.Descriptor.Envelopes, envelope.ID) || !contains(packet.Descriptor.Backends, envelope.Backend) || !contains(packet.Descriptor.Quantization, envelope.Quantization) {
		return refuse(RefusalUnsupportedNativeCombination, "hardware_envelope", "envelope id, backend, and quantization must be declared by the descriptor")
	}
	key := strings.Join([]string{envelope.Platform, envelope.Backend, envelope.Quantization, envelope.WeightLayout, envelope.StateLayout, envelope.StateResidency}, "|")
	if key != "linux/amd64+nvidia-a100|cuda|q4-k-m|gguf-q4-k|contiguous|device" {
		return refuse(RefusalUnsupportedNativeCombination, "hardware_envelope", fmt.Sprintf("native combination %q has no correctness contract", key))
	}
	return nil
}

func knownNativeQuantization(value string) bool {
	return contains([]string{"f16", "q4-k-m", "q8-0"}, value)
}

func knownNativeLayout(weight, state, residency string) bool {
	return contains([]string{"gguf-q4-k", "row-major"}, weight) && state == "contiguous" && contains([]string{"device", "host"}, residency)
}

func nativeOperationRuleFor(axis, value string) (nativeOperationRule, bool) {
	for _, rule := range nativeOperationRules {
		if rule.Axis == axis && rule.Value == value {
			return rule, true
		}
	}
	return nativeOperationRule{}, false
}

func nativeFusionLaunchFor(domains []NativeFusionLaunchDomain, operation string) (NativeFusionLaunchDomain, int) {
	var match NativeFusionLaunchDomain
	count := 0
	for _, domain := range domains {
		if domain.Operation == operation {
			match = domain
			count++
		}
	}
	return match, count
}

func normalizeNativeLaunchEnvelope(envelope NativeHardwareEnvelope) NativeHardwareEnvelope {
	envelope.FusionLaunches = append([]NativeFusionLaunchDomain(nil), envelope.FusionLaunches...)
	for i := range envelope.FusionLaunches {
		envelope.FusionLaunches[i].Dimensions = append([]NativeLaunchDimension(nil), envelope.FusionLaunches[i].Dimensions...)
		sort.Slice(envelope.FusionLaunches[i].Dimensions, func(a, b int) bool {
			return envelope.FusionLaunches[i].Dimensions[a].Name < envelope.FusionLaunches[i].Dimensions[b].Name
		})
	}
	sort.Slice(envelope.FusionLaunches, func(i, j int) bool {
		return envelope.FusionLaunches[i].Operation < envelope.FusionLaunches[j].Operation
	})
	return envelope
}

func admitNativeFusionLaunch(
	optionalFusionEligible bool,
	operation string,
	domain NativeFusionLaunchDomain,
	domainCount int,
	limits *NativeBackendLaunchLimits,
	platform string,
	backend string,
	residentBytes uint64,
	memoryEnvelopeBytes uint64,
) NativeLaunchAdmission {
	decision := nativeLaunchDecision(domain, limits, residentBytes, memoryEnvelopeBytes)
	refuseLaunch := func(reason NativeLaunchDecisionReason, detail string) NativeLaunchAdmission {
		decision.Path = NativeLaunchPathCorrectness
		decision.Admitted = false
		decision.Reason = reason
		decision.Detail = detail
		return decision
	}
	if !optionalFusionEligible {
		return refuseLaunch(NativeLaunchOptionalFusionIneligible, fmt.Sprintf("operation %q has no checkpoint-specific fusion witness for backend %q", operation, backend))
	}
	if domainCount != 1 || operation == "" || domain.Operation != operation {
		return refuseLaunch(NativeLaunchDomainInvalid, fmt.Sprintf("operation %q requires exactly one matching launch domain; found %d", operation, domainCount))
	}
	if limits == nil || limits.Platform == "" || limits.Backend == "" || limits.Platform != platform || limits.Backend != backend || limits.MaxThreadsPerBlock == 0 || limits.MaxWorkspaceBytes == 0 || hasZeroAxis(limits.MaxGrid) || hasZeroAxis(limits.MaxBlock) {
		return refuseLaunch(NativeLaunchBackendLimitsMissing, fmt.Sprintf("complete launch limits for %s/%s are required", platform, backend))
	}
	if domain.EmptyBatchPolicy != NativeLaunchEmptyBatchPolicyReject || len(domain.Dimensions) == 0 {
		return refuseLaunch(NativeLaunchDomainInvalid, "a non-empty dimension set and the reject empty-batch policy are required")
	}
	seenDimensions := make(map[string]struct{}, len(domain.Dimensions))
	for _, dimension := range domain.Dimensions {
		if dimension.Name == "" {
			return refuseLaunch(NativeLaunchUnknownDimension, "dimension name is unknown")
		}
		if _, duplicate := seenDimensions[dimension.Name]; duplicate {
			return refuseLaunch(NativeLaunchDomainInvalid, fmt.Sprintf("dimension %q is declared more than once", dimension.Name))
		}
		seenDimensions[dimension.Name] = struct{}{}
		if !dimension.Known {
			return refuseLaunch(NativeLaunchUnknownDimension, fmt.Sprintf("dimension %q has no concrete value", dimension.Name))
		}
		if !dimension.Bounded {
			return refuseLaunch(NativeLaunchUnboundedDimension, fmt.Sprintf("dimension %q has no finite bounds", dimension.Name))
		}
		if dimension.Min > dimension.Max {
			return refuseLaunch(NativeLaunchContradictoryRange, fmt.Sprintf("dimension %q has min %d greater than max %d", dimension.Name, dimension.Min, dimension.Max))
		}
		if dimension.Name == "batch" && dimension.Value == 0 {
			return refuseLaunch(NativeLaunchEmptyBatch, "batch dimension is empty and policy is reject")
		}
		if dimension.Value < dimension.Min || dimension.Value > dimension.Max {
			return refuseLaunch(NativeLaunchShapeOutOfRange, fmt.Sprintf("dimension %q value %d is outside [%d,%d]", dimension.Name, dimension.Value, dimension.Min, dimension.Max))
		}
		if dimension.DivisibleBy == 0 || dimension.Value%dimension.DivisibleBy != 0 {
			return refuseLaunch(NativeLaunchNonDivisible, fmt.Sprintf("dimension %q value %d is not divisible by %d", dimension.Name, dimension.Value, dimension.DivisibleBy))
		}
	}
	if _, found := seenDimensions["batch"]; !found {
		return refuseLaunch(NativeLaunchUnknownDimension, "batch dimension is not declared")
	}
	for axis := range domain.Grid {
		if domain.Grid[axis] == 0 || domain.Grid[axis] > limits.MaxGrid[axis] {
			return refuseLaunch(NativeLaunchGridIllegal, fmt.Sprintf("grid axis %d value %d exceeds legal range [1,%d]", axis, domain.Grid[axis], limits.MaxGrid[axis]))
		}
	}
	gridSize, ok := checkedProduct(domain.Grid[:])
	if !ok {
		return refuseLaunch(NativeLaunchOverflow, "grid size overflows uint64")
	}
	for axis := range domain.Block {
		if domain.Block[axis] == 0 || domain.Block[axis] > limits.MaxBlock[axis] {
			return refuseLaunch(NativeLaunchBlockIllegal, fmt.Sprintf("block axis %d value %d exceeds legal range [1,%d]", axis, domain.Block[axis], limits.MaxBlock[axis]))
		}
	}
	blockSize, ok := checkedProduct(domain.Block[:])
	if !ok {
		return refuseLaunch(NativeLaunchOverflow, "block size overflows uint64")
	}
	if blockSize > limits.MaxThreadsPerBlock {
		return refuseLaunch(NativeLaunchBlockIllegal, fmt.Sprintf("block has %d threads; backend permits %d", blockSize, limits.MaxThreadsPerBlock))
	}
	if gridSize > math.MaxUint64/blockSize {
		return refuseLaunch(NativeLaunchOverflow, "total launch threads overflow uint64")
	}
	if !domain.WorkspaceBounded {
		return refuseLaunch(NativeLaunchWorkspaceUnbounded, "peak workspace has no finite bound")
	}
	if domain.PeakWorkspaceBytes > limits.MaxWorkspaceBytes {
		return refuseLaunch(NativeLaunchWorkspaceExceedsEnvelope, fmt.Sprintf("peak workspace %d exceeds backend limit %d", domain.PeakWorkspaceBytes, limits.MaxWorkspaceBytes))
	}
	if residentBytes > math.MaxUint64-domain.PeakWorkspaceBytes {
		return refuseLaunch(NativeLaunchOverflow, "resident bytes plus peak workspace overflow uint64")
	}
	peakBytes := residentBytes + domain.PeakWorkspaceBytes
	if peakBytes > memoryEnvelopeBytes {
		return refuseLaunch(NativeLaunchWorkspaceExceedsEnvelope, fmt.Sprintf("resident plus peak workspace requires %d bytes; envelope permits %d", peakBytes, memoryEnvelopeBytes))
	}
	decision.Path = NativeLaunchPathFusion
	decision.Admitted = true
	decision.Reason = NativeLaunchAdmitted
	decision.Detail = fmt.Sprintf("operation %q launch is within the declared %s/%s domain", operation, platform, backend)
	return decision
}

func nativeLaunchDecision(domain NativeFusionLaunchDomain, limits *NativeBackendLaunchLimits, residentBytes, memoryEnvelopeBytes uint64) NativeLaunchAdmission {
	domain.Dimensions = append([]NativeLaunchDimension(nil), domain.Dimensions...)
	var limitsCopy *NativeBackendLaunchLimits
	if limits != nil {
		copy := *limits
		limitsCopy = &copy
	}
	return NativeLaunchAdmission{
		Engine: "fak-native", Phase: "pre-allocation", Domain: domain, Limits: limitsCopy,
		ResidentBytes: residentBytes, MemoryEnvelopeBytes: memoryEnvelopeBytes,
	}
}

func hasZeroAxis(values [3]uint64) bool {
	return values[0] == 0 || values[1] == 0 || values[2] == 0
}

func checkedProduct(values []uint64) (uint64, bool) {
	product := uint64(1)
	for _, value := range values {
		if value == 0 || product > math.MaxUint64/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}

func nativeStateRequirement(state []modeldescriptor.Geometry) ([]string, uint64, error) {
	kinds := make([]string, 0, len(state))
	var total uint64
	for _, geometry := range state {
		count := uint64(1)
		for _, dimension := range geometry.Shape {
			if dimension <= 0 || count > math.MaxUint64/uint64(dimension) {
				return nil, 0, refuse(RefusalHardwareEnvelopeInvalid, "descriptor.state", "state geometry overflows the planning envelope")
			}
			count *= uint64(dimension)
		}
		if geometry.BytesPerElement <= 0 || count > math.MaxUint64/uint64(geometry.BytesPerElement) {
			return nil, 0, refuse(RefusalHardwareEnvelopeInvalid, "descriptor.state", "state byte size overflows the planning envelope")
		}
		bytes := count * uint64(geometry.BytesPerElement)
		if total > math.MaxUint64-bytes {
			return nil, 0, refuse(RefusalHardwareEnvelopeInvalid, "descriptor.state", "total state bytes overflow the planning envelope")
		}
		total += bytes
		kinds = append(kinds, geometry.Kind)
	}
	sort.Strings(kinds)
	return kinds, total, nil
}

func nativeMemoryLayout(envelope NativeHardwareEnvelope, stateKinds []string) NativeMemoryLayoutObligation {
	return NativeMemoryLayoutObligation{
		Quantization: envelope.Quantization, QuantizationAuthority: envelope.QuantizationAuthority,
		WeightLayout: envelope.WeightLayout, StateLayout: envelope.StateLayout, StateResidency: envelope.StateResidency,
		RequiredStateKinds: append([]string(nil), stateKinds...), MaxBytes: envelope.MemoryBudgetBytes,
		Reason: "The checkpoint-authoritative quantization and tensor/state layout must remain reachable within the declared memory budget.",
	}
}

func nativeNode(id, class, operation string, deps []string, oracleID string, envelope NativeHardwareEnvelope, constraint NativeMemoryLayoutObligation, witnessKind, witnessID, reason string) NativeObligation {
	return NativeObligation{
		ID: id, Class: class, Reason: reason, Operation: operation, DependencyIDs: append([]string{}, deps...),
		Oracle:           NativeOracleObligation{ID: oracleID, Reason: "The named oracle must witness this operation without turning planning into a support claim."},
		Backend:          NativeBackendObligation{Engine: "fak-native", Platform: envelope.Platform, Backend: envelope.Backend, Operation: operation, Reason: "The operation remains owned by the selected fak-native platform/backend; no foreign fallback is permitted."},
		MemoryLayout:     constraint,
		PromotionWitness: NativePromotionWitness{ID: witnessID, Kind: witnessKind, Reason: "This prospective witness must pass before the obligation can be promoted."},
		Eligible:         true, Blockers: []string{},
	}
}

func orderNativeObligations(nodes []NativeObligation) ([]NativeObligation, error) {
	byID := make(map[string]NativeObligation, len(nodes))
	indegree := make(map[string]int, len(nodes))
	dependents := make(map[string][]string, len(nodes))
	for _, node := range nodes {
		if node.ID == "" || byID[node.ID].ID != "" {
			return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", fmt.Sprintf("duplicate or empty node id %q", node.ID))
		}
		if node.Reason == "" || node.Operation == "" || node.Oracle.ID == "" || node.Oracle.Reason == "" || node.Backend.Engine != "fak-native" || node.Backend.Platform == "" || node.Backend.Backend == "" || node.Backend.Reason == "" || node.MemoryLayout.Reason == "" || node.PromotionWitness.ID == "" || node.PromotionWitness.Kind == "" || node.PromotionWitness.Reason == "" {
			return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", fmt.Sprintf("node %q has an incomplete reason-bearing obligation", node.ID))
		}
		if node.Class == NativeObligationRequired && node.LaunchAdmission != nil {
			return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", fmt.Sprintf("required node %q unexpectedly carries fusion admission", node.ID))
		}
		if node.Class == NativeObligationFusion {
			admission := node.LaunchAdmission
			if admission == nil || admission.Engine != "fak-native" || admission.Phase != "pre-allocation" || admission.Reason == "" || admission.Detail == "" {
				return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", fmt.Sprintf("fusion node %q lacks complete launch admission", node.ID))
			}
			wantPath := NativeLaunchPathCorrectness
			if admission.Admitted {
				wantPath = NativeLaunchPathFusion
			}
			if admission.Path != wantPath || node.Eligible != admission.Admitted {
				return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", fmt.Sprintf("fusion node %q has contradictory launch admission", node.ID))
			}
		}
		byID[node.ID] = node
	}
	for _, node := range nodes {
		for _, dep := range node.DependencyIDs {
			if byID[dep].ID == "" {
				return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", fmt.Sprintf("node %q depends on unknown node %q", node.ID, dep))
			}
			indegree[node.ID]++
			dependents[dep] = append(dependents[dep], node.ID)
		}
	}
	ready := make([]string, 0)
	for id := range byID {
		if indegree[id] == 0 {
			ready = append(ready, id)
		}
	}
	sort.Strings(ready)
	ordered := make([]NativeObligation, 0, len(nodes))
	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]
		ordered = append(ordered, byID[id])
		for _, dependent := range dependents[id] {
			indegree[dependent]--
			if indegree[dependent] == 0 {
				ready = append(ready, dependent)
				sort.Strings(ready)
			}
		}
	}
	if len(ordered) != len(nodes) {
		return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", "dependency cycle")
	}
	seenFusion := false
	for _, node := range ordered {
		seenFusion = seenFusion || node.Class == NativeObligationFusion
		if seenFusion && node.Class == NativeObligationRequired {
			return nil, refuse(RefusalUnsupportedNativeCombination, "obligation_graph", "required correctness node sorted after optional fusion")
		}
	}
	return ordered, nil
}

func digestNativeJSON(value any) (string, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}
