package newmodel

import (
	"fmt"
	"math"
	"sort"
)

const (
	NativeEmptyBatchCorrectnessPath = "native-correctness-path"

	NativeLaunchDecisionAdmitted        NativeLaunchDecision = "admitted"
	NativeLaunchDecisionCorrectnessPath NativeLaunchDecision = "native-correctness-path"

	NativeLaunchReasonAdmitted               NativeLaunchReason = "launch-admitted"
	NativeLaunchReasonFusionIneligible       NativeLaunchReason = "fusion-ineligible"
	NativeLaunchReasonDomainMissing          NativeLaunchReason = "launch-domain-missing"
	NativeLaunchReasonDomainContradictory    NativeLaunchReason = "launch-domain-contradictory"
	NativeLaunchReasonLimitsMissing          NativeLaunchReason = "launch-limits-missing"
	NativeLaunchReasonLimitsMismatch         NativeLaunchReason = "launch-limits-mismatch"
	NativeLaunchReasonEmptyBatch             NativeLaunchReason = "empty-batch"
	NativeLaunchReasonEmptyPolicyUnknown     NativeLaunchReason = "empty-batch-policy-unknown"
	NativeLaunchReasonDimensionUnknown       NativeLaunchReason = "launch-dimension-unknown"
	NativeLaunchReasonDimensionContradictory NativeLaunchReason = "launch-dimension-contradictory"
	NativeLaunchReasonDimensionOversized     NativeLaunchReason = "launch-dimension-oversized"
	NativeLaunchReasonDimensionNonDivisible  NativeLaunchReason = "launch-dimension-non-divisible"
	NativeLaunchReasonShapeOverflow          NativeLaunchReason = "launch-shape-overflow"
	NativeLaunchReasonBlockIllegal           NativeLaunchReason = "launch-block-illegal"
	NativeLaunchReasonGridIllegal            NativeLaunchReason = "launch-grid-illegal"
	NativeLaunchReasonWorkspaceUnbounded     NativeLaunchReason = "launch-workspace-unbounded"
	NativeLaunchReasonWorkspaceExceedsLimit  NativeLaunchReason = "launch-workspace-exceeds-limit"
)

type NativeLaunchDecision string

type NativeLaunchReason string

// NativeLaunchLimits are checkpoint-envelope limits for one admitted backend
// and platform. Zero means missing evidence, never an unbounded allowance.
type NativeLaunchLimits struct {
	Platform          string `json:"platform"`
	Backend           string `json:"backend"`
	MaxDimensionValue uint64 `json:"max_dimension_value"`
	MaxGridBlocks     uint64 `json:"max_grid_blocks"`
	MaxBlockThreads   uint64 `json:"max_block_threads"`
	MaxWorkspaceBytes uint64 `json:"max_workspace_bytes"`
}

// NativeFusionLaunchDomain is a closed prospective launch shape. It is input
// to pre-allocation planning only and neither allocates nor invokes a kernel.
type NativeFusionLaunchDomain struct {
	Operation          string                  `json:"operation"`
	EmptyBatchPolicy   string                  `json:"empty_batch_policy"`
	Dimensions         []NativeLaunchDimension `json:"dimensions"`
	GridBlocks         uint64                  `json:"grid_blocks"`
	BlockThreads       uint64                  `json:"block_threads"`
	WorkspaceBounded   bool                    `json:"workspace_bounded"`
	PeakWorkspaceBytes uint64                  `json:"peak_workspace_bytes"`
}

type NativeLaunchDimension struct {
	Name        string `json:"name"`
	Value       uint64 `json:"value"`
	Min         uint64 `json:"min"`
	Max         uint64 `json:"max"`
	DivisibleBy uint64 `json:"divisible_by"`
}

// NativeLaunchAdmission records why an optional fusion may launch or why the
// already-required fak-native correctness path remains selected.
type NativeLaunchAdmission struct {
	Decision           NativeLaunchDecision `json:"decision"`
	ReasonCode         NativeLaunchReason   `json:"reason_code"`
	Reason             string               `json:"reason"`
	Operation          string               `json:"operation"`
	ShapeElements      uint64               `json:"shape_elements"`
	GridBlocks         uint64               `json:"grid_blocks"`
	BlockThreads       uint64               `json:"block_threads"`
	PeakWorkspaceBytes uint64               `json:"peak_workspace_bytes"`
}

func normalizeNativeLaunchEnvelope(envelope NativeHardwareEnvelope) NativeHardwareEnvelope {
	normalized := envelope
	normalized.FusionLaunchDomains = append([]NativeFusionLaunchDomain(nil), envelope.FusionLaunchDomains...)
	for i := range normalized.FusionLaunchDomains {
		normalized.FusionLaunchDomains[i].Dimensions = append([]NativeLaunchDimension(nil), normalized.FusionLaunchDomains[i].Dimensions...)
		sort.SliceStable(normalized.FusionLaunchDomains[i].Dimensions, func(a, b int) bool {
			return normalized.FusionLaunchDomains[i].Dimensions[a].Name < normalized.FusionLaunchDomains[i].Dimensions[b].Name
		})
	}
	sort.SliceStable(normalized.FusionLaunchDomains, func(i, j int) bool {
		return normalized.FusionLaunchDomains[i].Operation < normalized.FusionLaunchDomains[j].Operation
	})
	return normalized
}

func admitNativeFusionLaunch(operation string, backendEligible bool, envelope NativeHardwareEnvelope) *NativeLaunchAdmission {
	if !backendEligible {
		return nativeLaunchDecision(operation, NativeLaunchReasonFusionIneligible, "the optional fusion has no witnessed implementation for this fak-native backend", NativeFusionLaunchDomain{}, 0)
	}
	limits := envelope.LaunchLimits
	if limits.Platform == "" || limits.Backend == "" || limits.MaxDimensionValue == 0 || limits.MaxGridBlocks == 0 || limits.MaxBlockThreads == 0 || limits.MaxWorkspaceBytes == 0 {
		return nativeLaunchDecision(operation, NativeLaunchReasonLimitsMissing, "the backend/platform launch limits are incomplete", NativeFusionLaunchDomain{}, 0)
	}
	if limits.Platform != envelope.Platform || limits.Backend != envelope.Backend {
		return nativeLaunchDecision(operation, NativeLaunchReasonLimitsMismatch, "the launch limits do not belong to the admitted backend/platform", NativeFusionLaunchDomain{}, 0)
	}
	domain, count := nativeFusionLaunchDomainFor(operation, envelope.FusionLaunchDomains)
	if count == 0 {
		return nativeLaunchDecision(operation, NativeLaunchReasonDomainMissing, "the optional fusion has no bounded launch domain", NativeFusionLaunchDomain{}, 0)
	}
	if count != 1 {
		return nativeLaunchDecision(operation, NativeLaunchReasonDomainContradictory, "the optional fusion has more than one launch domain", domain, 0)
	}
	if domain.EmptyBatchPolicy != NativeEmptyBatchCorrectnessPath {
		return nativeLaunchDecision(operation, NativeLaunchReasonEmptyPolicyUnknown, "the empty-batch policy is not recognized", domain, 0)
	}
	if len(domain.Dimensions) == 0 {
		return nativeLaunchDecision(operation, NativeLaunchReasonDimensionUnknown, "the launch shape has no dimensions", domain, 0)
	}

	seen := make(map[string]bool, len(domain.Dimensions))
	batchSeen := false
	emptyBatch := false
	for _, dimension := range domain.Dimensions {
		if !knownNativeLaunchDimension(dimension.Name) || seen[dimension.Name] {
			return nativeLaunchDecision(operation, NativeLaunchReasonDimensionUnknown, fmt.Sprintf("launch dimension %q is unknown or duplicated", dimension.Name), domain, 0)
		}
		seen[dimension.Name] = true
		if dimension.Min == 0 || dimension.Max == 0 || dimension.DivisibleBy == 0 {
			return nativeLaunchDecision(operation, NativeLaunchReasonDimensionUnknown, fmt.Sprintf("launch dimension %q lacks a bound or divisor", dimension.Name), domain, 0)
		}
		if dimension.Min > dimension.Max {
			return nativeLaunchDecision(operation, NativeLaunchReasonDimensionContradictory, fmt.Sprintf("launch dimension %q has min greater than max", dimension.Name), domain, 0)
		}
		if dimension.Name == "batch" {
			batchSeen = true
			emptyBatch = dimension.Value == 0
		} else if dimension.Value == 0 {
			return nativeLaunchDecision(operation, NativeLaunchReasonDimensionUnknown, fmt.Sprintf("launch dimension %q has no concrete value", dimension.Name), domain, 0)
		}
	}
	if !batchSeen {
		return nativeLaunchDecision(operation, NativeLaunchReasonDimensionUnknown, "the launch shape has no batch dimension", domain, 0)
	}
	if emptyBatch {
		return nativeLaunchDecision(operation, NativeLaunchReasonEmptyBatch, "empty batches stay on the native correctness path without allocation", domain, 0)
	}

	shapeElements := uint64(1)
	for _, dimension := range domain.Dimensions {
		if dimension.Max > limits.MaxDimensionValue || dimension.Value < dimension.Min || dimension.Value > dimension.Max {
			return nativeLaunchDecision(operation, NativeLaunchReasonDimensionOversized, fmt.Sprintf("launch dimension %q falls outside the admitted bounds", dimension.Name), domain, shapeElements)
		}
		if dimension.Value%dimension.DivisibleBy != 0 {
			return nativeLaunchDecision(operation, NativeLaunchReasonDimensionNonDivisible, fmt.Sprintf("launch dimension %q violates divisibility by %d", dimension.Name, dimension.DivisibleBy), domain, shapeElements)
		}
		if shapeElements > math.MaxUint64/dimension.Value {
			return nativeLaunchDecision(operation, NativeLaunchReasonShapeOverflow, "the launch shape element count overflows uint64", domain, 0)
		}
		shapeElements *= dimension.Value
	}
	if domain.BlockThreads == 0 || domain.BlockThreads > limits.MaxBlockThreads {
		return nativeLaunchDecision(operation, NativeLaunchReasonBlockIllegal, "the launch block is outside the admitted backend limit", domain, shapeElements)
	}
	if domain.GridBlocks == 0 || domain.GridBlocks > limits.MaxGridBlocks {
		return nativeLaunchDecision(operation, NativeLaunchReasonGridIllegal, "the launch grid is outside the admitted backend limit", domain, shapeElements)
	}
	if shapeElements%domain.BlockThreads != 0 {
		return nativeLaunchDecision(operation, NativeLaunchReasonDimensionNonDivisible, "the launch shape is not divisible by the block size", domain, shapeElements)
	}
	if shapeElements/domain.BlockThreads != domain.GridBlocks {
		return nativeLaunchDecision(operation, NativeLaunchReasonGridIllegal, "the launch grid does not cover the declared shape exactly", domain, shapeElements)
	}
	if !domain.WorkspaceBounded {
		return nativeLaunchDecision(operation, NativeLaunchReasonWorkspaceUnbounded, "the optional fusion has no finite peak workspace", domain, shapeElements)
	}
	workspaceLimit := limits.MaxWorkspaceBytes
	if envelope.MemoryBudgetBytes < workspaceLimit {
		workspaceLimit = envelope.MemoryBudgetBytes
	}
	if domain.PeakWorkspaceBytes > workspaceLimit {
		return nativeLaunchDecision(operation, NativeLaunchReasonWorkspaceExceedsLimit, "the peak workspace exceeds the backend/platform memory envelope", domain, shapeElements)
	}

	return &NativeLaunchAdmission{
		Decision: NativeLaunchDecisionAdmitted, ReasonCode: NativeLaunchReasonAdmitted,
		Reason: "the bounded launch shape fits the admitted backend/platform limits before allocation", Operation: operation,
		ShapeElements: shapeElements, GridBlocks: domain.GridBlocks, BlockThreads: domain.BlockThreads,
		PeakWorkspaceBytes: domain.PeakWorkspaceBytes,
	}
}

func nativeFusionLaunchDomainFor(operation string, domains []NativeFusionLaunchDomain) (NativeFusionLaunchDomain, int) {
	var found NativeFusionLaunchDomain
	count := 0
	for _, domain := range domains {
		if domain.Operation == operation {
			found = domain
			count++
		}
	}
	return found, count
}

func knownNativeLaunchDimension(name string) bool {
	return name == "batch" || name == "channels" || name == "tokens"
}

func nativeLaunchDecision(operation string, reason NativeLaunchReason, detail string, domain NativeFusionLaunchDomain, shapeElements uint64) *NativeLaunchAdmission {
	return &NativeLaunchAdmission{
		Decision: NativeLaunchDecisionCorrectnessPath, ReasonCode: reason, Reason: detail, Operation: operation,
		ShapeElements: shapeElements, GridBlocks: domain.GridBlocks, BlockThreads: domain.BlockThreads,
		PeakWorkspaceBytes: domain.PeakWorkspaceBytes,
	}
}
