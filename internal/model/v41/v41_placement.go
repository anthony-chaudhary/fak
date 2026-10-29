package v41

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	V41Placement1017Revision     = "1bae0e1088993b9b82de08f59de5be5a34e42170"
	V41Placement1017ConfigSHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
	V41Placement1017IndexSHA256  = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
	// Official pinned index metadata and bounded safetensors headers. File bytes
	// are disk demand; tensor bytes are encoded payload, NEVER runtime residency.
	v41Placement1017TensorBytes int64 = 510286023000
	v41Placement1017VisionBytes int64 = 970506240
	v41Placement1017MTPBytes    int64 = 7932874632
	v41Placement1017ExpertBytes int64 = 288777830400
	v41Placement1017EngramBytes int64 = 202758032400
	// Reference promotion of head BF16->F32 and forty wo_a FP8->BF16 tensors.
	// These deltas are a declared representation policy, not a device witness.
	v41Placement1017HeadExpansion int64 = 1323827200
	v41Placement1017WOExpansion   int64 = 40 * 33554432
)

var ErrV41PlacementUnfit = errors.New("V4.1 placement exceeds declared capacity")

// DeviceBytes is usable device capacity AFTER operator reserves. HostBytes is
// ONLY the encoded cache budget AFTER OS, CPU decoder/output and host/pinned
// transfer-buffer reserves, not total physical RAM. WorkspaceBytes charges only
// DEVICE activations/staging/allocator peaks and expansion beyond the named
// head/wo_a promotions. No runtime fallback is admitted here.
type V41PlacementTarget struct {
	CheckpointRevision                string
	IndexSHA256                       string // caller must hash the actual pinned index; CLI does so
	GPUs                              int
	DeviceBytes, HostBytes, NVMeBytes int64
	ContextTokens                     int
	WorkspaceBytes                    int64
}

type V41PlacementRow struct {
	Class string `json:"class"`
	Tier  string `json:"tier,omitempty"`
	Bytes int64  `json:"bytes"`
}

type V41Placement struct {
	Schema                string             `json:"schema"`
	CheckpointRevision    string             `json:"checkpoint_revision"`
	CheckpointFileBytes   int64              `json:"checkpoint_file_bytes"`
	CheckpointTensorBytes int64              `json:"checkpoint_tensor_bytes"`
	TextTensorBytes       int64              `json:"text_tensor_bytes"`
	Classes               []V41PlacementRow  `json:"classes"`
	Excluded              []V41PlacementRow  `json:"excluded_text_only"`
	Device                compute.MemoryPlan `json:"device"`
	Host                  compute.MemoryPlan `json:"host"`
	Tiers                 []V41PlacementRow  `json:"encoded_weight_tiers"`
	StreamedBytes         int64              `json:"streamed_bytes"`
	SpillLayers           int                `json:"spill_layers"`
	EffectiveKVPrecision  string             `json:"effective_kv_precision"`
	Limitations           []string           `json:"limitations"`
}

// PlanDeepSeekV41Placement implements only the one-GPU, text-only 1017 leaf.
// Config identity is byte-exact before parsing or sizing, preventing arbitrary
// dimensions/compression values from overflowing the shared context sizer.
// No loader, numerical-profile, host-stage or hardware acceptance is implied.
func PlanDeepSeekV41Placement(configJSON []byte, t V41PlacementTarget) (V41Placement, error) {
	p := V41Placement{Schema: "fak.v41-placement/v1alpha1", CheckpointRevision: V41Placement1017Revision, CheckpointFileBytes: DeepSeekV41CheckpointBytes, CheckpointTensorBytes: v41Placement1017TensorBytes, EffectiveKVPrecision: "f32",
		Limitations: []string{"offline declared-capacity plan, not runtime-fit, loader or numerical admission", "1017 text only: vision+aligner and all three MTP/DSpark groups excluded; no Q2 or multi-GPU", "FP8 dense/shared and packed MXFP4 experts retained except explicit reference head/wo_a promotions", "workspace is DEVICE-only: additional casts, device staging and allocator peaks; hardware UNRUN", "host capacity is encoded CACHE budget after OS, CPU decoder/output and host/pinned-transfer reserves, not physical RAM", "host/SSD expert and Engram execution remains a separate acceptance debt"}}
	if len(configJSON) > 1<<20 {
		return p, fmt.Errorf("V4.1 placement: config exceeds metadata bound")
	}
	digest := sha256.Sum256(configJSON)
	if t.CheckpointRevision != V41Placement1017Revision || t.IndexSHA256 != V41Placement1017IndexSHA256 || hex.EncodeToString(digest[:]) != V41Placement1017ConfigSHA256 {
		return p, fmt.Errorf("V4.1 placement: exact1017 revision/config/index identity required")
	}
	if t.GPUs != 1 || t.DeviceBytes <= 0 || t.HostBytes <= 0 || t.NVMeBytes <= 0 || t.WorkspaceBytes <= 0 || t.ContextTokens <= 0 || t.ContextTokens > 1048576 {
		return p, fmt.Errorf("V4.1 placement: require one GPU, positive declared capacities/workspace and context in [1,1048576]")
	}
	if t.DeviceBytes > 1<<60 || t.HostBytes > 1<<60 || t.NVMeBytes > 1<<60 || t.WorkspaceBytes > 1<<60 {
		return p, fmt.Errorf("V4.1 placement: capacity exceeds supported integer range")
	}
	var c model.Config
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return p, err
	}
	if !c.IsDeepSeekV41() {
		return p, fmt.Errorf("V4.1 placement: config decoder did not retain V4.1 architecture")
	}
	p.TextTensorBytes = v41Placement1017TensorBytes - v41Placement1017VisionBytes - v41Placement1017MTPBytes
	dense := p.TextTensorBytes - v41Placement1017ExpertBytes - v41Placement1017EngramBytes
	p.Classes = []V41PlacementRow{{Class: "routed_experts_packed_mxfp4_with_scales", Bytes: v41Placement1017ExpertBytes}, {Class: "engram_fp8_rows_with_scales", Bytes: v41Placement1017EngramBytes}, {Class: "other_text_encoded", Bytes: dense}}
	p.Excluded = []V41PlacementRow{{Class: "vision_and_aligner", Bytes: v41Placement1017VisionBytes}, {Class: "mtp_0_1_2", Bytes: v41Placement1017MTPBytes}}
	p.Device = c.ContextSizeConfigWithPrecision(compute.KVPrecisionF32).PerContextMemoryPlan(t.ContextTokens)
	var contextBytes int64
	for i := range p.Device {
		d := &p.Device[i]
		if d.Bytes < 0 || d.Bytes > (1<<60)-contextBytes {
			return p, fmt.Errorf("V4.1 placement: invalid context charge")
		}
		contextBytes += d.Bytes
		d.Scope = compute.MemoryScopeDevice
	}
	if contextBytes <= 0 {
		return p, fmt.Errorf("V4.1 placement: missing context charge")
	}
	expansion := v41Placement1017HeadExpansion + v41Placement1017WOExpansion
	base := dense + expansion + contextBytes + t.WorkspaceBytes
	spill, err := model.AutoFitExpertSpill(model.ExpertSpillBudget{MoELayers: 40, ExpertBytesPerLayer: v41Placement1017ExpertBytes / 40, DeviceBaseBytes: base, BudgetBytes: t.DeviceBytes})
	if err != nil {
		return p, err
	}
	p.SpillLayers = spill.SpillLayers
	p.Device = append(p.Device, compute.MemoryDemand{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: dense, Detail: "1017-other-text-encoded"}, compute.MemoryDemand{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: expansion, Detail: "1017-reference-head-f32-and-wo-a-bf16-extra"}, compute.MemoryDemand{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: v41Placement1017ExpertBytes - spill.HostSpillBytes, Detail: "1017-packed-resident-experts"}, compute.MemoryDemand{Class: compute.MemoryScratchpad, Scope: compute.MemoryScopeDevice, Bytes: t.WorkspaceBytes, Detail: "1017-explicit-workspace-staging-and-additional-casts"})
	if err := compute.RefuseMemoryPlanIfTooBigForReportedDevice(nil, p.Device, t.DeviceBytes, t.DeviceBytes, true, 0); err != nil {
		return p, fmt.Errorf("%w: %w", ErrV41PlacementUnfit, err)
	}
	hostRemaining := t.HostBytes
	for _, r := range []V41PlacementRow{{Class: "engram", Bytes: v41Placement1017EngramBytes}, {Class: "routed_experts", Bytes: spill.HostSpillBytes}} {
		cached := r.Bytes
		if cached > hostRemaining {
			cached = hostRemaining
		}
		hostRemaining -= cached
		if cached > 0 {
			p.Host = append(p.Host, compute.MemoryDemand{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: cached, Detail: "1017-" + r.Class})
			p.Tiers = append(p.Tiers, V41PlacementRow{Class: r.Class, Tier: model.ExpertCacheTierL2.String(), Bytes: cached})
		}
		if streamed := r.Bytes - cached; streamed > 0 {
			p.StreamedBytes += streamed
			p.Tiers = append(p.Tiers, V41PlacementRow{Class: r.Class, Tier: model.ExpertCacheTierNVMe.String(), Bytes: streamed})
		}
	}
	p.Tiers = append(p.Tiers, V41PlacementRow{Class: "other_text_and_resident_experts", Tier: model.ExpertCacheTierVRAM.String(), Bytes: dense + v41Placement1017ExpertBytes - spill.HostSpillBytes})
	if err := compute.RefuseMemoryPlanIfTooBigForReportedHost(p.Host, t.HostBytes, t.HostBytes, true, 0); err != nil {
		return p, fmt.Errorf("%w: %w", ErrV41PlacementUnfit, err)
	}
	if t.NVMeBytes < DeepSeekV41CheckpointBytes {
		return p, fmt.Errorf("%w: disk requires all %d source shard bytes, not cache misses alone", ErrV41PlacementUnfit, DeepSeekV41CheckpointBytes)
	}
	return p, nil
}
