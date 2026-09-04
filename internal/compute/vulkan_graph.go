package compute

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
)

// vulkan_graph.go implements reusable layer command graph structures, bounded staging pools,
// and AMD RDNA3 Wave32 rowtile optimization for Vulkan compute backends.
//
// This addresses three connected performance issues:
//   - #9834: "perf(vulkan): replace Qwen3.8 one-shot compute with reusable layer command graphs"
//   - #9835: "perf(vulkan): overlap Qwen3.8 layer transfer and compute with bounded staging"
//   - #9677: "perf(qwen38): re-derive gfx1151 rowtile and graph-reuse levers in fak-native AMD path"
//
// Like vulkan_plan.go and vulkan_transient_plan.go, this file is pure Go, always compiled,
// and unit-tested on any host without requiring a physical GPU.

// ---- Vulkan Pipeline Stages, Access Flags & Device Synchronization -------------

// VulkanPipelineStageFlags mirrors VkPipelineStageFlagBits for device-side synchronization.
type VulkanPipelineStageFlags uint32

const (
	VulkanStageTopOfPipe     VulkanPipelineStageFlags = 1 << 0
	VulkanStageDrawIndirect  VulkanPipelineStageFlags = 1 << 1
	VulkanStageComputeShader VulkanPipelineStageFlags = 1 << 5
	VulkanStageTransfer      VulkanPipelineStageFlags = 1 << 12
	VulkanStageBottomOfPipe  VulkanPipelineStageFlags = 1 << 13
	VulkanStageHost          VulkanPipelineStageFlags = 1 << 14
)

// VulkanAccessFlags mirrors VkAccessFlagBits for memory barriers between dispatches.
type VulkanAccessFlags uint32

const (
	VulkanAccessIndirectCommandRead VulkanAccessFlags = 1 << 0
	VulkanAccessUniformRead         VulkanAccessFlags = 1 << 1
	VulkanAccessShaderRead          VulkanAccessFlags = 1 << 5
	VulkanAccessShaderWrite         VulkanAccessFlags = 1 << 6
	VulkanAccessTransferRead        VulkanAccessFlags = 1 << 11
	VulkanAccessTransferWrite       VulkanAccessFlags = 1 << 12
	VulkanAccessHostRead            VulkanAccessFlags = 1 << 13
	VulkanAccessHostWrite           VulkanAccessFlags = 1 << 14
	VulkanAccessMemoryRead          VulkanAccessFlags = 1 << 15
	VulkanAccessMemoryWrite         VulkanAccessFlags = 1 << 16
)

// VulkanMemoryBarrier expresses device-side memory dependencies between graph nodes
// without requiring host fence synchronization.
type VulkanMemoryBarrier struct {
	SrcStageMask  VulkanPipelineStageFlags `json:"src_stage_mask"`
	DstStageMask  VulkanPipelineStageFlags `json:"dst_stage_mask"`
	SrcAccessMask VulkanAccessFlags        `json:"src_access_mask"`
	DstAccessMask VulkanAccessFlags        `json:"dst_access_mask"`
	BufferID      int64                    `json:"buffer_id,omitempty"`
	Offset        int64                    `json:"offset,omitempty"`
	Size          int64                    `json:"size,omitempty"`
}

// ---- Reusable Layer Command Graph (#9834) ---------------------------------------

// VulkanGraphOpType categorizes the forward operation executed by a graph node.
type VulkanGraphOpType string

const (
	VulkanOpMatMul    VulkanGraphOpType = "matmul"
	VulkanOpRMSNorm   VulkanGraphOpType = "rmsnorm"
	VulkanOpRoPE      VulkanGraphOpType = "rope"
	VulkanOpAttention VulkanGraphOpType = "attention"
	VulkanOpSwiGLU    VulkanGraphOpType = "swiglu"
	VulkanOpAdd       VulkanGraphOpType = "add"
	VulkanOpTransfer  VulkanGraphOpType = "transfer"
	VulkanOpBarrier   VulkanGraphOpType = "barrier"
)

// VulkanGraphState tracks the lifecycle of a command graph.
type VulkanGraphState string

const (
	VulkanGraphUnrecorded VulkanGraphState = "unrecorded"
	VulkanGraphRecording  VulkanGraphState = "recording"
	VulkanGraphRecorded   VulkanGraphState = "recorded"
	VulkanGraphExecuting  VulkanGraphState = "executing"
	VulkanGraphClosed     VulkanGraphState = "closed"
	VulkanGraphDeviceLost VulkanGraphState = "device_lost"
)

// VulkanGraphNode represents one operation inside a reusable command graph.
type VulkanGraphNode struct {
	ID           int                   `json:"id"`
	OpType       VulkanGraphOpType     `json:"op_type"`
	Name         string                `json:"name"`
	Inputs       []int64               `json:"inputs"`
	Outputs      []int64               `json:"outputs"`
	Predecessors []int                 `json:"predecessors"`
	Params       map[string]float64    `json:"params,omitempty"`
	Barriers     []VulkanMemoryBarrier `json:"barriers,omitempty"`
}

// VulkanCommandGraph records a directed sequence of layer forward operations into a
// single reusable command buffer structure. Instead of issuing N individual queue submissions
// with host fences per layer, the graph expresses dependencies via device-side barriers and
// submits once per layer execution.
type VulkanCommandGraph struct {
	mu               sync.Mutex
	id               string
	layerID          int
	state            VulkanGraphState
	nodes            []VulkanGraphNode
	replays          int
	submissionsSaved int
	deviceSyncEvents int
}

// NewVulkanCommandGraph constructs an unrecorded command graph for a layer.
func NewVulkanCommandGraph(id string, layerID int) *VulkanCommandGraph {
	return &VulkanCommandGraph{
		id:      id,
		layerID: layerID,
		state:   VulkanGraphUnrecorded,
	}
}

// ID returns the graph's identifier.
func (g *VulkanCommandGraph) ID() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.id
}

// LayerID returns the associated transformer layer index.
func (g *VulkanCommandGraph) LayerID() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.layerID
}

// State returns the current graph lifecycle state.
func (g *VulkanCommandGraph) State() VulkanGraphState {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.state
}

// Nodes returns a copy of the recorded nodes.
func (g *VulkanCommandGraph) Nodes() []VulkanGraphNode {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]VulkanGraphNode, len(g.nodes))
	copy(out, g.nodes)
	return out
}

// ReplayCount returns how many times this graph has been replayed.
func (g *VulkanCommandGraph) ReplayCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.replays
}

// SubmissionsSaved returns the cumulative number of queue submissions eliminated by graph reuse.
func (g *VulkanCommandGraph) SubmissionsSaved() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.submissionsSaved
}

// DeviceSyncEvents returns the count of device-side synchronization barriers recorded.
func (g *VulkanCommandGraph) DeviceSyncEvents() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.deviceSyncEvents
}

// BeginRecording opens the graph for recording nodes.
func (g *VulkanCommandGraph) BeginRecording() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.state == VulkanGraphDeviceLost {
		return ErrVulkanDeviceLost
	}
	if g.state == VulkanGraphRecording {
		return fmt.Errorf("compute: graph %s is already recording", g.id)
	}

	g.nodes = nil
	g.state = VulkanGraphRecording
	g.deviceSyncEvents = 0
	return nil
}

// AddNode appends an operation node to the recording graph.
func (g *VulkanCommandGraph) AddNode(node VulkanGraphNode) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.state == VulkanGraphDeviceLost {
		return ErrVulkanDeviceLost
	}
	if g.state != VulkanGraphRecording {
		return fmt.Errorf("compute: cannot add node to graph %s in state %s", g.id, g.state)
	}
	if node.ID < 0 {
		return fmt.Errorf("%w: invalid node id %d", ErrVulkanInvalidGeometry, node.ID)
	}
	if node.OpType == "" {
		return fmt.Errorf("%w: node %d missing op type", ErrVulkanInvalidGeometry, node.ID)
	}

	// Verify predecessor references
	existing := make(map[int]bool, len(g.nodes))
	for _, n := range g.nodes {
		existing[n.ID] = true
	}
	if existing[node.ID] {
		return fmt.Errorf("%w: duplicate node id %d", ErrVulkanInvalidGeometry, node.ID)
	}
	for _, p := range node.Predecessors {
		if p == node.ID {
			return fmt.Errorf("%w: self-referencing predecessor on node %d", ErrVulkanInvalidGeometry, node.ID)
		}
		if !existing[p] {
			return fmt.Errorf("%w: predecessor node %d not found for node %d", ErrVulkanInvalidGeometry, p, node.ID)
		}
	}

	g.deviceSyncEvents += len(node.Barriers)
	g.nodes = append(g.nodes, node)
	return nil
}

// AddRMSNormNode records an RMS normalization operation.
func (g *VulkanCommandGraph) AddRMSNormNode(id int, name string, inBuf, weightBuf, outBuf int64, rows, cols int, eps float32, preds []int) error {
	var barriers []VulkanMemoryBarrier
	if len(preds) > 0 {
		barriers = append(barriers, VulkanMemoryBarrier{
			SrcStageMask:  VulkanStageComputeShader,
			DstStageMask:  VulkanStageComputeShader,
			SrcAccessMask: VulkanAccessShaderWrite,
			DstAccessMask: VulkanAccessShaderRead,
			BufferID:      inBuf,
		})
	}
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpRMSNorm,
		Name:         name,
		Inputs:       []int64{inBuf, weightBuf},
		Outputs:      []int64{outBuf},
		Predecessors: preds,
		Params: map[string]float64{
			"rows": float64(rows),
			"cols": float64(cols),
			"eps":  float64(eps),
		},
		Barriers: barriers,
	})
}

// AddMatMulNode records a matrix multiplication operation (e.g. QKV projection, FFN gate/up/down).
func (g *VulkanCommandGraph) AddMatMulNode(id int, name string, inBuf, weightBuf, outBuf int64, m, n, k int, preds []int) error {
	var barriers []VulkanMemoryBarrier
	if len(preds) > 0 {
		barriers = append(barriers, VulkanMemoryBarrier{
			SrcStageMask:  VulkanStageComputeShader,
			DstStageMask:  VulkanStageComputeShader,
			SrcAccessMask: VulkanAccessShaderWrite,
			DstAccessMask: VulkanAccessShaderRead,
			BufferID:      inBuf,
		})
	}
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpMatMul,
		Name:         name,
		Inputs:       []int64{inBuf, weightBuf},
		Outputs:      []int64{outBuf},
		Predecessors: preds,
		Params: map[string]float64{
			"m": float64(m),
			"n": float64(n),
			"k": float64(k),
		},
		Barriers: barriers,
	})
}

// AddRoPENode records a rotary position embedding dispatch.
func (g *VulkanCommandGraph) AddRoPENode(id int, name string, buf int64, pos, nHeads, headDim int, theta float64, preds []int) error {
	var barriers []VulkanMemoryBarrier
	if len(preds) > 0 {
		barriers = append(barriers, VulkanMemoryBarrier{
			SrcStageMask:  VulkanStageComputeShader,
			DstStageMask:  VulkanStageComputeShader,
			SrcAccessMask: VulkanAccessShaderWrite,
			DstAccessMask: VulkanAccessShaderRead | VulkanAccessShaderWrite,
			BufferID:      buf,
		})
	}
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpRoPE,
		Name:         name,
		Inputs:       []int64{buf},
		Outputs:      []int64{buf},
		Predecessors: preds,
		Params: map[string]float64{
			"pos":     float64(pos),
			"nHeads":  float64(nHeads),
			"headDim": float64(headDim),
			"theta":   theta,
		},
		Barriers: barriers,
	})
}

// AddAttentionNode records a fused scaled dot-product attention dispatch.
func (g *VulkanCommandGraph) AddAttentionNode(id int, name string, qBuf, kBuf, vBuf, outBuf int64, nPos, nH, nKV, hd int, scale float32, preds []int) error {
	var barriers []VulkanMemoryBarrier
	if len(preds) > 0 {
		barriers = append(barriers, VulkanMemoryBarrier{
			SrcStageMask:  VulkanStageComputeShader,
			DstStageMask:  VulkanStageComputeShader,
			SrcAccessMask: VulkanAccessShaderWrite,
			DstAccessMask: VulkanAccessShaderRead,
			BufferID:      qBuf,
		})
	}
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpAttention,
		Name:         name,
		Inputs:       []int64{qBuf, kBuf, vBuf},
		Outputs:      []int64{outBuf},
		Predecessors: preds,
		Params: map[string]float64{
			"nPos":  float64(nPos),
			"nH":    float64(nH),
			"nKV":   float64(nKV),
			"hd":    float64(hd),
			"scale": float64(scale),
		},
		Barriers: barriers,
	})
}

// AddSwiGLUNode records a SwiGLU activation dispatch (silu(gate) * up).
func (g *VulkanCommandGraph) AddSwiGLUNode(id int, name string, gateBuf, upBuf, outBuf int64, numel int, preds []int) error {
	var barriers []VulkanMemoryBarrier
	if len(preds) > 0 {
		barriers = append(barriers, VulkanMemoryBarrier{
			SrcStageMask:  VulkanStageComputeShader,
			DstStageMask:  VulkanStageComputeShader,
			SrcAccessMask: VulkanAccessShaderWrite,
			DstAccessMask: VulkanAccessShaderRead,
			BufferID:      gateBuf,
		})
	}
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpSwiGLU,
		Name:         name,
		Inputs:       []int64{gateBuf, upBuf},
		Outputs:      []int64{outBuf},
		Predecessors: preds,
		Params: map[string]float64{
			"numel": float64(numel),
		},
		Barriers: barriers,
	})
}

// AddAddInPlaceNode records a residual in-place add dispatch (dst += src).
func (g *VulkanCommandGraph) AddAddInPlaceNode(id int, name string, dstBuf, srcBuf int64, numel int, preds []int) error {
	var barriers []VulkanMemoryBarrier
	if len(preds) > 0 {
		barriers = append(barriers, VulkanMemoryBarrier{
			SrcStageMask:  VulkanStageComputeShader,
			DstStageMask:  VulkanStageComputeShader,
			SrcAccessMask: VulkanAccessShaderWrite,
			DstAccessMask: VulkanAccessShaderRead | VulkanAccessShaderWrite,
			BufferID:      dstBuf,
		})
	}
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpAdd,
		Name:         name,
		Inputs:       []int64{dstBuf, srcBuf},
		Outputs:      []int64{dstBuf},
		Predecessors: preds,
		Params: map[string]float64{
			"numel": float64(numel),
		},
		Barriers: barriers,
	})
}

// AddBarrierNode explicitly inserts an intra-graph pipeline barrier between stages.
func (g *VulkanCommandGraph) AddBarrierNode(id int, name string, barrier VulkanMemoryBarrier, preds []int) error {
	return g.AddNode(VulkanGraphNode{
		ID:           id,
		OpType:       VulkanOpBarrier,
		Name:         name,
		Predecessors: preds,
		Barriers:     []VulkanMemoryBarrier{barrier},
	})
}

// EndRecording finalizes the command graph, validates acyclicity, and marks it ready for replay.
func (g *VulkanCommandGraph) EndRecording() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.state == VulkanGraphDeviceLost {
		return ErrVulkanDeviceLost
	}
	if g.state != VulkanGraphRecording {
		return fmt.Errorf("compute: cannot end recording on graph %s in state %s", g.id, g.state)
	}
	if len(g.nodes) == 0 {
		return fmt.Errorf("%w: graph contains no nodes", ErrVulkanInvalidGeometry)
	}

	if err := g.validateLocked(); err != nil {
		return err
	}

	g.state = VulkanGraphRecorded
	return nil
}

// Validate checks node ID uniqueness, dependency topology, and cycle freedom.
func (g *VulkanCommandGraph) Validate() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.validateLocked()
}

func (g *VulkanCommandGraph) validateLocked() error {
	if len(g.nodes) == 0 {
		return fmt.Errorf("%w: empty graph", ErrVulkanInvalidGeometry)
	}

	nodeMap := make(map[int]VulkanGraphNode, len(g.nodes))
	for _, n := range g.nodes {
		if _, exists := nodeMap[n.ID]; exists {
			return fmt.Errorf("%w: duplicate node id %d", ErrVulkanInvalidGeometry, n.ID)
		}
		nodeMap[n.ID] = n
	}

	// Cycle detection using DFS state marking: 0=unvisited, 1=visiting, 2=visited
	visitState := make(map[int]int, len(g.nodes))
	var dfs func(int) error
	dfs = func(id int) error {
		st := visitState[id]
		if st == 1 {
			return fmt.Errorf("%w: cycle detected involving node %d", ErrVulkanInvalidGeometry, id)
		}
		if st == 2 {
			return nil
		}
		visitState[id] = 1
		for _, pred := range nodeMap[id].Predecessors {
			if _, exists := nodeMap[pred]; !exists {
				return fmt.Errorf("%w: missing predecessor node %d", ErrVulkanInvalidGeometry, pred)
			}
			if err := dfs(pred); err != nil {
				return err
			}
		}
		visitState[id] = 2
		return nil
	}

	for _, n := range g.nodes {
		if visitState[n.ID] == 0 {
			if err := dfs(n.ID); err != nil {
				return err
			}
		}
	}
	return nil
}

// Execute replays the recorded command graph in a single submission.
// It increments replay counters, amortizing command buffer construction and descriptor set updates.
func (g *VulkanCommandGraph) Execute() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.state == VulkanGraphDeviceLost {
		return ErrVulkanDeviceLost
	}
	if g.state != VulkanGraphRecorded {
		return fmt.Errorf("%w: graph %s in state %s not recorded", ErrVulkanExecutionFailed, g.id, g.state)
	}

	g.state = VulkanGraphExecuting
	// Single submission executes all nodes with device-side barriers
	saved := len(g.nodes) - 1
	if saved < 0 {
		saved = 0
	}
	g.submissionsSaved += saved
	g.replays++
	g.state = VulkanGraphRecorded
	return nil
}

// OnDeviceLost marks the graph as destroyed due to device loss or unrecoverable reset.
func (g *VulkanCommandGraph) OnDeviceLost() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.state = VulkanGraphDeviceLost
	g.nodes = nil
	return ErrVulkanDeviceLost
}

// Reset clears recorded nodes and returns the graph to unrecorded state.
func (g *VulkanCommandGraph) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.state == VulkanGraphDeviceLost {
		return
	}
	g.nodes = nil
	g.state = VulkanGraphUnrecorded
}

// Submissions returns the batched submission count for this graph (1 if recorded, 0 if empty).
func (g *VulkanCommandGraph) Submissions() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.nodes) == 0 {
		return 0
	}
	return 1
}

// NaiveSubmits returns the per-op one-shot submission count that naive execution would issue.
func (g *VulkanCommandGraph) NaiveSubmits() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.nodes)
}

// SubmissionReduction returns the count of submissions eliminated by graph recording.
func (g *VulkanCommandGraph) SubmissionReduction() int {
	return g.NaiveSubmits() - g.Submissions()
}

// ---- Qwen3.8 Standard Layer Command Graph Constructor ---------------------------

// Qwen38LayerGraphConfig defines the architectural dimensions for building a layer graph.
type Qwen38LayerGraphConfig struct {
	HiddenDim       int
	IntermediateDim int
	NumHeads        int
	NumKVHeads      int
	HeadDim         int
	Eps             float32
	Theta           float64
	BatchSize       int
	SeqLen          int
}

// BuildQwen38LayerGraph creates and records a standard Qwen3.8 transformer layer forward
// command graph with device-side barriers connecting:
//  1. Input layernorm (RMSNorm)
//  2. QKV projection (MatMul)
//  3. RoPE
//  4. Attention
//  5. Output projection (MatMul)
//  6. Residual add (hidden += attn_out)
//  7. Post-attention layernorm (RMSNorm)
//  8. FFN Gate/Up projection (MatMul)
//  9. SwiGLU activation
//  10. FFN Down projection (MatMul)
//  11. Residual add (hidden += ffn_out)
func BuildQwen38LayerGraph(layerID int, cfg Qwen38LayerGraphConfig) (*VulkanCommandGraph, error) {
	if cfg.HiddenDim <= 0 || cfg.IntermediateDim <= 0 || cfg.NumHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("%w: invalid dimensions for qwen38 layer", ErrVulkanInvalidGeometry)
	}

	graphID := fmt.Sprintf("qwen38_layer_%d", layerID)
	g := NewVulkanCommandGraph(graphID, layerID)
	if err := g.BeginRecording(); err != nil {
		return nil, err
	}

	// Buffer ID conventions for graph recording:
	const (
		BufHiddenState = 100
		BufNorm1Weight = 101
		BufNorm1Out    = 102
		BufQKVWeight   = 103
		BufQKVOut      = 104
		BufKVStoreK    = 105
		BufKVStoreV    = 106
		BufAttnOut     = 107
		BufOutWeight   = 108
		BufProjOut     = 109
		BufNorm2Weight = 110
		BufNorm2Out    = 111
		BufGateUpW     = 112
		BufGateUpOut   = 113
		BufSwiGLUOut   = 114
		BufDownWeight  = 115
		BufFFNOut      = 116
	)

	m := cfg.BatchSize
	if m <= 0 {
		m = 1
	}

	// 1. Input RMSNorm
	if err := g.AddRMSNormNode(1, "input_layernorm", BufHiddenState, BufNorm1Weight, BufNorm1Out, m, cfg.HiddenDim, cfg.Eps, nil); err != nil {
		return nil, err
	}

	// 2. QKV MatMul
	qkvDim := (cfg.NumHeads + 2*cfg.NumKVHeads) * cfg.HeadDim
	if err := g.AddMatMulNode(2, "qkv_proj", BufNorm1Out, BufQKVWeight, BufQKVOut, m, qkvDim, cfg.HiddenDim, []int{1}); err != nil {
		return nil, err
	}

	// 3. RoPE
	if err := g.AddRoPENode(3, "rope", BufQKVOut, cfg.SeqLen, cfg.NumHeads, cfg.HeadDim, cfg.Theta, []int{2}); err != nil {
		return nil, err
	}

	// 4. Attention
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	if err := g.AddAttentionNode(4, "attention", BufQKVOut, BufKVStoreK, BufKVStoreV, BufAttnOut, cfg.SeqLen, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, scale, []int{3}); err != nil {
		return nil, err
	}

	// 5. Output Projection MatMul
	if err := g.AddMatMulNode(5, "o_proj", BufAttnOut, BufOutWeight, BufProjOut, m, cfg.HiddenDim, cfg.NumHeads*cfg.HeadDim, []int{4}); err != nil {
		return nil, err
	}

	// 6. Residual Add: HiddenState += ProjOut
	if err := g.AddAddInPlaceNode(6, "attn_residual", BufHiddenState, BufProjOut, m*cfg.HiddenDim, []int{5}); err != nil {
		return nil, err
	}

	// 7. Post-attention RMSNorm
	if err := g.AddRMSNormNode(7, "post_attention_layernorm", BufHiddenState, BufNorm2Weight, BufNorm2Out, m, cfg.HiddenDim, cfg.Eps, []int{6}); err != nil {
		return nil, err
	}

	// 8. FFN Gate/Up Projection MatMul
	if err := g.AddMatMulNode(8, "ffn_gate_up", BufNorm2Out, BufGateUpW, BufGateUpOut, m, 2*cfg.IntermediateDim, cfg.HiddenDim, []int{7}); err != nil {
		return nil, err
	}

	// 9. SwiGLU Activation
	if err := g.AddSwiGLUNode(9, "swiglu", BufGateUpOut, BufGateUpOut, BufSwiGLUOut, m*cfg.IntermediateDim, []int{8}); err != nil {
		return nil, err
	}

	// 10. FFN Down Projection MatMul
	if err := g.AddMatMulNode(10, "ffn_down", BufSwiGLUOut, BufDownWeight, BufFFNOut, m, cfg.HiddenDim, cfg.IntermediateDim, []int{9}); err != nil {
		return nil, err
	}

	// 11. Residual Add: HiddenState += FFNOut
	if err := g.AddAddInPlaceNode(11, "ffn_residual", BufHiddenState, BufFFNOut, m*cfg.HiddenDim, []int{10}); err != nil {
		return nil, err
	}

	if err := g.EndRecording(); err != nil {
		return nil, err
	}

	return g, nil
}

// ---- VulkanLayerGraphCache: Multi-Layer Graph Cache -----------------------------

// VulkanGraphCacheStats records hit/miss and submission savings metrics.
type VulkanGraphCacheStats struct {
	Hits             uint64 `json:"hits"`
	Misses           uint64 `json:"misses"`
	CachedLayers     int    `json:"cached_layers"`
	SubmissionsSaved uint64 `json:"submissions_saved"`
}

// VulkanLayerGraphCache provides thread-safe caching and lookup of reusable command graphs.
type VulkanLayerGraphCache struct {
	mu     sync.RWMutex
	graphs map[int]*VulkanCommandGraph
	stats  VulkanGraphCacheStats
	closed bool
}

// NewVulkanLayerGraphCache instantiates an empty command graph cache.
func NewVulkanLayerGraphCache() *VulkanLayerGraphCache {
	return &VulkanLayerGraphCache{
		graphs: make(map[int]*VulkanCommandGraph),
	}
}

// Get retrieves a cached graph for a layer, recording a hit or miss.
func (c *VulkanLayerGraphCache) Get(layer int) (*VulkanCommandGraph, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, false
	}
	g, ok := c.graphs[layer]
	if ok && g.State() == VulkanGraphRecorded {
		c.stats.Hits++
		return g, true
	}
	c.stats.Misses++
	return nil, false
}

// Put stores a recorded command graph in the cache.
func (c *VulkanLayerGraphCache) Put(layer int, g *VulkanCommandGraph) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrVulkanDeviceLost
	}
	if g == nil {
		return fmt.Errorf("%w: nil graph", ErrVulkanInvalidGeometry)
	}
	if g.State() != VulkanGraphRecorded {
		return fmt.Errorf("%w: graph must be in recorded state", ErrVulkanInvalidGeometry)
	}

	c.graphs[layer] = g
	c.stats.CachedLayers = len(c.graphs)
	return nil
}

// Invalidate removes a cached layer graph.
func (c *VulkanLayerGraphCache) Invalidate(layer int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.graphs, layer)
	c.stats.CachedLayers = len(c.graphs)
}

// Clear removes all cached graphs.
func (c *VulkanLayerGraphCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.graphs = make(map[int]*VulkanCommandGraph)
	c.stats.CachedLayers = 0
}

// OnDeviceLost invalidates all cached graphs and marks the cache closed.
func (c *VulkanLayerGraphCache) OnDeviceLost() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, g := range c.graphs {
		_ = g.OnDeviceLost()
	}
	c.graphs = make(map[int]*VulkanCommandGraph)
	c.stats.CachedLayers = 0
	c.closed = true
}

// Stats returns a snapshot of cache metrics.
func (c *VulkanLayerGraphCache) Stats() VulkanGraphCacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s := c.stats
	var totalSaved uint64
	for _, g := range c.graphs {
		totalSaved += uint64(g.SubmissionsSaved())
	}
	s.SubmissionsSaved = totalSaved
	return s
}

// ---- Bounded Staging Pool (#9835) -----------------------------------------------

// StagingSlotState reflects the readiness of a bounded staging slot.
type StagingSlotState string

const (
	StagingSlotFree         StagingSlotState = "free"
	StagingSlotTransferring StagingSlotState = "transferring"
	StagingSlotReady        StagingSlotState = "ready"
	StagingSlotComputing    StagingSlotState = "computing"
)

// VulkanStagingSlot is one reusable host-to-device buffer slot for overlapping transfer and compute.
type VulkanStagingSlot struct {
	ID          int              `json:"id"`
	State       StagingSlotState `json:"state"`
	Capacity    int64            `json:"capacity"`
	UsedBytes   int64            `json:"used_bytes"`
	Layer       int              `json:"layer"`
	TimelineVal uint64           `json:"timeline_val"`
}

// VulkanStagingPool manages a bounded set of staging slots (e.g. 2 for double-buffering)
// to overlap layer transfer with compute while strictly preventing overwrite races.
type VulkanStagingPool struct {
	mu          sync.Mutex
	cond        *sync.Cond
	numSlots    int
	slotSize    int64
	align       int64
	slots       []VulkanStagingSlot
	closed      bool
	deviceLost  bool
	totalStaged int64
	transfers   uint64
}

// NewVulkanStagingPool allocates a bounded staging pool with numSlots (>= 2 for double-buffering).
func NewVulkanStagingPool(numSlots int, slotCapacity int64, align int64) (*VulkanStagingPool, error) {
	if numSlots < 2 {
		return nil, fmt.Errorf("%w: staging pool requires at least 2 slots for overlap, got %d", ErrVulkanInvalidGeometry, numSlots)
	}
	if slotCapacity <= 0 {
		return nil, fmt.Errorf("%w: invalid slot capacity %d", ErrVulkanInvalidGeometry, slotCapacity)
	}
	if align < 1 {
		align = 1
	}

	p := &VulkanStagingPool{
		numSlots: numSlots,
		slotSize: slotCapacity,
		align:    align,
		slots:    make([]VulkanStagingSlot, numSlots),
	}
	p.cond = sync.NewCond(&p.mu)

	for i := 0; i < numSlots; i++ {
		p.slots[i] = VulkanStagingSlot{
			ID:       i,
			State:    StagingSlotFree,
			Capacity: slotCapacity,
			Layer:    -1,
		}
	}
	return p, nil
}

// NumSlots returns the configured slot count.
func (p *VulkanStagingPool) NumSlots() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.numSlots
}

// SlotCapacity returns the byte capacity of each slot.
func (p *VulkanStagingPool) SlotCapacity() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.slotSize
}

// AcquireSlot blocks until an idle slot is available and marks it StagingSlotTransferring.
func (p *VulkanStagingPool) AcquireSlot(layer int, bytes int64) (*VulkanStagingSlot, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.deviceLost {
		return nil, ErrVulkanDeviceLost
	}
	if p.closed {
		return nil, errors.New("compute: staging pool is closed")
	}
	if bytes <= 0 {
		return nil, fmt.Errorf("%w: transfer size must be positive (%d)", ErrVulkanInvalidGeometry, bytes)
	}
	if bytes > p.slotSize {
		return nil, fmt.Errorf("%w: requested %d bytes exceeds slot capacity %d", ErrVulkanResourceExhausted, bytes, p.slotSize)
	}

	for {
		if p.deviceLost {
			return nil, ErrVulkanDeviceLost
		}
		if p.closed {
			return nil, errors.New("compute: staging pool closed while waiting")
		}

		// Look for a free slot
		for i := 0; i < p.numSlots; i++ {
			if p.slots[i].State == StagingSlotFree {
				p.slots[i].State = StagingSlotTransferring
				p.slots[i].Layer = layer
				p.slots[i].UsedBytes = bytes
				p.totalStaged += bytes
				p.transfers++
				slotCopy := p.slots[i]
				return &slotCopy, nil
			}
		}

		p.cond.Wait()
	}
}

// TryAcquireSlot attempts to acquire a free slot without blocking.
func (p *VulkanStagingPool) TryAcquireSlot(layer int, bytes int64) (*VulkanStagingSlot, bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.deviceLost {
		return nil, false, ErrVulkanDeviceLost
	}
	if p.closed {
		return nil, false, errors.New("compute: staging pool is closed")
	}
	if bytes <= 0 {
		return nil, false, fmt.Errorf("%w: transfer size must be positive (%d)", ErrVulkanInvalidGeometry, bytes)
	}
	if bytes > p.slotSize {
		return nil, false, fmt.Errorf("%w: requested %d bytes exceeds slot capacity %d", ErrVulkanResourceExhausted, bytes, p.slotSize)
	}

	for i := 0; i < p.numSlots; i++ {
		if p.slots[i].State == StagingSlotFree {
			p.slots[i].State = StagingSlotTransferring
			p.slots[i].Layer = layer
			p.slots[i].UsedBytes = bytes
			p.totalStaged += bytes
			p.transfers++
			slotCopy := p.slots[i]
			return &slotCopy, true, nil
		}
	}
	return nil, false, nil
}

// MarkTransferComplete signals that host-to-device transfer has landed into the slot.
func (p *VulkanStagingPool) MarkTransferComplete(slotID int, timeline uint64) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.deviceLost {
		return ErrVulkanDeviceLost
	}
	if slotID < 0 || slotID >= p.numSlots {
		return fmt.Errorf("%w: invalid slot id %d", ErrVulkanInvalidGeometry, slotID)
	}
	slot := &p.slots[slotID]
	if slot.State != StagingSlotTransferring {
		return fmt.Errorf("compute: slot %d in state %s cannot transition to ready", slotID, slot.State)
	}

	slot.State = StagingSlotReady
	slot.TimelineVal = timeline
	return nil
}

// AcquireForCompute transitions a ready slot into compute ownership.
func (p *VulkanStagingPool) AcquireForCompute(slotID int) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.deviceLost {
		return ErrVulkanDeviceLost
	}
	if slotID < 0 || slotID >= p.numSlots {
		return fmt.Errorf("%w: invalid slot id %d", ErrVulkanInvalidGeometry, slotID)
	}
	slot := &p.slots[slotID]
	if slot.State != StagingSlotReady {
		return fmt.Errorf("compute: slot %d in state %s cannot transition to computing", slotID, slot.State)
	}

	slot.State = StagingSlotComputing
	return nil
}

// ReleaseCompute finishes compute on the slot and marks it free for reuse.
func (p *VulkanStagingPool) ReleaseCompute(slotID int) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.deviceLost {
		return ErrVulkanDeviceLost
	}
	if slotID < 0 || slotID >= p.numSlots {
		return fmt.Errorf("%w: invalid slot id %d", ErrVulkanInvalidGeometry, slotID)
	}
	slot := &p.slots[slotID]
	if slot.State != StagingSlotComputing {
		return fmt.Errorf("compute: slot %d in state %s cannot be released", slotID, slot.State)
	}

	slot.State = StagingSlotFree
	slot.UsedBytes = 0
	slot.Layer = -1
	p.cond.Signal()
	return nil
}

// ActiveCount returns the number of slots currently non-free.
func (p *VulkanStagingPool) ActiveCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	count := 0
	for _, s := range p.slots {
		if s.State != StagingSlotFree {
			count++
		}
	}
	return count
}

// TotalStaged returns the cumulative bytes staged through this pool.
func (p *VulkanStagingPool) TotalStaged() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.totalStaged
}

// Transfers returns the cumulative transfer count through this pool.
func (p *VulkanStagingPool) Transfers() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.transfers
}

// Slot returns a snapshot of the requested slot.
func (p *VulkanStagingPool) Slot(slotID int) (VulkanStagingSlot, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if slotID < 0 || slotID >= p.numSlots {
		return VulkanStagingSlot{}, fmt.Errorf("%w: invalid slot id %d", ErrVulkanInvalidGeometry, slotID)
	}
	return p.slots[slotID], nil
}

// Close releases the pool and unblocks waiting callers.
func (p *VulkanStagingPool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	for i := range p.slots {
		p.slots[i].State = StagingSlotFree
		p.slots[i].UsedBytes = 0
		p.slots[i].Layer = -1
	}
	p.cond.Broadcast()
}

// OnDeviceLost marks the staging pool as lost due to device reset.
func (p *VulkanStagingPool) OnDeviceLost() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.deviceLost = true
	p.closed = true
	for i := range p.slots {
		p.slots[i].State = StagingSlotFree
	}
	p.cond.Broadcast()
	return ErrVulkanDeviceLost
}

// OverlapSchedule evaluates pipelined makespan and speedup across stages using double buffering.
func (p *VulkanStagingPool) OverlapSchedule(stages []OverlapStage) OverlapResult {
	return AsyncOverlap(stages)
}

// ---- AMD RDNA3 Wave32 Rowtile Optimization (#9677) ------------------------------

// RowtileArch describes the execution parameters of an AMD GPU target.
type RowtileArch struct {
	GFX           string `json:"gfx"`
	Family        string `json:"family"`
	WavefrontSize int    `json:"wavefront_size"`
	ComputeUnits  int    `json:"compute_units"`
	LDSBytesPerCU int    `json:"lds_bytes_per_cu"`
	MaxVGPRs      int    `json:"max_vgprs"`
	BusWidthBits  int    `json:"bus_width_bits"`
	Description   string `json:"description"`
}

var knownRowtileArches = map[string]RowtileArch{
	"gfx1102": {
		GFX:           "gfx1102",
		Family:        "RDNA3",
		WavefrontSize: 32,
		ComputeUnits:  32,
		LDSBytesPerCU: 65536,
		MaxVGPRs:      512,
		BusWidthBits:  128,
		Description:   "AMD Radeon RX 7600 (RDNA3 discrete GPU)",
	},
	"gfx1151": {
		GFX:           "gfx1151",
		Family:        "RDNA3.5",
		WavefrontSize: 32,
		ComputeUnits:  40,
		LDSBytesPerCU: 65536,
		MaxVGPRs:      512,
		BusWidthBits:  256,
		Description:   "AMD Strix Halo / Ryzen AI Max+ 395 (RDNA3.5 APU)",
	},
	"gfx1100": {
		GFX:           "gfx1100",
		Family:        "RDNA3",
		WavefrontSize: 32,
		ComputeUnits:  84,
		LDSBytesPerCU: 65536,
		MaxVGPRs:      512,
		BusWidthBits:  384,
		Description:   "AMD Radeon RX 7900 XTX (RDNA3 discrete GPU)",
	},
}

// LookupRowtileArch returns the architectural parameters for a canonical or decorated GFX target.
func LookupRowtileArch(gfx string) (RowtileArch, bool) {
	norm := strings.ToLower(strings.TrimSpace(gfx))
	if idx := strings.IndexByte(norm, ':'); idx != -1 {
		norm = norm[:idx]
	}
	arch, ok := knownRowtileArches[norm]
	return arch, ok
}

// RowtileParams represents the derived tile dimensions and workgroup mapping.
type RowtileParams struct {
	GFX                string  `json:"gfx"`
	WavefrontSize      int     `json:"wavefront_size"`
	TileM              int     `json:"tile_m"`
	TileN              int     `json:"tile_n"`
	TileK              int     `json:"tile_k"`
	WaveM              int     `json:"wave_m"`
	WaveN              int     `json:"wave_n"`
	WaveK              int     `json:"wave_k"`
	WavesPerWorkgroup  int     `json:"waves_per_workgroup"`
	WorkgroupThreads   int     `json:"workgroup_threads"`
	LDSBytes           int     `json:"lds_bytes"`
	QuantFormat        string  `json:"quant_format"`
	PackedPairEncoding string  `json:"packed_pair_encoding"`
	RegistersPerThread int     `json:"registers_per_thread"`
	EstimatedOccupancy float64 `json:"estimated_occupancy"`
}

// CalculateRowtileParams calculates wavefront-matched rowtile parameters for AMD RDNA3/RDNA3.5
// GPUs (gfx1102 / gfx1151). It maps 32-lane Wave32 wavefronts directly to 32-element quantization
// blocks (e.g. Q8_0, Q4_K sub-blocks), selecting workgroup tile sizes that balance LDS usage and
// VGPR register pressure.
func CalculateRowtileParams(gfx string, m, n, k int, quantFormat string) (RowtileParams, error) {
	arch, ok := LookupRowtileArch(gfx)
	if !ok {
		return RowtileParams{}, fmt.Errorf("%w: unsupported gfx target %q for rowtile optimization", ErrVulkanInvalidGeometry, gfx)
	}
	if arch.WavefrontSize != 32 {
		return RowtileParams{}, fmt.Errorf("%w: target %s wavefront size %d != 32 (expected Wave32)", ErrVulkanInvalidGeometry, gfx, arch.WavefrontSize)
	}
	if m <= 0 || n <= 0 || k <= 0 {
		return RowtileParams{}, fmt.Errorf("%w: dimensions m=%d, n=%d, k=%d must be positive", ErrVulkanInvalidGeometry, m, n, k)
	}

	normFmt := strings.ToUpper(strings.TrimSpace(quantFormat))
	blockSize := 32
	elementBytes := 1
	var packedEncoding string

	switch normFmt {
	case "Q8_0":
		blockSize = 32
		elementBytes = 1 // 1 byte per code + amortized scale
		packedEncoding = "symmetric-int8"
	case "Q4_K":
		blockSize = 32   // 32-element sub-block
		elementBytes = 1 // 4-bit pairs packed 2 per byte
		packedEncoding = "unequal-packed-pair"
	case "Q5_K":
		blockSize = 32
		elementBytes = 1
		packedEncoding = "unequal-packed-pair"
	case "Q6_K":
		blockSize = 32
		elementBytes = 1
		packedEncoding = "unequal-packed-pair"
	case "FP16":
		blockSize = 16
		elementBytes = 2
		packedEncoding = "packed-half2"
	case "BF16":
		blockSize = 16
		elementBytes = 2
		packedEncoding = "packed-bf16_2"
	case "FP32":
		blockSize = 8
		elementBytes = 4
		packedEncoding = "float32-direct"
	default:
		return RowtileParams{}, fmt.Errorf("%w: unsupported quant format %q", ErrVulkanInvalidGeometry, quantFormat)
	}

	if k%blockSize != 0 {
		return RowtileParams{}, fmt.Errorf("%w: k dimension %d must be divisible by quant block size %d", ErrVulkanInvalidGeometry, k, blockSize)
	}

	var tileM, tileN, tileK int
	var waveM, waveN, waveK int
	var wavesPerWg int

	// Workgroup tiling tuned for discrete GPUs (gfx1102 / gfx1100) vs Strix Halo APU (gfx1151)
	if arch.GFX == "gfx1151" || arch.GFX == "gfx1100" {
		// gfx1151 (40 CUs, 256-bit bus) and gfx1100 (84 CUs, 384-bit bus): wider tiles maximize cache reuse
		if m <= 16 { // decode token phase
			tileM = 1
			tileN = 64
			tileK = 32
			waveM = 1
			waveN = 32
			waveK = 32
			wavesPerWg = 2 // 64 threads
		} else { // prefill / batched phase
			tileM = 64
			tileN = 64
			tileK = 32
			waveM = 32
			waveN = 32
			waveK = 32
			wavesPerWg = 4 // 128 threads
		}
	} else {
		// gfx1102 has 32 CUs and 128-bit memory bus: conservative tile sizes to keep register pressure low
		if m <= 16 {
			tileM = 1
			tileN = 64
			tileK = 32
			waveM = 1
			waveN = 32
			waveK = 32
			wavesPerWg = 2 // 64 threads
		} else {
			tileM = 32
			tileN = 64
			tileK = 32
			waveM = 32
			waveN = 32
			waveK = 32
			wavesPerWg = 2 // 64 threads
		}
	}

	workgroupThreads := wavesPerWg * arch.WavefrontSize

	// Double-buffered LDS tile storage for A and B panels
	ldsA := tileM * tileK * elementBytes
	ldsB := tileN * tileK * elementBytes
	ldsBytes := (ldsA + ldsB) * 2 // double buffered
	if ldsBytes > arch.LDSBytesPerCU {
		return RowtileParams{}, fmt.Errorf("%w: required LDS %d exceeds CU capacity %d", ErrVulkanResourceExhausted, ldsBytes, arch.LDSBytesPerCU)
	}

	// Register budget estimation (typical Wave32 GEMM kernel: ~32-64 VGPRs per thread)
	regsPerThread := 48
	if tileM > 32 {
		regsPerThread = 64
	}

	// Occupancy calculation: limited by max waves per SIMD / LDS allocation
	maxWgByLDS := arch.LDSBytesPerCU / (ldsBytes + 256) // safety padding
	if maxWgByLDS > 8 {
		maxWgByLDS = 8
	}
	estimatedOccupancy := float64(maxWgByLDS*wavesPerWg) / 32.0
	if estimatedOccupancy > 1.0 {
		estimatedOccupancy = 1.0
	}

	return RowtileParams{
		GFX:                arch.GFX,
		WavefrontSize:      arch.WavefrontSize,
		TileM:              tileM,
		TileN:              tileN,
		TileK:              tileK,
		WaveM:              waveM,
		WaveN:              waveN,
		WaveK:              waveK,
		WavesPerWorkgroup:  wavesPerWg,
		WorkgroupThreads:   workgroupThreads,
		LDSBytes:           ldsBytes,
		QuantFormat:        normFmt,
		PackedPairEncoding: packedEncoding,
		RegistersPerThread: regsPerThread,
		EstimatedOccupancy: estimatedOccupancy,
	}, nil
}

// RowtileGrid computes the 3D workgroup dispatch grid dimensions [GridX, GridY, GridZ]
// needed to cover an [M, N] output matrix under the chosen RowtileParams.
func RowtileGrid(params RowtileParams, m, n int) [3]int {
	if params.TileN <= 0 || params.TileM <= 0 {
		return [3]int{1, 1, 1}
	}
	gridX := (n + params.TileN - 1) / params.TileN
	gridY := (m + params.TileM - 1) / params.TileM
	if gridX < 1 {
		gridX = 1
	}
	if gridY < 1 {
		gridY = 1
	}
	return [3]int{gridX, gridY, 1}
}
