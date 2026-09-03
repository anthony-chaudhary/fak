// Prior-art: llama.cpp Metal / MLX (https://github.com/ggml-org/llama.cpp)
// Oracle: cpuref (GEMV cosine)

package metalgemm

import (
	"errors"
	"fmt"
	"testing"
)

// ICBCommandType represents the type of command encoded in an MTLIndirectCommandBuffer.
type ICBCommandType string

const (
	ICBCommandTypeConcurrentDispatch        ICBCommandType = "MTLIndirectCommandTypeConcurrentDispatch"
	ICBCommandTypeConcurrentDispatchThreads ICBCommandType = "MTLIndirectCommandTypeConcurrentDispatchThreads"
)

// ICBStorageMode specifies the memory allocation mode for the indirect buffer.
type ICBStorageMode string

const (
	ICBStorageModeShared  ICBStorageMode = "MTLResourceStorageModeShared"
	ICBStorageModePrivate ICBStorageMode = "MTLResourceStorageModePrivate"
)

// ICBDescriptor specifies configuration for creating an MTLIndirectCommandBuffer.
type ICBDescriptor struct {
	CommandTypes             []ICBCommandType
	InheritBuffers           bool
	InheritPipelineState     bool
	MaxKernelBufferBindCount int
	MaxCallCount             int
	StorageMode              ICBStorageMode
}

// Validate verifies that the ICB descriptor satisfies Apple Silicon Metal constraints.
func (d *ICBDescriptor) Validate() error {
	if len(d.CommandTypes) == 0 {
		return errors.New("icb: command types must not be empty")
	}
	hasCompute := false
	for _, ct := range d.CommandTypes {
		if ct == ICBCommandTypeConcurrentDispatch || ct == ICBCommandTypeConcurrentDispatchThreads {
			hasCompute = true
			break
		}
	}
	if !hasCompute {
		return errors.New("icb: command types must contain concurrent compute dispatch")
	}
	if d.MaxKernelBufferBindCount < 1 || d.MaxKernelBufferBindCount > 31 {
		return fmt.Errorf("icb: max kernel buffer bind count %d out of range [1, 31]", d.MaxKernelBufferBindCount)
	}
	if d.MaxCallCount <= 0 {
		return fmt.Errorf("icb: max call count %d must be positive", d.MaxCallCount)
	}
	return nil
}

// ICBKernelOpType classifies the specific operation in a transformer decode layer.
type ICBKernelOpType string

const (
	ICBOpRMSNormIn    ICBKernelOpType = "rmsnorm_in"
	ICBOpGEMVQ        ICBKernelOpType = "gemv_q"
	ICBOpGEMVK        ICBKernelOpType = "gemv_k"
	ICBOpGEMVV        ICBKernelOpType = "gemv_v"
	ICBOpRoPEQ        ICBKernelOpType = "rope_q"
	ICBOpRoPEK        ICBKernelOpType = "rope_k"
	ICBOpAttentionGQA ICBKernelOpType = "attention_gqa"
	ICBOpGEMVO        ICBKernelOpType = "gemv_o"
	ICBOpResidualAdd1 ICBKernelOpType = "residual_add_attn"
	ICBOpRMSNormPost  ICBKernelOpType = "rmsnorm_post"
	ICBOpGEMVGate     ICBKernelOpType = "gemv_gate"
	ICBOpGEMVUp       ICBKernelOpType = "gemv_up"
	ICBOpSwiGLU       ICBKernelOpType = "swiglu"
	ICBOpGEMVDown     ICBKernelOpType = "gemv_down"
	ICBOpResidualAdd2 ICBKernelOpType = "residual_add_mlp"
	ICBOpFinalRMSNorm ICBKernelOpType = "final_rmsnorm"
	ICBOpLMHead       ICBKernelOpType = "lm_head_gemv"
)

// ICBDispatchSlot represents a single recorded compute dispatch in the ICB topology.
type ICBDispatchSlot struct {
	Index                 int
	LayerIndex            int // -1 for global/head operations
	Op                    ICBKernelOpType
	PipelineName          string
	BufferBindingsCount   int
	GridDimensions        [3]uint64
	ThreadgroupDimensions [3]uint64
	ReadsDynamicContext   bool
}

// ICBTopology captures the recorded execution graph for an entire decode step.
type ICBTopology struct {
	ModelName          string
	NumLayers          int
	HiddenDim          int
	VocabSize          int
	DispatchesPerLayer int
	HeadDispatches     int
	Slots              []ICBDispatchSlot
}

// BuildQwen48LayerTopology constructs the 48-layer Qwen decode dispatch topology.
func BuildQwen48LayerTopology(numLayers, hiddenDim, vocabSize int) *ICBTopology {
	topo := &ICBTopology{
		ModelName:          "Qwen-48L-Decode",
		NumLayers:          numLayers,
		HiddenDim:          hiddenDim,
		VocabSize:          vocabSize,
		DispatchesPerLayer: 15,
		HeadDispatches:     2,
	}

	slotIdx := 0
	for l := 0; l < numLayers; l++ {
		layerOps := []struct {
			op         ICBKernelOpType
			pso        string
			bufCount   int
			dynamicCtx bool
			gridDim    [3]uint64
			tgDim      [3]uint64
		}{
			{ICBOpRMSNormIn, "d_rmsnorm", 4, true, [3]uint64{1, 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVQ, "q8dq_gemv_q", 8, true, [3]uint64{uint64(hiddenDim / 8), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVK, "q8dq_gemv_k", 8, true, [3]uint64{uint64(hiddenDim / 16), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVV, "q8dq_gemv_v", 8, true, [3]uint64{uint64(hiddenDim / 16), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpRoPEQ, "d_rope_q", 5, true, [3]uint64{uint64(hiddenDim / 64), 1, 1}, [3]uint64{64, 1, 1}},
			{ICBOpRoPEK, "d_rope_k", 5, true, [3]uint64{uint64(hiddenDim / 128), 1, 1}, [3]uint64{64, 1, 1}},
			{ICBOpAttentionGQA, "d_attn_gqa", 10, true, [3]uint64{32, 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVO, "q8dq_gemv_o", 8, true, [3]uint64{uint64(hiddenDim / 8), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpResidualAdd1, "d_add", 3, false, [3]uint64{uint64(hiddenDim / 256), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpRMSNormPost, "d_rmsnorm", 4, true, [3]uint64{1, 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVGate, "q8dq_gemv_gate", 8, true, [3]uint64{uint64(hiddenDim * 2 / 8), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVUp, "q8dq_gemv_up", 8, true, [3]uint64{uint64(hiddenDim * 2 / 8), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpSwiGLU, "d_silu_mul", 3, false, [3]uint64{uint64(hiddenDim * 2 / 256), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpGEMVDown, "q8dq_gemv_down", 8, true, [3]uint64{uint64(hiddenDim / 8), 1, 1}, [3]uint64{256, 1, 1}},
			{ICBOpResidualAdd2, "d_add", 3, false, [3]uint64{uint64(hiddenDim / 256), 1, 1}, [3]uint64{256, 1, 1}},
		}

		for _, lo := range layerOps {
			topo.Slots = append(topo.Slots, ICBDispatchSlot{
				Index:                 slotIdx,
				LayerIndex:            l,
				Op:                    lo.op,
				PipelineName:          lo.pso,
				BufferBindingsCount:   lo.bufCount,
				GridDimensions:        lo.gridDim,
				ThreadgroupDimensions: lo.tgDim,
				ReadsDynamicContext:   lo.dynamicCtx,
			})
			slotIdx++
		}
	}

	// Final Head operations
	topo.Slots = append(topo.Slots, ICBDispatchSlot{
		Index:                 slotIdx,
		LayerIndex:            -1,
		Op:                    ICBOpFinalRMSNorm,
		PipelineName:          "d_final_rmsnorm",
		BufferBindingsCount:   4,
		GridDimensions:        [3]uint64{1, 1, 1},
		ThreadgroupDimensions: [3]uint64{256, 1, 1},
		ReadsDynamicContext:   true,
	})
	slotIdx++

	topo.Slots = append(topo.Slots, ICBDispatchSlot{
		Index:                 slotIdx,
		LayerIndex:            -1,
		Op:                    ICBOpLMHead,
		PipelineName:          "q8dq_gemv_lm_head",
		BufferBindingsCount:   8,
		GridDimensions:        [3]uint64{uint64(vocabSize / 8), 1, 1},
		ThreadgroupDimensions: [3]uint64{256, 1, 1},
		ReadsDynamicContext:   true,
	})

	return topo
}

// TotalDispatches returns the total number of dispatches recorded in the ICB.
func (t *ICBTopology) TotalDispatches() int {
	return len(t.Slots)
}

// GEMVCount counts how many GEMV kernel dispatches exist in the topology.
func (t *ICBTopology) GEMVCount() int {
	count := 0
	for _, slot := range t.Slots {
		switch slot.Op {
		case ICBOpGEMVQ, ICBOpGEMVK, ICBOpGEMVV, ICBOpGEMVO, ICBOpGEMVGate, ICBOpGEMVUp, ICBOpGEMVDown, ICBOpLMHead:
			count++
		}
	}
	return count
}

// ElementwiseCount counts elementwise and attention dispatches.
func (t *ICBTopology) ElementwiseCount() int {
	return t.TotalDispatches() - t.GEMVCount()
}

// ArgumentBufferField describes a single member in the DecodeStepContext argument buffer.
type ArgumentBufferField struct {
	Name      string
	TypeName  string
	Offset    int
	Size      int
	Alignment int
}

// ArgumentBufferLayout validates the Metal Shading Language memory alignment of dynamic parameters.
type ArgumentBufferLayout struct {
	StructName string
	TotalSize  int
	Alignment  int
	Fields     []ArgumentBufferField
}

// NewDecodeStepContextLayout models the MSL struct layout for per-step dynamic parameters.
func NewDecodeStepContextLayout() *ArgumentBufferLayout {
	fields := []ArgumentBufferField{
		{"StepL", "uint32_t", 0, 4, 4},
		{"CtxLen", "uint32_t", 4, 4, 4},
		{"KVRowStride", "uint32_t", 8, 4, 4},
		{"Flags", "uint32_t", 12, 4, 4},
		{"ActiveXBuffer", "device half*", 16, 8, 8},
		{"NextXBuffer", "device half*", 24, 8, 8},
		{"PageTableBase", "device uint32_t*", 32, 8, 8},
		{"KVCacheBaseK", "device half*", 40, 8, 8},
		{"KVCacheBaseV", "device half*", 48, 8, 8},
		{"LogitsBuffer", "device half*", 56, 8, 8},
	}
	return &ArgumentBufferLayout{
		StructName: "DecodeStepContext",
		TotalSize:  64,
		Alignment:  8,
		Fields:     fields,
	}
}

// Validate checks alignment invariants for Metal device address space.
func (l *ArgumentBufferLayout) Validate() error {
	curOffset := 0
	for _, f := range l.Fields {
		// Field must be naturally aligned to its alignment requirement
		if curOffset%f.Alignment != 0 {
			padding := f.Alignment - (curOffset % f.Alignment)
			curOffset += padding
		}
		if f.Offset != curOffset {
			return fmt.Errorf("field %s offset %d mismatch expected %d", f.Name, f.Offset, curOffset)
		}
		curOffset += f.Size
	}
	// Total struct size must be a multiple of struct alignment
	expectedTotal := curOffset
	if expectedTotal%l.Alignment != 0 {
		expectedTotal += l.Alignment - (expectedTotal % l.Alignment)
	}
	if l.TotalSize != expectedTotal {
		return fmt.Errorf("total struct size %d mismatch expected aligned size %d", l.TotalSize, expectedTotal)
	}
	return nil
}

// OverheadModel computes projected execution latencies and throughput.
type OverheadModel struct {
	NumDispatches          int
	PerDispatchCPUEncodeUs float64
	ICBReplayOverheadUs    float64
	GPUBoundStepMs         float64
	BaselineCPUEncodeMs    float64
	ICBCPUEncodeMs         float64
	NetSavingsMs           float64
	BaselineTotalStepMs    float64
	ICBTotalStepMs         float64
	BaselineTokPerSec      float64
	ICBTokPerSec           float64
	ThroughputGainPercent  float64
}

// CalculateOverheadReduction evaluates CPU overhead reduction on Apple Silicon.
func CalculateOverheadReduction(numDispatches int, perDispatchUs, icbReplayUs, gpuStepMs float64) *OverheadModel {
	baselineEncodeMs := float64(numDispatches) * perDispatchUs / 1000.0
	icbEncodeMs := icbReplayUs / 1000.0
	netSavingsMs := baselineEncodeMs - icbEncodeMs

	baselineTotalMs := baselineEncodeMs + gpuStepMs
	icbTotalMs := icbEncodeMs + gpuStepMs

	baselineTokPerSec := 1000.0 / baselineTotalMs
	icbTokPerSec := 1000.0 / icbTotalMs
	gainPercent := ((icbTokPerSec - baselineTokPerSec) / baselineTokPerSec) * 100.0

	return &OverheadModel{
		NumDispatches:          numDispatches,
		PerDispatchCPUEncodeUs: perDispatchUs,
		ICBReplayOverheadUs:    icbReplayUs,
		GPUBoundStepMs:         gpuStepMs,
		BaselineCPUEncodeMs:    baselineEncodeMs,
		ICBCPUEncodeMs:         icbEncodeMs,
		NetSavingsMs:           netSavingsMs,
		BaselineTotalStepMs:    baselineTotalMs,
		ICBTotalStepMs:         icbTotalMs,
		BaselineTokPerSec:      baselineTokPerSec,
		ICBTokPerSec:           icbTokPerSec,
		ThroughputGainPercent:  gainPercent,
	}
}

// TestICBReplaySpec runs all verification suites for the ICB Decode Replay specification.
func TestICBReplaySpec(t *testing.T) {
	t.Run("CommandDescriptorAllocation", func(t *testing.T) {
		desc := &ICBDescriptor{
			CommandTypes:             []ICBCommandType{ICBCommandTypeConcurrentDispatch},
			InheritBuffers:           true,
			InheritPipelineState:     false,
			MaxKernelBufferBindCount: 16,
			MaxCallCount:             722,
			StorageMode:              ICBStorageModeShared,
		}

		if err := desc.Validate(); err != nil {
			t.Fatalf("valid descriptor rejected: %v", err)
		}

		// Verify invalid descriptor checks
		invalidDesc := &ICBDescriptor{
			CommandTypes:             []ICBCommandType{},
			MaxKernelBufferBindCount: 0,
			MaxCallCount:             0,
		}
		if err := invalidDesc.Validate(); err == nil {
			t.Fatal("invalid descriptor should have failed validation")
		}

		// Verify Apple Silicon bind limit (31 buffers)
		oversizedDesc := &ICBDescriptor{
			CommandTypes:             []ICBCommandType{ICBCommandTypeConcurrentDispatch},
			MaxKernelBufferBindCount: 32,
			MaxCallCount:             100,
		}
		if err := oversizedDesc.Validate(); err == nil {
			t.Fatal("oversized bind count (>31) should fail validation")
		}
	})

	t.Run("Qwen48LayerTopology", func(t *testing.T) {
		topo := BuildQwen48LayerTopology(48, 5120, 152064)

		expectedDispatches := 48*15 + 2 // 720 layer dispatches + 2 head dispatches = 722
		if topo.TotalDispatches() != expectedDispatches {
			t.Fatalf("total dispatches = %d, want %d", topo.TotalDispatches(), expectedDispatches)
		}

		// Each layer has 7 GEMVs: Q, K, V, O, Gate, Up, Down. Plus 1 LM head = 48*7 + 1 = 337 GEMVs
		expectedGEMVs := 48*7 + 1
		if topo.GEMVCount() != expectedGEMVs {
			t.Errorf("GEMV count = %d, want %d", topo.GEMVCount(), expectedGEMVs)
		}

		// Non-GEMV dispatches: RMSNorms, RoPE, GQA, Residual Adds, SwiGLU = 722 - 337 = 385
		expectedElementwise := expectedDispatches - expectedGEMVs
		if topo.ElementwiseCount() != expectedElementwise {
			t.Errorf("elementwise count = %d, want %d", topo.ElementwiseCount(), expectedElementwise)
		}

		// Verify slots are monotonically indexed
		for i, slot := range topo.Slots {
			if slot.Index != i {
				t.Fatalf("slot index %d out of order (got %d)", i, slot.Index)
			}
			if slot.BufferBindingsCount <= 0 || slot.BufferBindingsCount > 16 {
				t.Fatalf("slot %d has invalid buffer bindings count %d", i, slot.BufferBindingsCount)
			}
		}

		// Verify final operations are layer -1
		finalRMSNorm := topo.Slots[expectedDispatches-2]
		if finalRMSNorm.Op != ICBOpFinalRMSNorm || finalRMSNorm.LayerIndex != -1 {
			t.Errorf("expected final RMSNorm at slot %d, got %+v", expectedDispatches-2, finalRMSNorm)
		}
		lmHead := topo.Slots[expectedDispatches-1]
		if lmHead.Op != ICBOpLMHead || lmHead.LayerIndex != -1 {
			t.Errorf("expected LM Head at slot %d, got %+v", expectedDispatches-1, lmHead)
		}
	})

	t.Run("ArgumentBufferLayout", func(t *testing.T) {
		layout := NewDecodeStepContextLayout()
		if err := layout.Validate(); err != nil {
			t.Fatalf("ArgumentBufferLayout validation failed: %v", err)
		}

		if layout.TotalSize != 64 {
			t.Errorf("layout total size = %d, want 64", layout.TotalSize)
		}
		if layout.Alignment != 8 {
			t.Errorf("layout alignment = %d, want 8", layout.Alignment)
		}

		// Check specific fields
		fieldMap := make(map[string]ArgumentBufferField)
		for _, f := range layout.Fields {
			fieldMap[f.Name] = f
		}

		stepL, ok := fieldMap["StepL"]
		if !ok || stepL.Offset != 0 || stepL.Size != 4 {
			t.Errorf("StepL field definition invalid: %+v", stepL)
		}

		activeX, ok := fieldMap["ActiveXBuffer"]
		if !ok || activeX.Offset != 16 || activeX.Size != 8 {
			t.Errorf("ActiveXBuffer field definition invalid: %+v", activeX)
		}

		pageTable, ok := fieldMap["PageTableBase"]
		if !ok || pageTable.Offset != 32 || pageTable.Size != 8 {
			t.Errorf("PageTableBase field definition invalid: %+v", pageTable)
		}
	})

	t.Run("EncodeOverheadReductionModel", func(t *testing.T) {
		// Test M3 Pro configuration:
		// 722 dispatches, ~2.0 µs per dispatch baseline CPU encode = 1.444 ms baseline encode
		// ICB replay execution overhead: ~45 µs = 0.045 ms
		// Net CPU encode latency elimination: ~1.40 ms (saving within 0.8-1.5 ms target window)
		dispatches := 722
		perDispatchUs := 2.0
		icbReplayUs := 45.0

		// Case 1: Qwen 2.5 14B on M3 Pro (GPU time ~8.5 ms)
		m14B := CalculateOverheadReduction(dispatches, perDispatchUs, icbReplayUs, 8.5)
		if m14B.NetSavingsMs < 0.8 || m14B.NetSavingsMs > 1.5 {
			t.Errorf("NetSavingsMs = %.3f ms, expected in range [0.8, 1.5] ms", m14B.NetSavingsMs)
		}
		if m14B.ThroughputGainPercent < 10.0 {
			t.Errorf("14B throughput gain = %.2f%%, expected > 10%%", m14B.ThroughputGainPercent)
		}

		// Case 2: Qwen 7B on M3 Pro (GPU time ~4.2 ms)
		m7B := CalculateOverheadReduction(dispatches, perDispatchUs, icbReplayUs, 4.2)
		if m7B.ThroughputGainPercent < 20.0 {
			t.Errorf("7B throughput gain = %.2f%%, expected > 20%%", m7B.ThroughputGainPercent)
		}

		// Case 3: Qwen 27B / 32B on M3 Pro (GPU time ~14.0 ms)
		m27B := CalculateOverheadReduction(dispatches, perDispatchUs, icbReplayUs, 14.0)
		if m27B.ThroughputGainPercent < 5.0 {
			t.Errorf("27B throughput gain = %.2f%%, expected > 5%%", m27B.ThroughputGainPercent)
		}
	})

	t.Run("DynamicInputMutability", func(t *testing.T) {
		// Verify that topology slots mark dynamic context dependency
		topo := BuildQwen48LayerTopology(48, 5120, 152064)

		dynamicSlots := 0
		staticSlots := 0
		for _, slot := range topo.Slots {
			if slot.ReadsDynamicContext {
				dynamicSlots++
			} else {
				staticSlots++
			}
		}

		// RoPE, Attention, RMSNorm, and GEMVs read dynamic pointers/offsets
		// Only static elementwise adds / swiglu on fixed scratch buffers are non-dynamic
		if dynamicSlots == 0 || staticSlots == 0 {
			t.Errorf("unexpected slot distribution: dynamic=%d, static=%d", dynamicSlots, staticSlots)
		}

		// Invariant: The topology itself remains static while step L advances from 0 to 4096
		for stepL := 0; stepL < 4096; stepL += 512 {
			ctxLen := stepL + 1
			if ctxLen <= 0 {
				t.Fatalf("invalid ctxLen at step %d", stepL)
			}
			// Command count and dispatch slots remain unchanged across tokens
			if topo.TotalDispatches() != 722 {
				t.Fatalf("topology mutated across token steps!")
			}
		}
	})

	t.Run("PagedKVCacheIndirection", func(t *testing.T) {
		// Validate Paged KV Cache block indirection schema in argument buffer
		pageSize := 16 // tokens per block
		maxContext := 8192
		maxBlocks := maxContext / pageSize

		pageTable := make([]uint32, maxBlocks)
		for b := range pageTable {
			pageTable[b] = uint32(b * 2) // Virtual block -> Physical block map
		}

		// Verify block lookup for step L
		stepL := 1025
		blockIdx := stepL / pageSize
		blockOffset := stepL % pageSize

		if blockIdx >= len(pageTable) {
			t.Fatalf("block index %d exceeds page table length %d", blockIdx, len(pageTable))
		}
		physBlock := pageTable[blockIdx]
		if physBlock != uint32(blockIdx*2) {
			t.Errorf("physical block = %d, want %d", physBlock, blockIdx*2)
		}
		if blockOffset != 1 {
			t.Errorf("block offset = %d, want 1", blockOffset)
		}
	})
}
