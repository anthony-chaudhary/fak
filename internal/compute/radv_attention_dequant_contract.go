package compute

import (
	"encoding/binary"
	"fmt"
)

// FlashAttnDequantShader identifies the GLSL/SPIR-V compute shader for single-pass KV dequantization.
const FlashAttnDequantShader = "flash_attn_dequant"

// FlashAttnDequantFormat constants represent format codes for push constants.
const (
	FlashAttnDequantFormatQ8_0 int32 = 0
	FlashAttnDequantFormatQ4_0 int32 = 1
	FlashAttnDequantFormatQ4_K int32 = 2
)

// FlashAttnDequantPushConstantBytes is the push constant buffer size in bytes (4 * 4B = 16B).
const FlashAttnDequantPushConstantBytes = 16

// FlashAttnDequantFormatCode maps QuantizedKVType to the integer format code used in push constants.
func FlashAttnDequantFormatCode(format QuantizedKVType) (int32, error) {
	switch format {
	case QuantizedKVQ8_0:
		return FlashAttnDequantFormatQ8_0, nil
	case QuantizedKVQ4_0:
		return FlashAttnDequantFormatQ4_0, nil
	case QuantizedKVQ4_K:
		return FlashAttnDequantFormatQ4_K, nil
	default:
		return -1, fmt.Errorf("compute: unsupported quantized KV format %q for dequant shader", format)
	}
}

// FlashAttnDequantFormatFromCode maps an integer format code back to QuantizedKVType.
func FlashAttnDequantFormatFromCode(code int32) (QuantizedKVType, error) {
	switch code {
	case FlashAttnDequantFormatQ8_0:
		return QuantizedKVQ8_0, nil
	case FlashAttnDequantFormatQ4_0:
		return QuantizedKVQ4_0, nil
	case FlashAttnDequantFormatQ4_K:
		return QuantizedKVQ4_K, nil
	default:
		return "", fmt.Errorf("compute: unknown dequant format code %d", code)
	}
}

// FlashAttnDequantPushConstants encapsulates the push constants for flash_attn_dequant.comp:
//   - nPos: sequence length / token count.
//   - nKV: number of key/value heads.
//   - headDim: dimension per head.
//   - format: 0 = Q8_0, 1 = Q4_0, 2 = Q4_K.
type FlashAttnDequantPushConstants struct {
	NPos    int32 `json:"n_pos"`
	NKV     int32 `json:"n_kv"`
	HeadDim int32 `json:"head_dim"`
	Format  int32 `json:"format"`
}

// NewFlashAttnDequantPushConstants constructs and validates push constants.
func NewFlashAttnDequantPushConstants(nPos, nKV, headDim int, format QuantizedKVType) (FlashAttnDequantPushConstants, error) {
	if nPos <= 0 || nKV <= 0 || headDim <= 0 {
		return FlashAttnDequantPushConstants{}, fmt.Errorf("compute: invalid dimensions (nPos=%d, nKV=%d, headDim=%d)", nPos, nKV, headDim)
	}
	fmtCode, err := FlashAttnDequantFormatCode(format)
	if err != nil {
		return FlashAttnDequantPushConstants{}, err
	}
	return FlashAttnDequantPushConstants{
		NPos:    int32(nPos),
		NKV:     int32(nKV),
		HeadDim: int32(headDim),
		Format:  fmtCode,
	}, nil
}

// Size returns the byte size of the encoded push constants (16 bytes).
func (pc FlashAttnDequantPushConstants) Size() int {
	return FlashAttnDequantPushConstantBytes
}

// Encode serializes the push constants into little-endian bytes.
func (pc FlashAttnDequantPushConstants) Encode() []byte {
	buf := make([]byte, FlashAttnDequantPushConstantBytes)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(pc.NPos))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(pc.NKV))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(pc.HeadDim))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(pc.Format))
	return buf
}

// Decode deserializes push constants from little-endian bytes.
func (pc *FlashAttnDequantPushConstants) Decode(buf []byte) error {
	if len(buf) < FlashAttnDequantPushConstantBytes {
		return fmt.Errorf("compute: push constant buffer too small (len=%d, want=%d)", len(buf), FlashAttnDequantPushConstantBytes)
	}
	pc.NPos = int32(binary.LittleEndian.Uint32(buf[0:4]))
	pc.NKV = int32(binary.LittleEndian.Uint32(buf[4:8]))
	pc.HeadDim = int32(binary.LittleEndian.Uint32(buf[8:12]))
	pc.Format = int32(binary.LittleEndian.Uint32(buf[12:16]))
	return pc.Validate()
}

// DecodeFlashAttnDequantPushConstants is a convenience constructor for decoding push constants.
func DecodeFlashAttnDequantPushConstants(buf []byte) (FlashAttnDequantPushConstants, error) {
	var pc FlashAttnDequantPushConstants
	if err := pc.Decode(buf); err != nil {
		return FlashAttnDequantPushConstants{}, err
	}
	return pc, nil
}

// Validate checks dimension positivity and format validity.
func (pc FlashAttnDequantPushConstants) Validate() error {
	if pc.NPos <= 0 || pc.NKV <= 0 || pc.HeadDim <= 0 {
		return fmt.Errorf("compute: invalid dimensions (nPos=%d, nKV=%d, headDim=%d)", pc.NPos, pc.NKV, pc.HeadDim)
	}
	if pc.Format < 0 || pc.Format > 2 {
		return fmt.Errorf("compute: invalid format code %d (must be 0, 1, or 2)", pc.Format)
	}
	return nil
}

// FlashAttnDequantPipelineDescriptor captures Vulkan compute pipeline configuration for the dequant shader.
type FlashAttnDequantPipelineDescriptor struct {
	ShaderName       string                  `json:"shader_name"`
	TargetArch       string                  `json:"target_arch"`
	ShaderStage      string                  `json:"shader_stage"`
	EntryPoint       string                  `json:"entry_point"`
	Format           QuantizedKVType         `json:"format"`
	FormatCode       int32                   `json:"format_code"`
	WorkgroupSize    uint32                  `json:"workgroup_size"`
	WaveSize         uint32                  `json:"wave_size"`
	PushConstantSize uint32                  `json:"push_constant_size"`
	Bindings         []RADVBindingDescriptor `json:"bindings"`
}

// NewFlashAttnDequantPipelineDescriptor creates and validates a pipeline descriptor for AMD RDNA 3.5 (gfx1151).
func NewFlashAttnDequantPipelineDescriptor(arch string, format QuantizedKVType) (*FlashAttnDequantPipelineDescriptor, error) {
	if arch == "" {
		arch = RADVTargetArchGfx1151
	}
	if !isStrixHaloArch(arch) {
		return nil, fmt.Errorf("compute: unsupported architecture %q for flash attn dequant (requires gfx1151 / Strix Halo)", arch)
	}
	fmtCode, err := FlashAttnDequantFormatCode(format)
	if err != nil {
		return nil, err
	}

	bindings := []RADVBindingDescriptor{
		{
			Binding:        0,
			DescriptorType: "storage_buffer",
			Access:         "readonly",
			Name:           "RawCodesBuffer",
			StageFlags:     RADVShaderStageCompute,
		},
		{
			Binding:        1,
			DescriptorType: "storage_buffer",
			Access:         "readonly",
			Name:           "ScaleBuffer",
			StageFlags:     RADVShaderStageCompute,
		},
		{
			Binding:        2,
			DescriptorType: "storage_buffer",
			Access:         "writeonly",
			Name:           "OutputScratchpad",
			StageFlags:     RADVShaderStageCompute,
		},
	}

	return &FlashAttnDequantPipelineDescriptor{
		ShaderName:       FlashAttnDequantShader,
		TargetArch:       arch,
		ShaderStage:      RADVShaderStageCompute,
		EntryPoint:       RADVDefaultEntryPoint,
		Format:           format,
		FormatCode:       fmtCode,
		WorkgroupSize:    256,
		WaveSize:         32, // RDNA 3.5 native Wave32
		PushConstantSize: FlashAttnDequantPushConstantBytes,
		Bindings:         bindings,
	}, nil
}

// FlashAttnDequantDispatchPlan describes the grid dispatch dimensions for executing flash_attn_dequant.comp.
type FlashAttnDequantDispatchPlan struct {
	WorkgroupSize   uint32                        `json:"workgroup_size"`
	GridX           uint32                        `json:"grid_x"`
	GridY           uint32                        `json:"grid_y"`
	GridZ           uint32                        `json:"grid_z"`
	TotalWorkgroups uint64                        `json:"total_workgroups"`
	TotalThreads    uint64                        `json:"total_threads"`
	TotalElements   uint64                        `json:"total_elements"`
	PushConstants   FlashAttnDequantPushConstants `json:"push_constants"`
}

// PlanFlashAttnDequantDispatch calculates workgroup dispatch grid dimensions for dequantizing [nKV, nPos, headDim].
func PlanFlashAttnDequantDispatch(nPos, nKV, headDim int, format QuantizedKVType) (FlashAttnDequantDispatchPlan, error) {
	pc, err := NewFlashAttnDequantPushConstants(nPos, nKV, headDim, format)
	if err != nil {
		return FlashAttnDequantDispatchPlan{}, err
	}
	const workgroupSize = 256
	totalElements := uint64(nPos) * uint64(nKV) * uint64(headDim)
	gridX := uint32((totalElements + workgroupSize - 1) / workgroupSize)

	return FlashAttnDequantDispatchPlan{
		WorkgroupSize:   workgroupSize,
		GridX:           gridX,
		GridY:           1,
		GridZ:           1,
		TotalWorkgroups: uint64(gridX),
		TotalThreads:    uint64(gridX) * workgroupSize,
		TotalElements:   totalElements,
		PushConstants:   pc,
	}, nil
}

// FlashAttnDequantPushConstants returns the push constants matching this scratchpad's configuration.
func (s *FlashAttnDequantScratchpad) FlashAttnDequantPushConstants() (FlashAttnDequantPushConstants, error) {
	return NewFlashAttnDequantPushConstants(s.NumPos, s.NumKVHeads, s.HeadDim, s.Format)
}

// ShaderPipelineDescriptor returns the Vulkan pipeline descriptor for this scratchpad.
func (s *FlashAttnDequantScratchpad) ShaderPipelineDescriptor() (*FlashAttnDequantPipelineDescriptor, error) {
	return NewFlashAttnDequantPipelineDescriptor(s.Arch, s.Format)
}

// PlanDispatch returns the dispatch plan for this scratchpad.
func (s *FlashAttnDequantScratchpad) PlanDispatch() (FlashAttnDequantDispatchPlan, error) {
	return PlanFlashAttnDequantDispatch(s.NumPos, s.NumKVHeads, s.HeadDim, s.Format)
}
