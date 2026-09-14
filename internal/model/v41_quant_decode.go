package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41FP8BlockDim and v41KVLoraRank are shared checkpoint constants declared in
// v41_inventory.go (the packed-tensor inventory landed via #12892); this decode
// path reuses them so the two V4.1 files cannot drift.

// v41DenseFP8Role is a V4.1 dense FP8 weight role and its published [O,I] shape.
type v41DenseFP8Role struct {
	weightName string
	shape      [2]int
}

// v41DenseFP8RoleForName maps a V4.1 dense tensor (or its .scale sibling) to the
// published [O,I] weight shape, derived from the admitted config geometry:
// hidden 5120, q_lora_rank 1280, o_lora_rank 1024 and o_groups 8 give
// wq_a [q_lora,hidden], wq_b [num_heads*head_dim,q_lora], wkv [v41KVLoraRank,hidden],
// wo_a [o_lora*o_groups,hidden], wo_b [hidden,o_lora*o_groups] and the shared
// expert w1/w3 [moe_intermediate,hidden] / w2 [hidden,moe_intermediate].
func v41DenseFP8RoleForName(name string, cfg Config) (v41DenseFP8Role, bool, error) {
	weightName := name
	if strings.HasSuffix(name, ".scale") {
		weightName = strings.TrimSuffix(name, ".scale") + ".weight"
	}
	rest, ok := strings.CutPrefix(weightName, "layers.")
	if !ok {
		return v41DenseFP8Role{}, false, nil
	}
	layerText, suffix, ok := strings.Cut(rest, ".")
	if !ok || !decimalOnly(layerText) {
		return v41DenseFP8Role{}, false, nil
	}
	layer, err := strconv.Atoi(layerText)
	if err != nil || layer < 0 || layer >= cfg.NumLayers {
		return v41DenseFP8Role{}, true, fmt.Errorf("safetensors: V4.1 dense FP8 tensor %s has layer %q outside [0,%d)", name, layerText, cfg.NumLayers)
	}
	oDim, ok := checkedShapeProduct(cfg.OLoraRank, cfg.OGroups)
	if !ok || oDim <= 0 {
		return v41DenseFP8Role{}, true, fmt.Errorf("safetensors: V4.1 dense FP8 tensor %s needs o_lora_rank*o_groups, got %d*%d", name, cfg.OLoraRank, cfg.OGroups)
	}
	qHeadDim, ok := checkedShapeProduct(cfg.NumHeads, cfg.HeadDim)
	if !ok || qHeadDim <= 0 {
		return v41DenseFP8Role{}, true, fmt.Errorf("safetensors: V4.1 dense FP8 tensor %s needs num_heads*head_dim, got %d*%d", name, cfg.NumHeads, cfg.HeadDim)
	}
	var shape [2]int
	switch suffix {
	case "attn.wq_a.weight":
		shape = [2]int{cfg.QLoraRank, cfg.HiddenSize}
	case "attn.wq_b.weight":
		shape = [2]int{qHeadDim, cfg.QLoraRank}
	case "attn.wkv.weight":
		shape = [2]int{v41KVLoraRank, cfg.HiddenSize}
	case "attn.wo_a.weight":
		shape = [2]int{oDim, cfg.HiddenSize}
	case "attn.wo_b.weight":
		shape = [2]int{cfg.HiddenSize, oDim}
	case "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight":
		shape = [2]int{cfg.MoEIntermediateSize, cfg.HiddenSize}
	case "ffn.shared_experts.w2.weight":
		shape = [2]int{cfg.HiddenSize, cfg.MoEIntermediateSize}
	default:
		return v41DenseFP8Role{}, false, nil
	}
	if shape[0] <= 0 || shape[1] <= 0 {
		return v41DenseFP8Role{}, true, fmt.Errorf("safetensors: V4.1 dense FP8 tensor %s resolved a non-positive shape %v", name, shape)
	}
	return v41DenseFP8Role{weightName: weightName, shape: shape}, true, nil
}

// quantizeV41DenseFP8TensorInto consumes the V4.1 dense FP8 pair (E4M3 weight +
// raw F8_E8M0 sibling scale, one scale per 32x32 tile). It mirrors
// quantizeV4DenseFP8TensorInto: the scale sorts first and validates/deferrs the
// pair, the weight visit decodes once and installs the Q8 resident tensor. It is
// fail-closed: a non-V4.1 config is left untouched, and an unknown role is not
// admitted to any generic dequant path.
func quantizeV41DenseFP8TensorInto(
	name string,
	hdr map[string]json.RawMessage,
	tensorBytes func(stEntry) ([]byte, error),
	m *Model,
	consumed map[string]bool,
) (bool, error) {
	if m == nil || !m.Cfg.IsDeepSeekV41() {
		return false, nil
	}
	role, ok, err := v41DenseFP8RoleForName(name, m.Cfg)
	if err != nil {
		return true, err
	}
	if !ok {
		return false, nil
	}
	weightName := role.weightName
	scaleName := strings.TrimSuffix(weightName, ".weight") + ".scale"
	weightEntry, scaleEntry, err := v41DenseFP8PairEntries(weightName, scaleName, role.shape, hdr)
	if err != nil {
		return true, err
	}
	if name == scaleName {
		return true, nil
	}

	weights, err := tensorBytes(weightEntry)
	if err != nil {
		return true, fmt.Errorf("safetensors: tensor %s: %w", weightName, err)
	}
	scales, err := tensorBytes(scaleEntry)
	if err != nil {
		return true, fmt.Errorf("safetensors: tensor %s: %w", scaleName, err)
	}
	decoded, err := decodeV41DenseFP8(weightName, weightEntry.Shape, weights, scales)
	if err != nil {
		return true, err
	}
	m.q8w[weightName] = quantizeQ8(decoded, weightEntry.Shape[0], weightEntry.Shape[1])
	consumed[weightName] = true
	consumed[scaleName] = true
	return true, nil
}

func v41DenseFP8PairEntries(weightName, scaleName string, wantWeight [2]int, hdr map[string]json.RawMessage) (stEntry, stEntry, error) {
	weightRaw, ok := hdr[weightName]
	if !ok {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense FP8 pair is missing %s", weightName)
	}
	scaleRaw, ok := hdr[scaleName]
	if !ok {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense FP8 pair %s is missing %s", weightName, scaleName)
	}
	var weightEntry, scaleEntry stEntry
	if err := json.Unmarshal(weightRaw, &weightEntry); err != nil {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: entry %s: %w", weightName, err)
	}
	if err := json.Unmarshal(scaleRaw, &scaleEntry); err != nil {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: entry %s: %w", scaleName, err)
	}
	if weightEntry.Dtype != "F8_E4M3" {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense weight %s dtype %q, want F8_E4M3", weightName, weightEntry.Dtype)
	}
	if scaleEntry.Dtype != "F8_E8M0" {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense scale %s dtype %q, want F8_E8M0", scaleName, scaleEntry.Dtype)
	}
	if len(weightEntry.Shape) != 2 {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense weight %s shape %v, want rank-2", weightName, weightEntry.Shape)
	}
	if !sameShape(weightEntry.Shape, wantWeight[:]) {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense weight %s shape %v, want published shape %v", weightName, weightEntry.Shape, wantWeight)
	}
	rows, cols := weightEntry.Shape[0], weightEntry.Shape[1]
	if rows <= 0 || cols <= 0 {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense weight %s shape %v, want positive dimensions", weightName, weightEntry.Shape)
	}
	wantScale := []int{(rows-1)/v41FP8BlockDim + 1, (cols-1)/v41FP8BlockDim + 1}
	if !sameShape(scaleEntry.Shape, wantScale) {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4.1 dense scale %s shape %v, want %v for weight shape %v", scaleName, scaleEntry.Shape, wantScale, weightEntry.Shape)
	}
	return weightEntry, scaleEntry, nil
}

// decodeV41DenseFP8 dequantizes a rank-2 [O,I] float8_e4m3fn weight stored with
// one raw F8_E8M0 scale byte per 32x32 tile (scale shape
// [ceil(O/32),ceil(I/32)]). It validates shape and length, rejects E4M3 NaN/Inf
// and the E8M0 0xff NaN byte, and refuses any non-finite product before
// returning. name is error context only.
func decodeV41DenseFP8(name string, shape []int, weights, scales []byte) ([]float32, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("safetensors: V4.1 dense weight %s shape %v, want rank-2", name, shape)
	}
	rows, cols := shape[0], shape[1]
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("safetensors: V4.1 dense weight %s has non-positive dimension %v", name, shape)
	}
	wantWeights, ok := checkedShapeProduct(rows, cols)
	if !ok || len(weights) != wantWeights {
		return nil, fmt.Errorf("safetensors: V4.1 dense weight %s has %d bytes, shape %v implies %d", name, len(weights), shape, wantWeights)
	}
	scaleRows := (rows-1)/v41FP8BlockDim + 1
	scaleCols := (cols-1)/v41FP8BlockDim + 1
	wantScales, ok := checkedShapeProduct(scaleRows, scaleCols)
	if !ok || len(scales) != wantScales {
		return nil, fmt.Errorf("safetensors: V4.1 dense scale for %s has %d bytes, want %d", name, len(scales), wantScales)
	}
	for i, value := range weights {
		if decoded := fp8E4M3ToF32(value); math.IsNaN(float64(decoded)) || math.IsInf(float64(decoded), 0) {
			return nil, fmt.Errorf("safetensors: V4.1 dense weight %s byte %d is E4M3 NaN", name, i)
		}
	}
	scaleValues := make([]float32, len(scales))
	for i, value := range scales {
		if value == 0xff {
			return nil, fmt.Errorf("safetensors: V4.1 dense scale for %s byte %d is E8M0 NaN", name, i)
		}
		scaleValues[i] = float32(math.Ldexp(1, int(value)-127))
		if math.IsInf(float64(scaleValues[i]), 0) {
			return nil, fmt.Errorf("safetensors: V4.1 dense scale for %s byte %d is non-finite", name, i)
		}
	}
	out := make([]float32, wantWeights)
	for o := 0; o < rows; o++ {
		scaleRow := (o / v41FP8BlockDim) * scaleCols
		for i := 0; i < cols; i++ {
			value := fp8E4M3ToF32(weights[o*cols+i]) * scaleValues[scaleRow+i/v41FP8BlockDim]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("safetensors: V4.1 dense weight %s decoded value [%d,%d] is non-finite", name, o, i)
			}
			out[o*cols+i] = value
		}
	}
	return out, nil
}

// v41ExpertQuantSpec is the packed [rows,cols] weight and [rows,cols] scale
// geometry of one V4.1 routed-expert projection.
type v41ExpertQuantSpec struct {
	weightRows int
	weightCols int
	scaleRows  int
	scaleCols  int
}

// v41ExpertQuantSpecs pins the V4.1 routed-expert MXFP4 geometry derived from
// MoE intermediate 2304 and hidden 5120: w1/w3 [2304,5120/2] and w2
// [5120,2304/2], each packed weight byte carrying two E2M1 nibbles and one
// E8M0 byte scaling 16 packed bytes / 32 unpacked K values.
var v41ExpertQuantSpecs = map[string]v41ExpertQuantSpec{
	"w1": {weightRows: 2304, weightCols: 2560, scaleRows: 2304, scaleCols: 160},
	"w2": {weightRows: 5120, weightCols: 1152, scaleRows: 5120, scaleCols: 72},
	"w3": {weightRows: 2304, weightCols: 2560, scaleRows: 2304, scaleCols: 160},
}

func selectV41ExpertQuantSpec(projection string, weightShape, scaleShape []int) (v41ExpertQuantSpec, bool) {
	spec, ok := v41ExpertQuantSpecs[projection]
	if !ok {
		return v41ExpertQuantSpec{}, false
	}
	if sameShape(weightShape, []int{spec.weightRows, spec.weightCols}) &&
		sameShape(scaleShape, []int{spec.scaleRows, spec.scaleCols}) {
		return spec, true
	}
	return v41ExpertQuantSpec{}, false
}

// decodeV41ExpertQuant is the scalar reference decoder for V4.1 routed experts.
// It mirrors decodeV4ExpertQuant: packed I8 bytes hold low-nibble val0 and
// high-nibble val1 of the OCP E2M1 table, and one F8_E8M0 byte scales each group
// of 32 unpacked K values. The destination is allocated only after every
// metadata, length and E8M0 scale byte validates.
func decodeV41ExpertQuant(weightName, scaleName string, weightEntry, scaleEntry stEntry, weights, scales []byte) ([]byte, []int, error) {
	stem, projection, err := parseV41ExpertQuantWeightName(weightName)
	if err != nil {
		return nil, nil, err
	}
	wantScaleName := stem + projection + ".scale"
	if scaleName != wantScaleName {
		return nil, nil, v4QuantMetadataf("V4.1 scale name %q, want %q", scaleName, wantScaleName)
	}
	if weightEntry.Dtype != "I8" {
		return nil, nil, v4QuantMetadataf("V4.1 %s dtype %q, want I8", weightName, weightEntry.Dtype)
	}
	if scaleEntry.Dtype != "F8_E8M0" {
		return nil, nil, v4QuantMetadataf("V4.1 %s dtype %q, want F8_E8M0", scaleName, scaleEntry.Dtype)
	}
	spec, ok := selectV41ExpertQuantSpec(projection, weightEntry.Shape, scaleEntry.Shape)
	if !ok {
		return nil, nil, v4QuantMetadataf("V4.1 %s shape %v with %s shape %v does not match a supported V4.1 profile", weightName, weightEntry.Shape, scaleName, scaleEntry.Shape)
	}
	if spec.weightCols != spec.scaleCols*16 {
		return nil, nil, v4QuantMetadataf("V4.1 %s scale ratio %d:%d, want 16 packed weight bytes per scale", weightName, spec.weightCols, spec.scaleCols)
	}
	weightBytes, ok := checkedShapeProduct(spec.weightRows, spec.weightCols)
	if !ok {
		return nil, nil, v4QuantMetadataf("V4.1 %s shape overflows byte count", weightName)
	}
	scaleBytes, ok := checkedShapeProduct(spec.scaleRows, spec.scaleCols)
	if !ok {
		return nil, nil, v4QuantMetadataf("V4.1 %s scale shape overflows byte count", scaleName)
	}
	if len(weights) != weightBytes {
		return nil, nil, v4QuantMetadataf("V4.1 %s has %d bytes, want %d", weightName, len(weights), weightBytes)
	}
	if len(scales) != scaleBytes {
		return nil, nil, v4QuantMetadataf("V4.1 %s has %d bytes, want %d", scaleName, len(scales), scaleBytes)
	}
	for i, scale := range scales {
		if scale == 0xff {
			return nil, nil, v4QuantMetadataf("V4.1 %s scale byte %d is F8_E8M0 NaN", scaleName, i)
		}
	}
	unpackedCols, ok := checkedShapeProduct(spec.weightCols, 2)
	if !ok {
		return nil, nil, v4QuantMetadataf("V4.1 %s unpacked shape overflows", weightName)
	}
	outElems, ok := checkedShapeProduct(spec.weightRows, unpackedCols)
	if !ok || outElems > int(^uint(0)>>1)/4 {
		return nil, nil, v4QuantMetadataf("V4.1 %s decoded byte count overflows", weightName)
	}
	out := make([]byte, outElems*4)
	for row := 0; row < spec.weightRows; row++ {
		packedRow := weights[row*spec.weightCols : (row+1)*spec.weightCols]
		scaleRow := scales[row*spec.scaleCols : (row+1)*spec.scaleCols]
		for packedCol, packed := range packedRow {
			exp := int(scaleRow[packedCol/16]) - 127
			base := (row*unpackedCols + packedCol*2) * 4
			lo := float32(math.Ldexp(float64(v4ExpertE2M1Values[packed&0x0f]), exp))
			hi := float32(math.Ldexp(float64(v4ExpertE2M1Values[packed>>4]), exp))
			binary.LittleEndian.PutUint32(out[base:], math.Float32bits(lo))
			binary.LittleEndian.PutUint32(out[base+4:], math.Float32bits(hi))
		}
	}
	return out, []int{spec.weightRows, unpackedCols}, nil
}

func parseV41ExpertQuantWeightName(name string) (stem, projection string, err error) {
	const marker = ".ffn.experts."
	if !strings.HasPrefix(name, "layers.") || !strings.Contains(name, marker) {
		return "", "", v4QuantMetadataf("V4.1 weight name %q is not a routed expert", name)
	}
	for projection := range v41ExpertQuantSpecs {
		suffix := projection + ".weight"
		if strings.HasSuffix(name, suffix) {
			stem := strings.TrimSuffix(name, suffix)
			identity := strings.TrimSuffix(stem, ".")
			parts := strings.Split(identity, ".")
			if len(parts) == 5 && parts[0] == "layers" && decimalOnly(parts[1]) && parts[2] == "ffn" && parts[3] == "experts" && decimalOnly(parts[4]) {
				return stem, projection, nil
			}
		}
	}
	return "", "", v4QuantMetadataf("V4.1 weight name %q has unsupported suffix or identity", name)
}

// ---- Routed-expert device-resident install + device matmul ---------------------
//
// The dense V4.1 path installs decoded FP8 into m.q8w. The routed-expert tier is
// different in two ways that matter to decode throughput:
//
//  1. The expert weight dominates the byte stream (top-6 of 384 experts x 3 matrices
//     x 5120 x 2304 x 40 layers), so its information-carrying representation must
//     stay PACKED. Expanding it to f32 (the scalar decodeV41ExpertQuant reference
//     above) is a full-checkpoint dequantization that would multiply resident bytes
//     ~4x and defeat the residency budget.
//  2. The matmul must run on the DEVICE. A scalar loop on the expert tier is exactly
//     the memory-bound roofline regression the keystone exists to remove.
//
// So this store holds the checkpoint's OWN packed bytes (E4M3 for FP8, I8-packed
// E2M1 for MXFP4) plus the raw E8M0 scale bytes, keyed by projection name, and the
// device matmul below hands them unchanged to compute's RoutedExpertDeviceKernel.
// It is a device RESIDENT install: installV41ExpertQuantDevice mirrors the dense
// `m.q8w[weightName] = ...` line but stores the packed pair rather than a Q8 tensor,
// and it never calls decodeV41ExpertQuant / decodeV41DenseFP8 (no f32 materialization).

// v41ExpertDeviceWeight is one routed-expert projection resident in its PACKED
// checkpoint form: weight bytes and raw E8M0 scale bytes, plus the geometry needed to
// size the device kernel. dtype is the checkpoint dtype string ("F8_E4M3" for the FP8
// block-32 path, "I8" for the MXFP4 nibble path), which selects the device kernel.
type v41ExpertDeviceWeight struct {
	projection string
	dtype      string
	O, I       int // unpacked output/input dims
	weight     []byte
	scales     []byte
}

// v41ExpertDeviceInstall is the per-Model device-expert store. It is kept in a
// package-level registry (rather than a Model field) so this leaf does not edit
// weights.go, and it is lazily created on first install. packedPreserved records that
// the install path never expanded weights to f32 — the invariant the witness pins.
type v41ExpertDeviceInstall struct {
	weights         map[string]*v41ExpertDeviceWeight
	packedPreserved bool
}

var (
	v41ExpertDeviceMu   sync.Mutex
	v41ExpertDeviceRegs = map[*Model]*v41ExpertDeviceInstall{}
)

func v41ExpertDeviceStore(m *Model, create bool) *v41ExpertDeviceInstall {
	v41ExpertDeviceMu.Lock()
	defer v41ExpertDeviceMu.Unlock()
	store := v41ExpertDeviceRegs[m]
	if store == nil && create {
		store = &v41ExpertDeviceInstall{weights: map[string]*v41ExpertDeviceWeight{}}
		v41ExpertDeviceRegs[m] = store
	}
	return store
}

// installV41ExpertQuantDevice validates one V4.1 routed-expert (weight, scale) pair and
// installs its PACKED bytes into the model's device-expert store. It mirrors the dense
// quantizeV41DenseFP8TensorInto install (m.q8w[weightName]) but preserves the packed
// representation: MXFP4 (I8 code nibbles + F8_E8M0) is stored verbatim, with NO call to
// decodeV41ExpertQuant and therefore NO full-checkpoint dequantization. name is the
// .weight tensor name; scaleName its .scale sibling.
func installV41ExpertQuantDevice(
	weightName, scaleName string,
	weightEntry, scaleEntry stEntry,
	weights, scales []byte,
	m *Model,
) error {
	if m == nil || !m.Cfg.IsDeepSeekV41() {
		return nil
	}
	_, projection, err := parseV41ExpertQuantWeightName(weightName)
	if err != nil {
		return err
	}
	wantScaleName := strings.TrimSuffix(weightName, ".weight") + ".scale"
	if scaleName != wantScaleName {
		return v4QuantMetadataf("V4.1 device install scale name %q, want %q", scaleName, wantScaleName)
	}
	if weightEntry.Dtype != "I8" {
		return v4QuantMetadataf("V4.1 device install %s dtype %q, want I8", weightName, weightEntry.Dtype)
	}
	if scaleEntry.Dtype != "F8_E8M0" {
		return v4QuantMetadataf("V4.1 device install %s dtype %q, want F8_E8M0", scaleName, scaleEntry.Dtype)
	}
	spec, ok := selectV41ExpertQuantSpec(projection, weightEntry.Shape, scaleEntry.Shape)
	if !ok {
		return v4QuantMetadataf("V4.1 device install %s shape %v with %s shape %v does not match a supported V4.1 profile", weightName, weightEntry.Shape, scaleName, scaleEntry.Shape)
	}
	weightBytes, ok := checkedShapeProduct(spec.weightRows, spec.weightCols)
	if !ok || len(weights) != weightBytes {
		return v4QuantMetadataf("V4.1 device install %s has %d bytes, want %d", weightName, len(weights), weightBytes)
	}
	scaleBytes, ok := checkedShapeProduct(spec.scaleRows, spec.scaleCols)
	if !ok || len(scales) != scaleBytes {
		return v4QuantMetadataf("V4.1 device install %s has %d bytes, want %d", scaleName, len(scales), scaleBytes)
	}
	for i, scale := range scales {
		if scale == 0xff {
			return v4QuantMetadataf("V4.1 device install %s scale byte %d is F8_E8M0 NaN", scaleName, i)
		}
	}
	unpackedCols, ok := checkedShapeProduct(spec.weightCols, 2)
	if !ok {
		return v4QuantMetadataf("V4.1 device install %s unpacked shape overflows", weightName)
	}
	store := v41ExpertDeviceStore(m, true)
	// Preserve the packed checkpoint bytes verbatim (copy so the caller's slab can be
	// released); this is the "no full-checkpoint dequantization" invariant.
	store.weights[weightName] = &v41ExpertDeviceWeight{
		projection: projection,
		dtype:      weightEntry.Dtype,
		O:          spec.weightRows,
		I:          unpackedCols,
		weight:     append([]byte(nil), weights...),
		scales:     append([]byte(nil), scales...),
	}
	store.packedPreserved = true
	return nil
}

// V41ExpertDeviceInstalled reports whether a routed expert was installed device-resident
// with its packed bytes preserved (no f32 expansion). It is the observable the witness
// reads to prove the install happened on the intended representation.
func (m *Model) V41ExpertDeviceInstalled(weightName string) bool {
	store := v41ExpertDeviceStore(m, false)
	if store == nil {
		return false
	}
	return store.weights[weightName] != nil
}

// V41ExpertDevicePackedPreserved reports whether the install path kept every expert's
// packed bytes (never materializing f32). False before any install.
func (m *Model) V41ExpertDevicePackedPreserved() bool {
	store := v41ExpertDeviceStore(m, false)
	return store != nil && store.packedPreserved
}

// V41ExpertDeviceMatmul runs one installed routed-expert projection's GEMV on the
// device backend be. It is the roofline callable from expert decode: it hands the
// PACKED weight + E8M0 scale bytes to compute's RoutedExpertDeviceKernel and returns
// the device result. When be implements no such kernel, compute returns
// compute.ErrExpertDeviceKernelAbsent and this propagates it wrapped — the call FAILS
// CLOSED and NEVER falls back to the scalar decodeV41ExpertQuant path. Returns
// ErrV41ExpertQuantMetadata when the named expert was not installed.
func (m *Model) V41ExpertDeviceMatmul(weightName string, be compute.Backend, x []float32) ([]float32, error) {
	store := v41ExpertDeviceStore(m, false)
	if store == nil {
		return nil, v4QuantMetadataf("V4.1 device matmul: no routed experts installed")
	}
	w := store.weights[weightName]
	if w == nil {
		return nil, v4QuantMetadataf("V4.1 device matmul: routed expert %q is not installed", weightName)
	}
	if len(x) != w.I {
		return nil, v4QuantMetadataf("V4.1 device matmul %s: x has %d entries, want %d", weightName, len(x), w.I)
	}
	switch w.dtype {
	case "F8_E4M3":
		y, err := compute.GemvFP8Block32ExpertDevice(weightName, be, w.O, w.I, w.weight, w.scales, x)
		if err != nil {
			return nil, err
		}
		return y, nil
	case "I8":
		y, err := compute.GemvMXFP4ExpertDevice(weightName, be, w.O, w.I, w.weight, w.scales, x)
		if err != nil {
			return nil, err
		}
		return y, nil
	default:
		return nil, v4QuantMetadataf("V4.1 device matmul %s: unsupported packed dtype %q", weightName, w.dtype)
	}
}
