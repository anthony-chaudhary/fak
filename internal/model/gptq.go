package model

// gptq.go — AutoGPTQ/GPTQModel weight-only quantization support.
//
// GPTQ checkpoints store quantized matmul weights as qweight/qzeros/scales triples:
//
//   - <base>.qweight: int32 [in/pack, out], with pack = 32 / bits
//   - <base>.qzeros:  int32 [groups, out/pack], zero-points packed across output rows
//   - <base>.scales:  f16/bf16/f32 [groups, out]
//   - <base>.g_idx:   optional int32/int64 [in] activation-order group index
//
// The runtime matmul name is <base>.weight because the decoder asks for HuggingFace
// projection names such as model.layers.0.self_attn.q_proj.weight. The GPTQ zero-point
// convention is AutoGPTQ's: unpack(qzeros)+1 is the effective zero point.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type gptqTensor struct {
	out, in   int
	bits      int
	pack      int
	groupSize int
	nGroups   int
	qweight   []uint32
	qzeros    []uint32
	scales    []float32
	gidx      []int
}

type gptqQuantConfig struct {
	Bits        int    `json:"bits"`
	GroupSize   int    `json:"group_size"`
	DescAct     bool   `json:"desc_act"`
	QuantMethod string `json:"quant_method"`
}

type gptqPart struct {
	entry stEntry
	data  []byte
}

type gptqTriple struct {
	qweight *gptqPart
	qzeros  *gptqPart
	scales  *gptqPart
	gidx    *gptqPart
}

type gptqOpenedFile struct {
	path string
	sf   *safetensorsFile
}

// LoadGPTQ loads an AutoGPTQ/GPTQModel safetensors checkpoint directory. Normal
// float tensors (embeddings, norms, biases, untied heads) stay in the f32 manifest;
// qweight/qzeros/scales triples become resident GPTQ tensors consumed only by
// Session.GPTQ and residentMatRows.
func LoadGPTQ(dir string) (*Model, error) {
	var cfg Config
	if err := readJSON(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return nil, fmt.Errorf("gptq load config: %w", err)
	}
	qcfg, err := readGPTQQuantConfig(dir)
	if err != nil {
		return nil, err
	}
	files, err := gptqSafetensorsFiles(dir)
	if err != nil {
		return nil, err
	}
	return loadGPTQSafetensorsFiles(files, cfg, qcfg)
}

func readGPTQQuantConfig(dir string) (gptqQuantConfig, error) {
	qcfg := gptqQuantConfig{Bits: 4}
	merge := func(next gptqQuantConfig) {
		if next.Bits != 0 {
			qcfg.Bits = next.Bits
		}
		if next.GroupSize != 0 {
			qcfg.GroupSize = next.GroupSize
		}
		if next.DescAct {
			qcfg.DescAct = true
		}
		if next.QuantMethod != "" {
			qcfg.QuantMethod = next.QuantMethod
		}
	}
	if b, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
		var doc struct {
			QuantizationConfig gptqQuantConfig `json:"quantization_config"`
		}
		if err := json.Unmarshal(b, &doc); err != nil {
			return qcfg, fmt.Errorf("gptq config quantization_config: %w", err)
		}
		merge(doc.QuantizationConfig)
	}
	if b, err := os.ReadFile(filepath.Join(dir, "quantize_config.json")); err == nil {
		var flat gptqQuantConfig
		if err := json.Unmarshal(b, &flat); err != nil {
			return qcfg, fmt.Errorf("gptq quantize_config.json: %w", err)
		}
		merge(flat)
	}
	if qcfg.Bits != 4 && qcfg.Bits != 8 {
		return qcfg, fmt.Errorf("gptq load: unsupported bits=%d (want 4 or 8)", qcfg.Bits)
	}
	return qcfg, nil
}

func gptqSafetensorsFiles(dir string) ([]string, error) {
	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	if b, err := os.ReadFile(idxPath); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(b, &index); err != nil {
			return nil, fmt.Errorf("gptq safetensors index: %w", err)
		}
		set := map[string]bool{}
		for _, shard := range index.WeightMap {
			set[filepath.Join(dir, shard)] = true
		}
		files := make([]string, 0, len(set))
		for f := range set {
			files = append(files, f)
		}
		sort.Strings(files)
		return files, nil
	}
	modelPath := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(modelPath); err == nil {
		return []string{modelPath}, nil
	}
	matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("gptq glob safetensors: %w", err)
	}
	sort.Strings(matches)
	if len(matches) == 0 {
		return nil, fmt.Errorf("gptq load: no safetensors files found in %s", dir)
	}
	return matches, nil
}

func loadGPTQSafetensorsFiles(files []string, cfg Config, qcfg gptqQuantConfig) (*Model, error) {
	openedFiles := make([]gptqOpenedFile, 0, len(files))
	for _, path := range files {
		sf, err := openSafetensorsFile(path)
		if err != nil {
			closeSafetensorsFiles(openedFiles)
			return nil, fmt.Errorf("gptq open %s: %w", path, err)
		}
		openedFiles = append(openedFiles, gptqOpenedFile{path: path, sf: sf})
	}
	defer closeSafetensorsFiles(openedFiles)

	bases := map[string]bool{}
	for _, of := range openedFiles {
		for name := range of.sf.hdr {
			if strings.HasSuffix(name, ".qweight") {
				bases[strings.TrimSuffix(name, ".qweight")] = true
			}
		}
	}
	if len(bases) == 0 {
		return nil, fmt.Errorf("gptq load: no qweight/qzeros/scales triples found")
	}
	gptqNames := gptqAuxNameSet(bases)

	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	triples := map[string]*gptqTriple{}
	for _, of := range openedFiles {
		if err := appendGPTQF32FileInto(of.sf, gptqNames, man, &raw, &off, cfg); err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(of.path), err)
		}
		if err := collectGPTQParts(of.sf, bases, triples); err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(of.path), err)
		}
	}

	gptqw := make(map[string]*gptqTensor, len(triples))
	for base := range bases {
		tr := triples[base]
		if tr == nil || tr.qweight == nil || tr.qzeros == nil || tr.scales == nil {
			return nil, fmt.Errorf("gptq load: %s missing qweight/qzeros/scales", base)
		}
		name := gptqWeightName(base)
		qt, err := decodeGPTQTensor(base, tr, qcfg)
		if err != nil {
			return nil, err
		}
		if _, exists := gptqw[name]; exists {
			return nil, fmt.Errorf("gptq load: duplicate tensor %s", name)
		}
		gptqw[name] = qt
	}

	m, err := newModel(cfg, man, raw)
	if err != nil {
		return nil, err
	}
	m.gptqw = gptqw
	return m, nil
}

func closeSafetensorsFiles(files []gptqOpenedFile) {
	for _, of := range files {
		_ = of.sf.Close()
	}
}

func gptqAuxNameSet(bases map[string]bool) map[string]bool {
	names := map[string]bool{}
	for base := range bases {
		names[base+".qweight"] = true
		names[base+".qzeros"] = true
		names[base+".scales"] = true
		names[base+".g_idx"] = true
	}
	return names
}

func appendGPTQF32FileInto(sf *safetensorsFile, skip map[string]bool, man map[string]tensorMeta, raw *[]byte, off *int, cfg Config) error {
	for _, name := range safetensorsTensorNames(sf.hdr) {
		if skip[name] || skipLoadTensor(cfg, name) {
			continue
		}
		if _, ok := man[name]; ok {
			return fmt.Errorf("safetensors: duplicate tensor %s", name)
		}
		if skipped, err := decodeAppendF32Tensor(sf, name, man, raw, off); err != nil {
			return err
		} else if skipped {
			continue
		}
	}
	return nil
}

func collectGPTQParts(sf *safetensorsFile, bases map[string]bool, triples map[string]*gptqTriple) error {
	for _, name := range safetensorsTensorNames(sf.hdr) {
		base, kind, ok := splitGPTQName(name)
		if !ok || !bases[base] {
			continue
		}
		var e stEntry
		if err := json.Unmarshal(sf.hdr[name], &e); err != nil {
			return fmt.Errorf("gptq entry %s: %w", name, err)
		}
		data, err := sf.tensorBytes(e)
		if err != nil {
			return fmt.Errorf("gptq tensor %s: %w", name, err)
		}
		owned := append([]byte(nil), data...)
		tr := triples[base]
		if tr == nil {
			tr = &gptqTriple{}
			triples[base] = tr
		}
		part := &gptqPart{entry: e, data: owned}
		switch kind {
		case "qweight":
			tr.qweight = part
		case "qzeros":
			tr.qzeros = part
		case "scales":
			tr.scales = part
		case "g_idx":
			tr.gidx = part
		}
	}
	return nil
}

func splitGPTQName(name string) (base, kind string, ok bool) {
	for _, suffix := range []string{".qweight", ".qzeros", ".scales", ".g_idx"} {
		if strings.HasSuffix(name, suffix) {
			return strings.TrimSuffix(name, suffix), strings.TrimPrefix(suffix, "."), true
		}
	}
	return "", "", false
}

func gptqWeightName(base string) string {
	if strings.HasSuffix(base, ".weight") {
		return base
	}
	return base + ".weight"
}

func decodeGPTQTensor(base string, tr *gptqTriple, qcfg gptqQuantConfig) (*gptqTensor, error) {
	bits := qcfg.Bits
	pack := 32 / bits
	wEntry := tr.qweight.entry
	if wEntry.Dtype != "I32" && wEntry.Dtype != "U32" {
		return nil, fmt.Errorf("gptq load %s.qweight dtype %q, want I32/U32", base, wEntry.Dtype)
	}
	if len(wEntry.Shape) != 2 {
		return nil, fmt.Errorf("gptq load %s.qweight shape %v, want rank-2", base, wEntry.Shape)
	}
	inPacked, out := wEntry.Shape[0], wEntry.Shape[1]
	in := inPacked * pack
	qweight := bytesToU32LE(tr.qweight.data)
	if len(qweight) != inPacked*out {
		return nil, fmt.Errorf("gptq load %s.qweight len %d != %d", base, len(qweight), inPacked*out)
	}

	scF32Bytes, err := decodeSafetensorF32(base+".scales", tr.scales.entry, tr.scales.data)
	if err != nil {
		return nil, fmt.Errorf("gptq load %s.scales: %w", base, err)
	}
	scales := make([]float32, len(scF32Bytes)/4)
	for i := range scales {
		scales[i] = math.Float32frombits(binary.LittleEndian.Uint32(scF32Bytes[i*4 : i*4+4]))
	}
	if len(tr.scales.entry.Shape) != 2 || tr.scales.entry.Shape[1] != out {
		return nil, fmt.Errorf("gptq load %s.scales shape %v, want [groups,%d]", base, tr.scales.entry.Shape, out)
	}
	nGroups := tr.scales.entry.Shape[0]
	if nGroups <= 0 || len(scales) != nGroups*out {
		return nil, fmt.Errorf("gptq load %s.scales len %d != groups*out %d", base, len(scales), nGroups*out)
	}

	zEntry := tr.qzeros.entry
	if zEntry.Dtype != "I32" && zEntry.Dtype != "U32" {
		return nil, fmt.Errorf("gptq load %s.qzeros dtype %q, want I32/U32", base, zEntry.Dtype)
	}
	if out%pack != 0 {
		return nil, fmt.Errorf("gptq load %s out=%d not divisible by pack=%d", base, out, pack)
	}
	if !sameShape(zEntry.Shape, []int{nGroups, out / pack}) {
		return nil, fmt.Errorf("gptq load %s.qzeros shape %v, want [%d,%d]", base, zEntry.Shape, nGroups, out/pack)
	}
	qzeros := bytesToU32LE(tr.qzeros.data)
	if len(qzeros) != nGroups*(out/pack) {
		return nil, fmt.Errorf("gptq load %s.qzeros len %d != groups*out/pack %d", base, len(qzeros), nGroups*(out/pack))
	}

	gidx, err := decodeGPTQGIdx(base, tr.gidx, in)
	if err != nil {
		return nil, err
	}
	groupSize := qcfg.GroupSize
	if groupSize <= 0 {
		if gidx == nil && nGroups == 1 {
			groupSize = in
		} else if gidx == nil && in%nGroups == 0 {
			groupSize = in / nGroups
		}
	}
	qt, err := newGPTQTensor(qweight, qzeros, scales, gidx, out, in, bits, groupSize)
	if err != nil {
		return nil, fmt.Errorf("gptq load %s: %w", base, err)
	}
	return qt, nil
}

func decodeGPTQGIdx(base string, part *gptqPart, in int) ([]int, error) {
	if part == nil {
		return nil, nil
	}
	if len(part.entry.Shape) != 1 || part.entry.Shape[0] != in {
		return nil, fmt.Errorf("gptq load %s.g_idx shape %v, want [%d]", base, part.entry.Shape, in)
	}
	out := make([]int, in)
	switch part.entry.Dtype {
	case "I32", "U32":
		if len(part.data) != in*4 {
			return nil, fmt.Errorf("gptq load %s.g_idx bytes %d != %d", base, len(part.data), in*4)
		}
		for i := range out {
			out[i] = int(int32(binary.LittleEndian.Uint32(part.data[i*4:])))
		}
	case "I64", "U64":
		if len(part.data) != in*8 {
			return nil, fmt.Errorf("gptq load %s.g_idx bytes %d != %d", base, len(part.data), in*8)
		}
		for i := range out {
			out[i] = int(int64(binary.LittleEndian.Uint64(part.data[i*8:])))
		}
	default:
		return nil, fmt.Errorf("gptq load %s.g_idx dtype %q, want I32/I64", base, part.entry.Dtype)
	}
	return out, nil
}

func newGPTQTensor(qweight, qzeros []uint32, scales []float32, gidx []int, out, in, bits, groupSize int) (*gptqTensor, error) {
	if bits != 4 && bits != 8 {
		return nil, fmt.Errorf("unsupported bits=%d", bits)
	}
	pack := 32 / bits
	if in%pack != 0 {
		return nil, fmt.Errorf("input dim %d not divisible by pack=%d", in, pack)
	}
	if out%pack != 0 {
		return nil, fmt.Errorf("output dim %d not divisible by pack=%d", out, pack)
	}
	if len(scales)%out != 0 {
		return nil, fmt.Errorf("scales len %d not divisible by out=%d", len(scales), out)
	}
	nGroups := len(scales) / out
	if nGroups <= 0 {
		return nil, fmt.Errorf("no scale groups")
	}
	if len(qweight) != (in/pack)*out {
		return nil, fmt.Errorf("qweight len %d != in/pack*out %d", len(qweight), (in/pack)*out)
	}
	if len(qzeros) != nGroups*(out/pack) {
		return nil, fmt.Errorf("qzeros len %d != groups*out/pack %d", len(qzeros), nGroups*(out/pack))
	}
	if gidx != nil {
		if len(gidx) != in {
			return nil, fmt.Errorf("g_idx len %d != in=%d", len(gidx), in)
		}
		for i, g := range gidx {
			if g < 0 || g >= nGroups {
				return nil, fmt.Errorf("g_idx[%d]=%d outside [0,%d)", i, g, nGroups)
			}
		}
	} else {
		if groupSize <= 0 {
			return nil, fmt.Errorf("group_size must be positive when g_idx is absent")
		}
		if ceilDiv(in, groupSize) != nGroups {
			return nil, fmt.Errorf("group_size=%d implies %d groups for in=%d, scales carry %d", groupSize, ceilDiv(in, groupSize), in, nGroups)
		}
	}
	return &gptqTensor{
		out:       out,
		in:        in,
		bits:      bits,
		pack:      pack,
		groupSize: groupSize,
		nGroups:   nGroups,
		qweight:   qweight,
		qzeros:    qzeros,
		scales:    scales,
		gidx:      gidx,
	}, nil
}

func ceilDiv(n, d int) int {
	return (n + d - 1) / d
}

func (qt *gptqTensor) groupForInput(i int) int {
	if qt.gidx != nil {
		return qt.gidx[i]
	}
	g := i / qt.groupSize
	if g >= qt.nGroups {
		return qt.nGroups - 1
	}
	return g
}

func (qt *gptqTensor) code(i, o int) uint32 {
	v := qt.qweight[(i/qt.pack)*qt.out+o]
	return (v >> (uint(i%qt.pack) * uint(qt.bits))) & ((1 << uint(qt.bits)) - 1)
}

func (qt *gptqTensor) zero(g, o int) uint32 {
	v := qt.qzeros[g*(qt.out/qt.pack)+o/qt.pack]
	return ((v >> (uint(o%qt.pack) * uint(qt.bits))) & ((1 << uint(qt.bits)) - 1)) + 1
}

func (qt *gptqTensor) weight(i, o int) float32 {
	g := qt.groupForInput(i)
	return (float32(qt.code(i, o)) - float32(qt.zero(g, o))) * qt.scales[g*qt.out+o]
}

func gptqDequantRow(dst []float32, qt *gptqTensor, o int) {
	dst = dst[:qt.in]
	for i := 0; i < qt.in; i++ {
		dst[i] = qt.weight(i, o)
	}
}

func gptqMatRows(qt *gptqTensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	gptqMatRowsInto(qt, x, y)
	return y
}

func gptqMatRowsInto(qt *gptqTensor, x, y []float32) {
	y = y[:qt.out]
	if numWorkers <= 1 || qt.out*qt.in < parThreshold {
		gptqMatRowsRange(qt, x, y, 0, qt.out)
		return
	}
	parFor(qt.out, numWorkers, func(lo, hi int) { gptqMatRowsRange(qt, x, y, lo, hi) })
}

func gptqMatRowsRange(qt *gptqTensor, x, y []float32, lo, hi int) {
	for o := lo; o < hi; o++ {
		var s0, s1, s2, s3 float32
		i := 0
		for ; i+3 < qt.in; i += 4 {
			s0 += qt.weight(i, o) * x[i]
			s1 += qt.weight(i+1, o) * x[i+1]
			s2 += qt.weight(i+2, o) * x[i+2]
			s3 += qt.weight(i+3, o) * x[i+3]
		}
		acc := (s0 + s1) + (s2 + s3)
		for ; i < qt.in; i++ {
			acc += qt.weight(i, o) * x[i]
		}
		y[o] = acc
	}
}

func (m *Model) gptq(name string) *gptqTensor {
	if m.gptqw == nil {
		panic("model: no GPTQ tensors loaded (call LoadGPTQ)")
	}
	qt, ok := m.gptqw[name]
	if !ok {
		panic("model: GPTQ tensor not found: " + name)
	}
	return qt
}

func (m *Model) hasGPTQ(name string) bool {
	return m.gptqw != nil && m.gptqw[name] != nil
}

func (m *Model) GPTQCount() int { return len(m.gptqw) }

func (m *Model) GPTQShape(name string) (out, in, bits, groupSize int) {
	if qt := m.gptqw[name]; qt != nil {
		return qt.out, qt.in, qt.bits, qt.groupSize
	}
	return 0, 0, 0, 0
}

func (m *Model) gptqHeadName() string {
	if m.hasGPTQ("lm_head.weight") {
		return "lm_head.weight"
	}
	return "model.embed_tokens.weight"
}

func (s *Session) headGPTQ(xf []float32) []float32 {
	y, t := s.headLogitsBuf()
	gptqMatRowsInto(s.M.gptq(s.M.gptqHeadName()), xf, y)
	logitScaleInPlace(y, s.M.Cfg)
	s.phaseEnd("lm_head_gptq", t)
	return y
}

func (s *Session) tokenHiddenGPTQ(id, pos int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg)
	mat := matKernel(residentKernel{m})
	for l := 0; l < cfg.NumLayers; l++ {
		cos, sin := ropeRowForLayer(cfg, l, pos)
		x = s.blockStep(l, pos, x, cos, sin, mat)
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	return rmsnormCfg(x, m.tensor("model.norm.weight"), float32(cfg.RMSNormEps), cfg)
}
