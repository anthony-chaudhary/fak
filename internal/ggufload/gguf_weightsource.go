package ggufload

import (
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// qwen35MTPMaterializationName maps the closed Qwen3.8 trailing-block vocabulary into
// the existing native mtp.* namespace. It adapts the target/MTP split and four glue-role
// remaps from llama.cpp's MIT-licensed Qwen converter:
// https://github.com/ggml-org/llama.cpp/blob/9723942adc518b43c4b95dc4dce6906903eb5e09/conversion/qwen.py#L277-L354
//
// handled is true for every trailing-block, nextn, or vision-sidecar tensor in this family.
// An empty canonical name means skip: default target-only loads, vision tensors, misplaced
// NextN tensors, and unknown future sidecars never leak into the target manifest.
func qwen35MTPMaterializationName(name string, cfg model.Config) (canonical string, handled bool) {
	if cfg.ModelType != "qwen35" && cfg.ModelType != "qwen35moe" {
		return "", false
	}
	if isGLMMoeDsaVisionTensor(name) {
		return "", true
	}
	layer, suffix, ok := parseGLMBlkLayerSuffix(name)
	if !ok {
		return "", isGLMMoeDsaMTPTensor(name)
	}
	firstMTP := cfg.NumLayers
	if layer < firstMTP || layer >= firstMTP+cfg.NumNextNPredictLayers {
		return "", isGLMMoeDsaMTPTensor(name)
	}
	if !model.RetainMTP {
		return "", true
	}

	switch suffix {
	case "nextn.eh_proj.weight":
		return "mtp.fc.weight", true
	case "nextn.enorm.weight":
		return "mtp.pre_fc_norm_embedding.weight", true
	case "nextn.hnorm.weight":
		return "mtp.pre_fc_norm_hidden.weight", true
	case "nextn.shared_head_norm.weight":
		return "mtp.norm.weight", true
	}
	if strings.HasPrefix(suffix, "nextn.") {
		return "", true
	}

	base, ok := CanonicalTensorNameArch(name, cfg.ModelType)
	if !ok {
		return "", true
	}
	prefix := fmt.Sprintf("model.layers.%d.", layer)
	decoderSuffix := strings.TrimPrefix(base, prefix)
	if decoderSuffix == base {
		return "", true
	}
	switch decoderSuffix {
	case "input_layernorm.weight",
		"post_attention_layernorm.weight",
		"self_attn.q_norm.weight",
		"self_attn.k_norm.weight",
		"self_attn.q_proj.weight",
		"self_attn.k_proj.weight",
		"self_attn.v_proj.weight",
		"self_attn.o_proj.weight",
		"mlp.gate_proj.weight",
		"mlp.up_proj.weight",
		"mlp.down_proj.weight":
		return fmt.Sprintf("mtp.layers.%d.%s", layer-firstMTP, decoderSuffix), true
	default:
		return "", true
	}
}

var qwen35MTPRequiredMaterialized = [...]string{
	"mtp.fc.weight",
	"mtp.pre_fc_norm_embedding.weight",
	"mtp.pre_fc_norm_hidden.weight",
	"mtp.norm.weight",
	"mtp.layers.0.input_layernorm.weight",
	"mtp.layers.0.post_attention_layernorm.weight",
	"mtp.layers.0.self_attn.q_norm.weight",
	"mtp.layers.0.self_attn.k_norm.weight",
	"mtp.layers.0.self_attn.q_proj.weight",
	"mtp.layers.0.self_attn.k_proj.weight",
	"mtp.layers.0.self_attn.v_proj.weight",
	"mtp.layers.0.self_attn.o_proj.weight",
	"mtp.layers.0.mlp.gate_proj.weight",
	"mtp.layers.0.mlp.up_proj.weight",
	"mtp.layers.0.mlp.down_proj.weight",
}

func newQwen35MTPSeen(cfg model.Config) map[string]bool {
	if !model.RetainMTP || cfg.NumNextNPredictLayers == 0 || (cfg.ModelType != "qwen35" && cfg.ModelType != "qwen35moe") {
		return nil
	}
	return make(map[string]bool, len(qwen35MTPRequiredMaterialized))
}

func validateQwen35MTPMaterialized(seen map[string]bool) error {
	if seen == nil {
		return nil
	}
	missing := make([]string, 0)
	for _, name := range qwen35MTPRequiredMaterialized {
		if !seen[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) != 0 {
		return fmt.Errorf("gguf: incomplete retained Qwen MTP head: missing %s", strings.Join(missing, ", "))
	}
	return nil
}

func validateQwen35MTPShape(name string, shape []int, cfg model.Config) error {
	h, hd := cfg.HiddenSize, cfg.HeadDim
	qOut := cfg.NumHeads * hd
	if cfg.AttnOutputGate {
		qOut *= 2
	}
	var want []int
	switch name {
	case "mtp.fc.weight":
		want = []int{h, 2 * h}
	case "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight",
		"mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight":
		want = []int{h}
	case "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight":
		want = []int{hd}
	case "mtp.layers.0.self_attn.q_proj.weight":
		want = []int{qOut, h}
	case "mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.self_attn.v_proj.weight":
		want = []int{cfg.NumKVHeads * hd, h}
	case "mtp.layers.0.self_attn.o_proj.weight":
		want = []int{h, cfg.NumHeads * hd}
	case "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight":
		want = []int{cfg.IntermediateSize, h}
	case "mtp.layers.0.mlp.down_proj.weight":
		want = []int{h, cfg.IntermediateSize}
	default:
		return nil
	}
	if len(shape) != len(want) {
		return fmt.Errorf("gguf: Qwen MTP tensor %s has shape %v, want %v", name, shape, want)
	}
	for i := range shape {
		if shape[i] != want[i] {
			return fmt.Errorf("gguf: Qwen MTP tensor %s has shape %v, want %v", name, shape, want)
		}
	}
	return nil
}

// mappedQ4KReaderAt carries the retained shard mapping beside the unchanged
// ReaderAt fallback. span starts at shard offset zero and is capped to complete
// pages because Metal's no-copy buffer contract requires a page-multiple view.
type mappedQ4KReaderAt struct {
	io.ReaderAt
	span   []byte
	offset int
}

// LoadModel loads a GGUF checkpoint through the default dequant-to-f32 path and returns a
// regular in-kernel model.Model. GGUF tensor names are normalized to the canonical HF-Llama
// names that internal/model already consumes.
func LoadModel(path string) (*model.Model, error) {
	return loadVia(path, (*WeightSource).Model)
}

// loadVia is the shared open/defer-close/delegate skeleton for the GGUF entry points that take
// no profiler (LoadModel, LoadModelQ4KProfile): open the checkpoint, ensure the source is closed,
// and hand it to build, which picks the resident path (dequant-f32, lean-Q8, direct-Q4_K).
func loadVia(path string, build func(*WeightSource) (*model.Model, error)) (*model.Model, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return build(ws)
}

// LoadModelQuant loads a GGUF checkpoint through the memory-lean quant-on-load path:
// each tensor is dequantized only long enough to normalize/quantize it, resident matmul
// weights are kept as Q8_0, and only small non-matmul tensors remain f32.
func LoadModelQuant(path string) (*model.Model, error) {
	return LoadModelQuantProfile(path, nil)
}

// LoadModelQuantProfile is LoadModelQuant with an optional LoadProfiler that records
// per-tensor and per-phase timings of the quant-on-load path. A nil profiler is a no-op.
func LoadModelQuantProfile(path string, p *LoadProfiler) (*model.Model, error) {
	t := loadProfileStart(p)
	ws, err := OpenWeights(path)
	loadProfileEnd(p, "gguf_open_index", t, 0, 0)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.QuantModelProfile(p)
}

// NewWeightSource builds a WeightSource over a parsed GGUF File and its reader,
// indexing tensors by name and erroring on a duplicate tensor name.
func NewWeightSource(f *File, r io.ReaderAt, size int64) (*WeightSource, error) {
	byName := make(map[string]int, len(f.Tensors))
	for i, t := range f.Tensors {
		if _, ok := byName[t.Name]; ok {
			return nil, fmt.Errorf("gguf: duplicate tensor %s", t.Name)
		}
		byName[t.Name] = i
	}
	return &WeightSource{File: f, r: r, size: size, byName: byName}, nil
}

// Close closes every shard reader the source opened, returning the first close error.
func (s *WeightSource) Close() error {
	if len(s.closers) == 0 {
		return nil
	}
	var firstErr error
	for _, c := range s.closers {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.closers = nil
	return firstErr
}

// Tensor looks up a tensor's TensorInfo by GGUF name, reporting whether it is present.
func (s *WeightSource) Tensor(name string) (TensorInfo, bool) {
	i, ok := s.byName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return s.File.Tensors[i], true
}

func (s *WeightSource) tensorReader(info TensorInfo) (io.ReaderAt, int64, error) {
	r, size := s.r, s.size
	if idx, ok := s.byName[info.Name]; ok && idx < len(s.readerFor) && s.readerFor[idx] != nil {
		r, size = s.readerFor[idx], s.sizeFor[idx]
	}
	if r == nil {
		return nil, 0, fmt.Errorf("gguf: tensor %s has no reader", info.Name)
	}
	return r, size, nil
}

// mappedQ4KReader returns the ordinary shard reader unless this tensor is fully
// contained in a retained, page-aligned mmap prefix accepted by the Metal span
// upload. A tensor in the shard's partial final page keeps the byte-identical
// ReadAt path rather than exposing bytes outside the retained mapping.
func (s *WeightSource) mappedQ4KReader(info TensorInfo, r io.ReaderAt, size int64, n int) io.ReaderAt {
	data := s.data
	if idx, ok := s.byName[info.Name]; ok && idx < len(s.dataFor) && s.dataFor[idx] != nil {
		data = s.dataFor[idx]
	}
	page := os.Getpagesize()
	if n <= 0 || len(data) == 0 || page <= 0 || info.FileOffset < 0 || info.FileOffset%32 != 0 ||
		int64(len(data)) != size || uintptr(unsafe.Pointer(&data[0]))%uintptr(page) != 0 {
		return r
	}
	spanLen := len(data) - len(data)%page
	if spanLen == 0 || info.FileOffset > int64(spanLen) || int64(n) > int64(spanLen)-info.FileOffset {
		return r
	}
	offset := int(info.FileOffset)
	span := data[:spanLen:spanLen]
	return &mappedQ4KReaderAt{ReaderAt: r, span: span, offset: offset}
}

// TensorBytes reads a named tensor's raw (still-quantized) payload bytes from the
// shard reader that holds it, bounds-checking the offset and length against the file.
// It always returns a fresh heap copy — even when the shard reader is a FAK_GGUF_MMAP
// map (gguf_mmap.go) whose ws.data/ws.dataFor could be aliased — so callers may retain
// the bytes past ws.Close(); the zero-copy aliasing read is a later SSD-offload slice.
func (s *WeightSource) TensorBytes(name string) ([]byte, TensorInfo, error) {
	info, ok := s.Tensor(name)
	if !ok {
		return nil, TensorInfo{}, fmt.Errorf("gguf: missing tensor %s", name)
	}
	n, err := tensorPayloadBytes(info)
	if err != nil {
		return nil, info, err
	}
	if n > uint64(math.MaxInt) || n > uint64(math.MaxInt64) {
		return nil, info, fmt.Errorf("gguf: tensor %s payload is too large", name)
	}
	// Route to the shard reader that holds this tensor's bytes. For a single-file
	// checkpoint readerFor is nil and we read from the primary reader, as before.
	r, sz := s.r, s.size
	if idx, ok := s.byName[name]; ok && idx < len(s.readerFor) && s.readerFor[idx] != nil {
		r = s.readerFor[idx]
		sz = s.sizeFor[idx]
	}
	if info.FileOffset < 0 || info.FileOffset > math.MaxInt64-int64(n) || info.FileOffset+int64(n) > sz {
		return nil, info, fmt.Errorf("gguf: tensor %s overruns file", name)
	}
	buf := make([]byte, int(n))
	if _, err := r.ReadAt(buf, info.FileOffset); err != nil {
		return nil, info, fmt.Errorf("gguf: read tensor %s: %w", name, err)
	}
	return buf, info, nil
}

// TensorF32 reads a named tensor and dequantizes its payload to float32.
func (s *WeightSource) TensorF32(name string) ([]float32, TensorInfo, error) {
	raw, info, err := s.TensorBytes(name)
	if err != nil {
		return nil, info, err
	}
	out, err := dequantF32(info, raw)
	return out, info, err
}

// Model builds an in-kernel model.Model from this source via the dequant-to-f32 path.
func (s *WeightSource) Model() (*model.Model, error) {
	cfg, tensors, err := s.F32Tensors()
	if err != nil {
		return nil, err
	}
	return model.NewFromF32Tensors(cfg, tensors)
}

// QuantModel builds an in-kernel model.Model via the memory-lean quant-on-load path
// (matmul weights kept Q8_0), without profiling.
func (s *WeightSource) QuantModel() (*model.Model, error) {
	return s.QuantModelProfile(nil)
}

// QuantModelProfile builds an in-kernel model.Model via the quant-on-load path,
// dequantizing each GGUF tensor only long enough to normalize and re-quantize it into
// the model.QuantBuilder; glm_moe_dsa batched experts are split 1->E first, and an
// optional LoadProfiler records per-phase timings. A nil profiler is a no-op.
func (s *WeightSource) QuantModelProfile(p *LoadProfiler) (*model.Model, error) {
	t := loadProfileStart(p)
	cfg, err := s.File.Config()
	loadProfileEnd(p, "gguf_config", t, 0, 0)
	if err != nil {
		return nil, err
	}
	builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	// One dequant arena reused across every tensor: each weight is dequantized only long
	// enough to be re-quantized into the builder, so without reuse the 27B path would
	// allocate (and the GC unmap) 800+ throwaway elems*4 f32 buffers — the load-time page
	// churn #440 targets. Safe because each tensor's f32 is fully consumed (quantized or
	// copied into the f32 blob) before the next dequantF32Into overwrites it.
	//
	// FAK_GGUF_NO_ARENA_REUSE=1 turns the reuse OFF (every tensor gets a fresh f32 buffer,
	// the pre-#440 behavior). It exists only so the page-churn win is A/B-measurable on a
	// single host via modelbench -load-only -load-profile: the reuse-safe vs fresh-alloc
	// arms produce the bit-identical model (proven by TestDequantF32IntoReusesArenaAndNeverLeaksStaleData)
	// while their peak-RSS / page-fault profiles differ. Unset on every normal load.
	reuseArena := os.Getenv("FAK_GGUF_NO_ARENA_REUSE") == ""
	var dequantBuf []float32
	// glm_moe_dsa MLA KV-b merge buffer: the split attn_k_b / attn_v_b for a layer may not be
	// adjacent in the tensor stream, so buffer the first half seen per layer and emit the merged
	// kv_b_proj when its partner arrives (mergeGLMMoeDsaKVB). See gguf_glm_tensors.go.
	kvbHalf := map[int]glmKVBHalf{}
	qwenMTPSeen := newQwen35MTPSeen(cfg)
	p.SetTotal(len(s.File.Tensors)) // arm the progress reporter (no-op on nil / unset Progress)
	for _, info := range s.File.Tensors {
		p.Tick(tensorOnDiskBytes(info)) // one GGUF tensor consumed -> advance the % status
		name, qwenMTPHandled := qwen35MTPMaterializationName(info.Name, cfg)
		if qwenMTPHandled && name == "" {
			continue
		}
		// Non-Qwen sidecars retain their historical materialization behavior: neither their
		// MTP layout nor their vision layout has a canonical native slot in this loader.
		if !qwenMTPHandled && archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name) {
			continue
		}
		if archUsesMLAMoELayout(cfg.ModelType) {
			if layer, half, ok := glmMoeDsaSplitKVB(info.Name); ok {
				shape, data, err := s.dequantGGUFShapeF32(info)
				if err != nil {
					return nil, err
				}
				dataCopy := append([]float32(nil), data...) // detach from the reused dequant arena
				merged, ready, err := bufferGLMKVBHalf(kvbHalf, layer, half, shape, dataCopy)
				if err != nil {
					return nil, err
				}
				if ready {
					if err := builder.AddF32Tensor(merged.Name, merged.Shape, merged.Data); err != nil {
						return nil, err
					}
				}
				continue
			}
		}
		// GGUF batched routed experts: split the [E,out,in] blob 1->E into per-expert
		// canonical tensors and add each (the quant builder narrows the 2-D matmul weights as
		// usual). Handled before CanonicalTensorNameArch, which leaves these unmapped.
		if archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
			if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				shape, data, err := s.dequantGGUFShapeF32(info)
				if err != nil {
					return nil, err
				}
				experts, err := splitGLMMoeDsaExperts(layer, proj, shape, data)
				if err != nil {
					return nil, err
				}
				for _, ex := range experts {
					if err := builder.AddF32Tensor(ex.Name, ex.Shape, ex.Data); err != nil {
						return nil, err
					}
				}
				continue
			}
		}
		var tensorStart time.Time
		var tt LoadTensorStat
		if p != nil {
			tensorStart = time.Now()
			tt = LoadTensorStat{Name: info.Name, Type: info.Type.String()}
		}

		t = loadProfileStart(p)
		if !qwenMTPHandled {
			var ok bool
			name, ok = CanonicalTensorNameArch(info.Name, cfg.ModelType)
			if !ok {
				return nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
			}
		}
		if p != nil {
			tt.CanonicalName = name
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return nil, err
		}
		if qwenMTPHandled {
			if err := validateQwen35MTPShape(name, shape, cfg); err != nil {
				return nil, err
			}
		}
		if p != nil {
			tt.Shape = append([]int(nil), shape...)
		}
		loadProfileEnd(p, "gguf_map_shape", t, 0, 1)

		t = loadProfileStart(p)
		raw, _, err := s.TensorBytes(info.Name)
		readNanos := loadProfileEnd(p, "gguf_read", t, int64(len(raw)), 1)
		if p != nil {
			tt.ReadNanos = readNanos
			tt.PayloadBytes = int64(len(raw))
		}
		if err != nil {
			return nil, err
		}

		t = loadProfileStart(p)
		scratch := dequantBuf
		if !reuseArena {
			scratch = nil // force a fresh allocation per tensor (pre-#440 page-churn arm)
		}
		data, err := dequantF32Into(scratch, info, raw)
		dequantNanos := loadProfileEnd(p, "gguf_dequant", t, int64(len(data))*4, 1)
		if p != nil {
			tt.DequantNanos = dequantNanos
			tt.Values = len(data)
		}
		if err != nil {
			return nil, err
		}
		// Carry the (possibly grown) arena forward. Capture it before normalize, which may
		// hand back a fresh reordered buffer instead of data — dequantBuf must stay the
		// dequant arena so the next tensor reuses it. Skipped on the no-reuse arm so the next
		// tensor allocates fresh.
		if reuseArena {
			dequantBuf = data
		}

		t = loadProfileStart(p)
		data, err = normalizeCanonicalTensorData(name, data, cfg)
		normalizeNanos := loadProfileEnd(p, "gguf_normalize", t, int64(len(data))*4, 1)
		if p != nil {
			tt.NormalizeNanos = normalizeNanos
		}
		if err != nil {
			return nil, err
		}

		t = loadProfileStart(p)
		if err := builder.AddF32Tensor(name, shape, data); err != nil {
			addNanos := loadProfileEnd(p, "quant_builder_add", t, int64(len(data))*4, 1)
			if p != nil {
				tt.AddNanos = addNanos
			}
			return nil, err
		}
		if qwenMTPHandled {
			qwenMTPSeen[name] = true
		}
		addNanos := loadProfileEnd(p, "quant_builder_add", t, int64(len(data))*4, 1)
		if p != nil {
			tt.AddNanos = addNanos
			tt.TotalNanos = time.Since(tensorStart).Nanoseconds()
			p.recordTensor(tt)
		}
	}
	if err := glmKVBUnpaired(kvbHalf); err != nil {
		return nil, err
	}
	if err := validateQwen35MTPMaterialized(qwenMTPSeen); err != nil {
		return nil, err
	}
	t = loadProfileStart(p)
	m, err := builder.Build()
	loadProfileEnd(p, "quant_builder_finalize", t, 0, 0)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// F32Tensors dequantizes every GGUF tensor to float32, mapping each to its canonical
// HF name and normalizing its data, and returns the model Config plus the named f32
// tensors (GGUF batched MoE experts split 1->E). It is the f32 path's Model builds on.
func (s *WeightSource) F32Tensors() (model.Config, []model.NamedTensorF32, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return model.Config{}, nil, err
	}
	tensors := make([]model.NamedTensorF32, 0, len(s.File.Tensors))
	kvbHalf := map[int]glmKVBHalf{} // MLA KV-b 2->1 merge buffer (see QuantModelProfile)
	qwenMTPSeen := newQwen35MTPSeen(cfg)
	for _, info := range s.File.Tensors {
		name, qwenMTPHandled := qwen35MTPMaterializationName(info.Name, cfg)
		if qwenMTPHandled && name == "" {
			continue
		}
		if !qwenMTPHandled && archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name) {
			continue
		}
		// GGUF batched routed experts: one [E,out,in] blob splits 1->E into per-expert
		// canonical tensors. Handled before CanonicalTensorNameArch (which leaves them unmapped).
		if archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
			if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				shape, data, err := s.shapeAndTensorF32(info)
				if err != nil {
					return model.Config{}, nil, err
				}
				experts, err := splitGLMMoeDsaExperts(layer, proj, shape, data)
				if err != nil {
					return model.Config{}, nil, err
				}
				tensors = append(tensors, experts...)
				continue
			}
			// MLA KV-b: buffer attn_k_b/attn_v_b and emit the combined kv_b_proj when both arrive.
			if archUsesMLAMoELayout(cfg.ModelType) {
				if layer, half, ok := glmMoeDsaSplitKVB(info.Name); ok {
					shape, data, err := s.shapeAndTensorF32(info)
					if err != nil {
						return model.Config{}, nil, err
					}
					merged, ready, err := bufferGLMKVBHalf(kvbHalf, layer, half, shape, data)
					if err != nil {
						return model.Config{}, nil, err
					}
					if ready {
						merged.Data, err = normalizeCanonicalTensorData(merged.Name, merged.Data, cfg)
						if err != nil {
							return model.Config{}, nil, err
						}
						tensors = append(tensors, merged)
					}
					continue
				}
			}
		}
		if !qwenMTPHandled {
			var ok bool
			name, ok = CanonicalTensorNameArch(info.Name, cfg.ModelType)
			if !ok {
				return model.Config{}, nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
			}
		}
		shape, data, err := s.shapeAndTensorF32(info)
		if err != nil {
			return model.Config{}, nil, err
		}
		if qwenMTPHandled {
			if err := validateQwen35MTPShape(name, shape, cfg); err != nil {
				return model.Config{}, nil, err
			}
		}
		data, err = normalizeCanonicalTensorData(name, data, cfg)
		if err != nil {
			return model.Config{}, nil, err
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: shape, Data: data})
		if qwenMTPHandled {
			qwenMTPSeen[name] = true
		}
	}
	if err := glmKVBUnpaired(kvbHalf); err != nil {
		return model.Config{}, nil, err
	}
	if err := validateQwen35MTPMaterialized(qwenMTPSeen); err != nil {
		return model.Config{}, nil, err
	}
	return cfg, tensors, nil
}

// shapeAndBytes returns a GGUF tensor's model shape and its still-quantized on-disk payload
// bytes, reading the raw bytes via TensorBytes. It is the shared shape+read prelude of the
// tensor handlers that need the raw payload (batched-expert split, canonical map) and of
// dequantGGUFShapeF32.
func (s *WeightSource) shapeAndBytes(info TensorInfo) ([]int, []byte, error) {
	shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
	if err != nil {
		return nil, nil, err
	}
	raw, _, err := s.TensorBytes(info.Name)
	if err != nil {
		return nil, nil, err
	}
	return shape, raw, nil
}

// dequantGGUFShapeF32 returns a GGUF tensor's model shape and its dequantized f32 payload,
// reading the still-quantized bytes via TensorBytes and dequantizing them. It is the shared
// shape+read+dequant prelude of QuantModelProfile's glm_moe_dsa split paths.
func (s *WeightSource) dequantGGUFShapeF32(info TensorInfo) ([]int, []float32, error) {
	shape, raw, err := s.shapeAndBytes(info)
	if err != nil {
		return nil, nil, err
	}
	data, err := dequantF32(info, raw)
	if err != nil {
		return nil, nil, err
	}
	return shape, data, nil
}

// shapeAndTensorF32 returns a GGUF tensor's model shape and its dequantized f32 payload via
// the TensorF32 path. It is the shared shape+TensorF32 prelude of F32Tensors' tensor handlers.
func (s *WeightSource) shapeAndTensorF32(info TensorInfo) ([]int, []float32, error) {
	shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
	if err != nil {
		return nil, nil, err
	}
	data, _, err := s.TensorF32(info.Name)
	if err != nil {
		return nil, nil, err
	}
	return shape, data, nil
}
