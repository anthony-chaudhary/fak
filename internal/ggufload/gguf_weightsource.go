package ggufload

import (
	"fmt"
	"io"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// LoadModel loads a GGUF checkpoint through the default dequant-to-f32 path and returns a
// regular in-kernel model.Model. GGUF tensor names are normalized to the canonical HF-Llama
// names that internal/model already consumes.
func LoadModel(path string) (*model.Model, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.Model()
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

// TensorBytes reads a named tensor's raw (still-quantized) payload bytes from the
// shard reader that holds it, bounds-checking the offset and length against the file.
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
	var dequantBuf []float32
	// glm_moe_dsa MLA KV-b merge buffer: the split attn_k_b / attn_v_b for a layer may not be
	// adjacent in the tensor stream, so buffer the first half seen per layer and emit the merged
	// kv_b_proj when its partner arrives (mergeGLMMoeDsaKVB). See gguf_glm_tensors.go.
	kvbHalf := map[int]glmKVBHalf{}
	p.SetTotal(len(s.File.Tensors)) // arm the progress reporter (no-op on nil / unset Progress)
	for _, info := range s.File.Tensors {
		p.Tick(tensorOnDiskBytes(info)) // one GGUF tensor consumed -> advance the % status
		if cfg.ModelType == "glm_moe_dsa" {
			// Drop the MTP ("nextn") speculative head + any vision tower the text forward never
			// reads (llama.cpp ignores them too), before canonical mapping would reject them.
			if glmMoeDsaSkipGGUFTensor(info.Name) {
				continue
			}
			if layer, half, ok := glmMoeDsaSplitKVB(info.Name); ok {
				shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
				if err != nil {
					return nil, err
				}
				raw, _, err := s.TensorBytes(info.Name)
				if err != nil {
					return nil, err
				}
				data, err := dequantF32(info, raw)
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
		// glm_moe_dsa batched routed experts: split the [E,out,in] blob 1->E into per-expert
		// canonical tensors and add each (the quant builder narrows the 2-D matmul weights as
		// usual). Handled before CanonicalTensorNameArch, which leaves these unmapped.
		if cfg.ModelType == "glm_moe_dsa" {
			if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
				if err != nil {
					return nil, err
				}
				raw, _, err := s.TensorBytes(info.Name)
				if err != nil {
					return nil, err
				}
				data, err := dequantF32(info, raw)
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
		name, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
		}
		if p != nil {
			tt.CanonicalName = name
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return nil, err
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
		data, err := dequantF32Into(dequantBuf, info, raw)
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
		// dequant arena so the next tensor reuses it.
		dequantBuf = data

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
// tensors (glm_moe_dsa batched experts split 1->E). It is the f32 path's Model builds on.
func (s *WeightSource) F32Tensors() (model.Config, []model.NamedTensorF32, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return model.Config{}, nil, err
	}
	tensors := make([]model.NamedTensorF32, 0, len(s.File.Tensors))
	kvbHalf := map[int]glmKVBHalf{} // MLA KV-b 2->1 merge buffer (see QuantModelProfile)
	for _, info := range s.File.Tensors {
		// glm_moe_dsa batched routed experts: one [E,out,in] blob splits 1->E into per-expert
		// canonical tensors. Handled before CanonicalTensorNameArch (which leaves them unmapped).
		if cfg.ModelType == "glm_moe_dsa" {
			// Drop the MTP ("nextn") head + any vision tower the text forward never reads.
			if glmMoeDsaSkipGGUFTensor(info.Name) {
				continue
			}
			if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
				if err != nil {
					return model.Config{}, nil, err
				}
				data, _, err := s.TensorF32(info.Name)
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
			if layer, half, ok := glmMoeDsaSplitKVB(info.Name); ok {
				shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
				if err != nil {
					return model.Config{}, nil, err
				}
				data, _, err := s.TensorF32(info.Name)
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
		name, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return model.Config{}, nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return model.Config{}, nil, err
		}
		data, _, err := s.TensorF32(info.Name)
		if err != nil {
			return model.Config{}, nil, err
		}
		data, err = normalizeCanonicalTensorData(name, data, cfg)
		if err != nil {
			return model.Config{}, nil, err
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	if err := glmKVBUnpaired(kvbHalf); err != nil {
		return model.Config{}, nil, err
	}
	return cfg, tensors, nil
}
