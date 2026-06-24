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

func (s *WeightSource) Tensor(name string) (TensorInfo, bool) {
	i, ok := s.byName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return s.File.Tensors[i], true
}

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

func (s *WeightSource) TensorF32(name string) ([]float32, TensorInfo, error) {
	raw, info, err := s.TensorBytes(name)
	if err != nil {
		return nil, info, err
	}
	out, err := dequantF32(info, raw)
	return out, info, err
}

func (s *WeightSource) Model() (*model.Model, error) {
	cfg, tensors, err := s.F32Tensors()
	if err != nil {
		return nil, err
	}
	return model.NewFromF32Tensors(cfg, tensors)
}

func (s *WeightSource) QuantModel() (*model.Model, error) {
	return s.QuantModelProfile(nil)
}

func (s *WeightSource) QuantModelProfile(p *LoadProfiler) (*model.Model, error) {
	t := loadProfileStart(p)
	cfg, err := s.File.Config()
	loadProfileEnd(p, "gguf_config", t, 0, 0)
	if err != nil {
		return nil, err
	}
	builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	for _, info := range s.File.Tensors {
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
		data, err := dequantF32(info, raw)
		dequantNanos := loadProfileEnd(p, "gguf_dequant", t, int64(len(data))*4, 1)
		if p != nil {
			tt.DequantNanos = dequantNanos
			tt.Values = len(data)
		}
		if err != nil {
			return nil, err
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
		addNanos := loadProfileEnd(p, "quant_builder_add", t, int64(len(data))*4, 1)
		if p != nil {
			tt.AddNanos = addNanos
			tt.TotalNanos = time.Since(tensorStart).Nanoseconds()
			p.recordTensor(tt)
		}
	}
	t = loadProfileStart(p)
	m, err := builder.Build()
	loadProfileEnd(p, "quant_builder_finalize", t, 0, 0)
	if err != nil {
		return nil, err
	}
	return m, nil
}

func (s *WeightSource) F32Tensors() (model.Config, []model.NamedTensorF32, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return model.Config{}, nil, err
	}
	tensors := make([]model.NamedTensorF32, 0, len(s.File.Tensors))
	for _, info := range s.File.Tensors {
		// glm_moe_dsa batched routed experts: one [E,out,in] blob splits 1->E into per-expert
		// canonical tensors. Handled before CanonicalTensorNameArch (which leaves them unmapped).
		if cfg.ModelType == "glm_moe_dsa" {
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
	return cfg, tensors, nil
}
