package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unsafe"
)

// safetensors_quant.go — the memory-lean loader that quantizes the big matmul weights to Q8_0
// AT LOAD TIME and drops their f32 copies, so the dominant weights cost ~1.1 bytes/param
// resident instead of f32's 4. This is the "needed update to run the best possible model on
// this box": a regular Load (f32-resident) + Quantize holds 4+1.1 ≈ 5.1 bytes/param, which
// tops out around a 3B model on 36 GB; the lean load holds ~1.1 for the projections/head and
// keeps only the small f32 tensors (embeddings, norms, q/k/v biases) the Q8 forward path reads
// directly — fitting a 7B-class checkpoint in ~10 GB.
//
// The result is QUANT-ONLY by construction: the f32 forward path would panic on the dropped
// weights (that is the point — it must not hold them), so a Session over this Model must set
// Quant=true. The Q8 numerics are bit-identical to a regular Load+Quantize: the same
// quantizeQ8 over the same in-Go bf16->f32 decode (TestLoadSafetensorsQuantMatchesRegular).

// isQuantWeight reports whether a tensor is one of the big matmul weights the Q8 forward path
// consumes from q8w (the dense-attention projections, MoE/dense FFN matrices, GLM
// DSA MLA/indexer projections, and an untied lm_head). These are the ones decoded,
// quantized, and have their f32 dropped at load. Everything else stays f32 in raw.
func isQuantWeight(name string) bool {
	switch {
	case strings.HasSuffix(name, ".self_attn.q_proj.weight"),
		strings.HasSuffix(name, ".self_attn.k_proj.weight"),
		strings.HasSuffix(name, ".self_attn.v_proj.weight"),
		strings.HasSuffix(name, ".self_attn.o_proj.weight"),
		strings.HasSuffix(name, ".self_attn.q_a_proj.weight"),
		strings.HasSuffix(name, ".self_attn.q_b_proj.weight"),
		strings.HasSuffix(name, ".self_attn.kv_a_proj_with_mqa.weight"),
		strings.HasSuffix(name, ".self_attn.kv_b_proj.weight"),
		strings.HasSuffix(name, ".self_attn.indexer.wq_b.weight"),
		strings.HasSuffix(name, ".self_attn.indexer.wk.weight"),
		strings.HasSuffix(name, ".self_attn.indexer.weights_proj.weight"),
		strings.HasSuffix(name, ".linear_attn.in_proj_qkv.weight"),
		strings.HasSuffix(name, ".linear_attn.in_proj_z.weight"),
		strings.HasSuffix(name, ".linear_attn.in_proj_a.weight"),
		strings.HasSuffix(name, ".linear_attn.in_proj_b.weight"),
		strings.HasSuffix(name, ".linear_attn.out_proj.weight"),
		strings.HasSuffix(name, ".mlp.gate.weight"),
		strings.HasSuffix(name, ".mlp.gate_proj.weight"),
		strings.HasSuffix(name, ".mlp.up_proj.weight"),
		strings.HasSuffix(name, ".mlp.down_proj.weight"),
		strings.Contains(name, ".mlp.experts.") && strings.HasSuffix(name, ".gate_proj.weight"),
		strings.Contains(name, ".mlp.experts.") && strings.HasSuffix(name, ".up_proj.weight"),
		strings.Contains(name, ".mlp.experts.") && strings.HasSuffix(name, ".down_proj.weight"),
		strings.Contains(name, ".mlp.shared_experts.") && strings.HasSuffix(name, ".gate_proj.weight"),
		strings.Contains(name, ".mlp.shared_experts.") && strings.HasSuffix(name, ".up_proj.weight"),
		strings.Contains(name, ".mlp.shared_experts.") && strings.HasSuffix(name, ".down_proj.weight"):
		return true
	}
	return name == "lm_head.weight"
}

// f32View reinterprets an f32-little-endian byte slice as []float32 with no copy (arm64/amd64
// are little-endian; same trick as Model.tensor). Used only to feed quantizeQ8 from the decoded
// bytes before they are dropped.
func f32View(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

type safetensorsFileOpener func(path string) (*safetensorsFile, error)

// LoadSafetensorsQuant loads a single-file HuggingFace .safetensors checkpoint in pure Go,
// streaming one source tensor at a time, and returns a quant-only Model with the big matmul
// weights pre-quantized to Q8_0 and their f32 copies dropped. See the file header for the
// memory rationale. The caller must run sessions with Quant=true; Quantize() is a no-op on the
// result (q8w is already built). For sharded checkpoints (a model.safetensors.index.json) use
// LoadSafetensorsQuantDir, which processes one shard at a time so the whole file set is never
// resident at once.
func LoadSafetensorsQuant(path string, cfg Config, opts ...LoadOption) (*Model, error) {
	return loadSafetensorsQuantFile(path, cfg, openSafetensorsFile, opts...)
}

func loadSafetensorsQuantFile(path string, cfg Config, open safetensorsFileOpener, opts ...LoadOption) (*Model, error) {
	lo := resolveLoadOptions(opts)
	sf, err := open(path)
	if err != nil {
		return nil, err
	}
	defer sf.Close()
	tied := safetensorsTiedHeader(sf.hdr)
	m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	if err := quantizeFileInto(sf, m, tied, &raw, &off, lo); err != nil {
		return nil, err
	}
	m.raw = raw
	if len(m.q8w) == 0 {
		return nil, fmt.Errorf("safetensors: no quantizable weights found (wrong tensor names?)")
	}
	m.initQ8CacheIfComplete()
	return m, nil
}

// LoadSafetensorsQuantDir loads a HuggingFace snapshot directory the memory-lean way. If the dir
// has a model.safetensors.index.json it streams each shard one source tensor at a time, so peak
// memory is the current tensor decode + the growing Q8 store, not a whole shard or file set.
// Without an index it falls back to the single model.safetensors. This is the path that lets a
// 7B-class model load on a 36 GB box.
func LoadSafetensorsQuantDir(dir string, cfg Config, opts ...LoadOption) (*Model, error) {
	return loadSafetensorsQuantDir(dir, cfg, openSafetensorsFile, opts...)
}

func loadSafetensorsQuantDir(dir string, cfg Config, open safetensorsFileOpener, opts ...LoadOption) (*Model, error) {
	lo := resolveLoadOptions(opts)
	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(idxPath); err != nil {
		return loadSafetensorsQuantFile(filepath.Join(dir, "model.safetensors"), cfg, open, opts...)
	}
	shards, weightMap, err := safetensorsIndexShards(idxPath)
	if err != nil {
		return nil, err
	}
	// tied is a whole-model property: the embedding is the LM head iff no shard carries lm_head.
	tied := true
	if _, ok := weightMap["lm_head.weight"]; ok {
		tied = false
	}

	m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	for _, sh := range shards {
		sf, err := open(filepath.Join(dir, sh))
		if err != nil {
			return nil, fmt.Errorf("shard %s: %w", sh, err)
		}
		err = func() error {
			defer sf.Close()
			return quantizeFileInto(sf, m, tied, &raw, &off, lo)
		}()
		if err != nil {
			return nil, fmt.Errorf("shard %s: %w", sh, err)
		}
	}
	m.raw = raw
	if len(m.q8w) == 0 {
		return nil, fmt.Errorf("safetensors: no quantizable weights found across %d shards", len(shards))
	}
	m.initQ8CacheIfComplete()
	return m, nil
}

// QuantBuilder is the loader-neutral quant-on-load builder. Source-format loaders that
// can produce one decoded f32 tensor at a time (GGUF, safetensors shards) use it to
// quantize resident matmul weights immediately and keep only the small f32 tensors.
type QuantBuilder struct {
	m     *Model
	raw   []byte
	off   int
	tied  bool
	built bool
}

// NewQuantBuilder starts a memory-lean model build for already-decoded f32 tensors. tied
// means model.embed_tokens.weight also serves as the LM head and should be quantized for
// the head path while still remaining f32 for embedding lookup.
func NewQuantBuilder(cfg Config, tied bool) *QuantBuilder {
	return &QuantBuilder{
		m:    &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}},
		tied: tied,
	}
}

// AddF32Tensor adds one decoded source tensor. If it is a resident matmul weight the
// builder stores only its Q8_0 copy; otherwise it appends it to the packed f32 blob.
func (b *QuantBuilder) AddF32Tensor(name string, shape []int, data []float32) error {
	if b.built {
		return fmt.Errorf("model: QuantBuilder already built")
	}
	elems, err := tensorShapeElems(name, shape)
	if err != nil {
		return err
	}
	if elems != len(data) {
		return fmt.Errorf("model: tensor %s has %d values, shape wants %d", name, len(data), elems)
	}
	return quantizeDecodedFloatTensorInto(name, shape, data, nil, b.m, b.tied, &b.raw, &b.off)
}

// Build finalizes the Model. The result is quant-only for the big matmul weights; callers
// should use the Q8/cacheless paths that can read q8w for those tensors.
func (b *QuantBuilder) Build() (*Model, error) {
	if b.built {
		return nil, fmt.Errorf("model: QuantBuilder already built")
	}
	b.built = true
	b.m.raw = b.raw
	if len(b.m.q8w) == 0 {
		return nil, fmt.Errorf("model: no quantizable weights found")
	}
	b.m.initQ8CacheIfComplete()
	return b.m, nil
}

// safetensorsTied reports whether a single-file checkpoint's embedding doubles as the LM head
// (no lm_head.weight tensor in the header).
func safetensorsTied(buf []byte) (bool, error) {
	hdr, _, err := parseSafetensorsHeader(buf)
	if err != nil {
		return false, err
	}
	return safetensorsTiedHeader(hdr), nil
}

// quantizeBlobInto decodes one in-memory safetensors blob, quantizing the big matmul weights
// into m.q8w (their f32 dropped) and appending the small f32 tensors (embeddings, norms, biases)
// to *raw + m.manifest. tied says the embedding doubles as the LM head (quantize it too). The
// read-file equivalence tests use this historical whole-blob path; runtime loaders stream via
// quantizeFileInto.
func quantizeBlobInto(buf []byte, m *Model, tied bool, raw *[]byte, off *int) error {
	hdr, dataBase, err := parseSafetensorsHeader(buf)
	if err != nil {
		return err
	}
	names := safetensorsTensorNames(hdr)
	consumed := map[string]bool{}

	for _, name := range names {
		if consumed[name] {
			continue
		}
		handled, err := quantizeMXFP4TensorInto(name, hdr, func(e stEntry) ([]byte, error) {
			return safetensorsBufferBytes(buf, dataBase, e)
		}, m, tied, raw, off, consumed)
		if err != nil {
			return err
		}
		if handled {
			continue
		}
		var e stEntry
		if err := json.Unmarshal(hdr[name], &e); err != nil {
			return fmt.Errorf("safetensors: entry %s: %w", name, err)
		}
		if skipSafetensorsTensor(name, e) {
			continue
		}
		src, err := safetensorsBufferBytes(buf, dataBase, e)
		if err != nil {
			return fmt.Errorf("safetensors: tensor %s: %w", name, err)
		}
		if err := quantizeTensorInto(name, e, src, m, tied, raw, off); err != nil {
			return err
		}
	}
	return nil
}

func quantizeFileInto(sf *safetensorsFile, m *Model, tied bool, raw *[]byte, off *int, lo loadOptions) error {
	names := safetensorsTensorNames(sf.hdr)
	consumed := map[string]bool{}
	for _, name := range names {
		if consumed[name] {
			continue
		}
		// Pipeline-parallel partition: skip tensors whose transformer layer is
		// outside this worker's band. Layer-agnostic tensors (embeddings, final
		// norm, untied lm_head) report layer -1 and are always kept. The default
		// window keeps every layer, so a plain load is unchanged (no-op gate).
		if !lo.window.keepsLayer(tensorLayerForWindow(name)) {
			consumed[name] = true
			continue
		}
		handled, err := quantizeMXFP4TensorInto(name, sf.hdr, sf.tensorBytes, m, tied, raw, off, consumed)
		if err != nil {
			return err
		}
		if handled {
			continue
		}
		var e stEntry
		if err := json.Unmarshal(sf.hdr[name], &e); err != nil {
			return fmt.Errorf("safetensors: entry %s: %w", name, err)
		}
		if skipSafetensorsTensor(name, e) {
			continue
		}
		src, err := sf.tensorBytes(e)
		if err != nil {
			return fmt.Errorf("safetensors: tensor %s: %w", name, err)
		}
		if err := quantizeTensorInto(name, e, src, m, tied, raw, off); err != nil {
			return err
		}
	}
	return nil
}

func quantizeMXFP4TensorInto(
	name string,
	hdr map[string]json.RawMessage,
	tensorBytes func(stEntry) ([]byte, error),
	m *Model,
	tied bool,
	raw *[]byte,
	off *int,
	consumed map[string]bool,
) (bool, error) {
	if !strings.HasSuffix(name, "_blocks") {
		return false, nil
	}
	base := strings.TrimSuffix(name, "_blocks")
	scaleName := base + "_scales"
	scaleRaw, ok := hdr[scaleName]
	if !ok {
		return false, nil
	}
	var blockEntry, scaleEntry stEntry
	if err := json.Unmarshal(hdr[name], &blockEntry); err != nil {
		return true, fmt.Errorf("safetensors: entry %s: %w", name, err)
	}
	if err := json.Unmarshal(scaleRaw, &scaleEntry); err != nil {
		return true, fmt.Errorf("safetensors: entry %s: %w", scaleName, err)
	}
	blocks, err := tensorBytes(blockEntry)
	if err != nil {
		return true, fmt.Errorf("safetensors: tensor %s: %w", name, err)
	}
	scales, err := tensorBytes(scaleEntry)
	if err != nil {
		return true, fmt.Errorf("safetensors: tensor %s: %w", scaleName, err)
	}
	f32, shape, err := decodeMXFP4Blocks(base, blockEntry, scaleEntry, blocks, scales)
	if err != nil {
		return true, err
	}
	consumed[name] = true
	consumed[scaleName] = true
	return true, quantizeDecodedTensorInto(base, shape, f32, m, tied, raw, off)
}

func quantizeTensorInto(name string, e stEntry, src []byte, m *Model, tied bool, raw *[]byte, off *int) error {
	fb, err := decodeSafetensorF32(name, e, src)
	if err != nil {
		return err
	}
	return quantizeDecodedTensorInto(name, e.Shape, fb, m, tied, raw, off)
}

func quantizeDecodedTensorInto(name string, shape []int, fb []byte, m *Model, tied bool, raw *[]byte, off *int) error {
	return quantizeDecodedFloatTensorInto(name, shape, f32View(fb), fb, m, tied, raw, off)
}

func quantizeDecodedFloatTensorInto(name string, shape []int, f32 []float32, fb []byte, m *Model, tied bool, raw *[]byte, off *int) error {
	var keep bool
	name, keep = quantSourceTensorName(m.Cfg, name)
	if !keep {
		return nil
	}

	if handled, err := quantizeFusedProjectionTensorInto(name, shape, f32, m); handled || err != nil {
		return err
	}
	if handled, err := quantizeSourceMoETensorInto(name, shape, f32, m, raw, off); handled || err != nil {
		return err
	}

	if isQuantWeight(name) && len(shape) == 2 {
		m.q8w[name] = quantizeQ8(f32, shape[0], shape[1])
		return nil // f32 dropped — the memory win
	}

	if fb != nil {
		m.manifest[name] = tensorMeta{Dtype: "f32", Shape: append([]int(nil), shape...), Offset: *off, Nbytes: len(fb)}
		*raw = append(*raw, fb...)
		*off += len(fb)
	} else {
		appendQuantF32Tensor(m, raw, off, name, shape, f32)
	}
	// Tied models: the embedding matrix is also the LM head — keep it f32 in raw (for the
	// row-gather lookup) AND quantize it into q8w (headName() resolves to this key).
	if tied && name == "model.embed_tokens.weight" && len(shape) == 2 {
		m.q8w[name] = quantizeQ8(f32, shape[0], shape[1])
	}
	return nil
}

func quantSourceTensorName(cfg Config, name string) (string, bool) {
	// GLM-family (glm_moe_dsa) and MiniMax-M3 checkpoints carry a multimodal vision
	// encoder and an MTP head the quantized text forward never reads; drop them at the
	// source-name gate exactly as skipLoadTensor does for the f32 path. The Qwen3.5
	// block below additionally remaps linear-attention tensors, which does not apply here.
	if (cfg.isGLM() || cfg.isMiniMax()) && (strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.")) {
		return "", false
	}
	if !cfg.IsQwen35Hybrid() {
		return name, true
	}
	const lm = "model.language_model."
	switch {
	case strings.HasPrefix(name, lm):
		name = "model." + name[len(lm):]
	case strings.HasPrefix(name, "model.visual."), strings.HasPrefix(name, "mtp."):
		return "", false
	}
	layer, suffix, ok := parseLayerTensorSuffix(name)
	if !ok || !cfg.isLinearAttnLayer(layer) {
		return name, true
	}
	switch suffix {
	case suffixQKVProj:
		return layerPrefix(layer) + "linear_attn.in_proj_qkv.weight", true
	case "self_attn.q_gate_proj.weight":
		return layerPrefix(layer) + "linear_attn.in_proj_z.weight", true
	default:
		return name, true
	}
}

func quantizeFusedProjectionTensorInto(name string, shape []int, data []float32, m *Model) (bool, error) {
	layer, suffix, ok := parseLayerTensorSuffix(name)
	if !ok || layer < 0 || layer >= m.Cfg.NumLayers {
		return false, nil
	}
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	qRows, kRows, vRows := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	switch suffix {
	case suffixQKVProj:
		if err := requireSafetensorShape(name, shape, []int{qRows + kRows + vRows, H}); err != nil {
			return true, err
		}
		prefix := layerPrefix(layer)
		parts := []struct {
			name string
			rows int
			off  int
		}{
			{prefix + suffixQProj, qRows, 0},
			{prefix + suffixKProj, kRows, qRows},
			{prefix + suffixVProj, vRows, qRows + kRows},
		}
		if anyQ8Present(m, parts[0].name, parts[1].name, parts[2].name) {
			return true, fmt.Errorf("safetensors: cannot split %s: one or more q/k/v components already exist", name)
		}
		for _, part := range parts {
			m.q8w[part.name] = quantizeQ8(data[part.off*H:(part.off+part.rows)*H], part.rows, H)
		}
		return true, nil
	case suffixGateUpProj:
		if err := requireSafetensorShape(name, shape, []int{2 * I, H}); err != nil {
			return true, err
		}
		prefix := layerPrefix(layer)
		gateName, upName := prefix+suffixGateProj, prefix+suffixUpProj
		if anyQ8Present(m, gateName, upName) {
			return true, fmt.Errorf("safetensors: cannot split %s: gate/up component already exists", name)
		}
		m.q8w[gateName] = quantizeQ8(data[:I*H], I, H)
		m.q8w[upName] = quantizeQ8(data[I*H:2*I*H], I, H)
		return true, nil
	default:
		return false, nil
	}
}

func quantizeSourceMoETensorInto(name string, shape []int, data []float32, m *Model, raw *[]byte, off *int) (bool, error) {
	layer, suffix, ok := parseLayerTensorSuffix(name)
	if !ok || layer < 0 || layer >= m.Cfg.NumLayers || m.Cfg.NumExperts <= 0 {
		return false, nil
	}
	if handled, err := quantizeGPTOSSSourceMoETensor(name, layer, suffix, shape, data, m, raw, off); handled || err != nil {
		return handled, err
	}
	if handled, err := quantizeMixtralBlockSparseMoETensor(name, layer, suffix, shape, data, m); handled || err != nil {
		return handled, err
	}
	return false, nil
}

func quantizeGPTOSSSourceMoETensor(name string, layer int, suffix string, shape []int, data []float32, m *Model, raw *[]byte, off *int) (bool, error) {
	cfg := m.Cfg
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	switch suffix {
	case "mlp.router.weight":
		return true, quantizeAliasedQ8(name, routerName(layer), shape, []int{E, H}, data, m)
	case "mlp.router.bias":
		return true, appendAliasedF32(name, routerBiasName(layer), shape, []int{E}, data, m, raw, off)
	case "mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj.weight":
		if err := requireSafetensorShape(name, shape, []int{E, H, 2 * I}); err != nil {
			return true, err
		}
		for e := 0; e < E; e++ {
			gateName := expertName(layer, e, "gate_proj.weight")
			upName := expertName(layer, e, "up_proj.weight")
			if anyQ8Present(m, gateName, upName) {
				return true, fmt.Errorf("safetensors: cannot materialize %s: expert %d gate/up component already exists", name, e)
			}
			gate := make([]float32, I*H)
			up := make([]float32, I*H)
			for i := 0; i < I; i++ {
				for h := 0; h < H; h++ {
					src := ((e*H+h)*2*I + 2*i)
					gate[i*H+h] = data[src]
					up[i*H+h] = data[src+1]
				}
			}
			m.q8w[gateName] = quantizeQ8(gate, I, H)
			m.q8w[upName] = quantizeQ8(up, I, H)
		}
		return true, nil
	case "mlp.experts.down_proj", "mlp.experts.down_proj.weight":
		if err := requireSafetensorShape(name, shape, []int{E, I, H}); err != nil {
			return true, err
		}
		for e := 0; e < E; e++ {
			downName := expertName(layer, e, "down_proj.weight")
			if anyQ8Present(m, downName) {
				return true, fmt.Errorf("safetensors: cannot materialize %s: expert %d down component already exists", name, e)
			}
			down := make([]float32, H*I)
			for h := 0; h < H; h++ {
				for i := 0; i < I; i++ {
					down[h*I+i] = data[(e*I+i)*H+h]
				}
			}
			m.q8w[downName] = quantizeQ8(down, H, I)
		}
		return true, nil
	case "mlp.experts.gate_up_proj_bias", "mlp.experts.gate_up_proj.bias":
		if err := requireSafetensorShape(name, shape, []int{E, 2 * I}); err != nil {
			return true, err
		}
		for e := 0; e < E; e++ {
			gateName := expertName(layer, e, "gate_proj.bias")
			upName := expertName(layer, e, "up_proj.bias")
			if anyF32Present(m, gateName, upName) {
				return true, fmt.Errorf("safetensors: cannot materialize %s: expert %d gate/up bias already exists", name, e)
			}
			gate := make([]float32, I)
			up := make([]float32, I)
			for i := 0; i < I; i++ {
				src := e*2*I + 2*i
				gate[i] = data[src]
				up[i] = data[src+1]
			}
			appendQuantF32Tensor(m, raw, off, gateName, []int{I}, gate)
			appendQuantF32Tensor(m, raw, off, upName, []int{I}, up)
		}
		return true, nil
	case "mlp.experts.down_proj_bias", "mlp.experts.down_proj.bias":
		if err := requireSafetensorShape(name, shape, []int{E, H}); err != nil {
			return true, err
		}
		for e := 0; e < E; e++ {
			downName := expertName(layer, e, "down_proj.bias")
			if anyF32Present(m, downName) {
				return true, fmt.Errorf("safetensors: cannot materialize %s: expert %d down bias already exists", name, e)
			}
			down := make([]float32, H)
			copy(down, data[e*H:(e+1)*H])
			appendQuantF32Tensor(m, raw, off, downName, []int{H}, down)
		}
		return true, nil
	default:
		return false, nil
	}
}

func quantizeMixtralBlockSparseMoETensor(name string, layer int, suffix string, shape []int, data []float32, m *Model) (bool, error) {
	cfg := m.Cfg
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	if suffix == "block_sparse_moe.gate.weight" {
		return true, quantizeAliasedQ8(name, routerName(layer), shape, []int{E, H}, data, m)
	}
	rest, ok := strings.CutPrefix(suffix, "block_sparse_moe.experts.")
	if !ok {
		return false, nil
	}
	expertText, weightSuffix, ok := strings.Cut(rest, ".")
	if !ok {
		return false, nil
	}
	expert, err := strconv.Atoi(expertText)
	if err != nil || expert < 0 || expert >= E {
		return false, nil
	}
	switch weightSuffix {
	case "w1.weight":
		return true, quantizeAliasedQ8(name, expertName(layer, expert, "gate_proj.weight"), shape, []int{I, H}, data, m)
	case "w2.weight":
		return true, quantizeAliasedQ8(name, expertName(layer, expert, "down_proj.weight"), shape, []int{H, I}, data, m)
	case "w3.weight":
		return true, quantizeAliasedQ8(name, expertName(layer, expert, "up_proj.weight"), shape, []int{I, H}, data, m)
	default:
		return false, nil
	}
}

func parseLayerTensorSuffix(name string) (int, string, bool) {
	rest, ok := strings.CutPrefix(name, "model.layers.")
	if !ok {
		return 0, "", false
	}
	layerText, suffix, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, "", false
	}
	layer, err := strconv.Atoi(layerText)
	if err != nil {
		return 0, "", false
	}
	return layer, suffix, true
}

func quantizeAliasedQ8(source, canonical string, got, want []int, data []float32, m *Model) error {
	if _, exists := m.q8w[canonical]; exists {
		return nil
	}
	if err := requireSafetensorShape(source, got, want); err != nil {
		return err
	}
	m.q8w[canonical] = quantizeQ8(data, want[0], want[1])
	return nil
}

func appendAliasedF32(source, canonical string, got, want []int, data []float32, m *Model, raw *[]byte, off *int) error {
	if _, exists := m.manifest[canonical]; exists {
		return nil
	}
	if err := requireSafetensorShape(source, got, want); err != nil {
		return err
	}
	appendQuantF32Tensor(m, raw, off, canonical, want, data)
	return nil
}

func appendQuantF32Tensor(m *Model, raw *[]byte, off *int, name string, shape []int, data []float32) {
	appendF32Tensor(m.manifest, raw, name, shape, data)
	*off = len(*raw)
}

func anyQ8Present(m *Model, names ...string) bool {
	for _, name := range names {
		if _, ok := m.q8w[name]; ok {
			return true
		}
	}
	return false
}

func anyF32Present(m *Model, names ...string) bool {
	for _, name := range names {
		if _, ok := m.manifest[name]; ok {
			return true
		}
	}
	return false
}

func requireSafetensorShape(name string, got, want []int) error {
	if !sameShape(got, want) {
		return fmt.Errorf("safetensors: tensor %s has shape %v, want %v", name, got, want)
	}
	return nil
}
