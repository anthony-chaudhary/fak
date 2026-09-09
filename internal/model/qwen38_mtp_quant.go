package model

import (
	"fmt"
	"strings"
)

// Qwen38MTPTensorFormat names either an individual tensor's retained storage or
// the aggregate execution family selected by an admitted layout.
type Qwen38MTPTensorFormat string

const (
	Qwen38MTPFormatF32  Qwen38MTPTensorFormat = "F32"
	Qwen38MTPFormatBF16 Qwen38MTPTensorFormat = "BF16"
	Qwen38MTPFormatQ8   Qwen38MTPTensorFormat = "Q8_0"
	Qwen38MTPFormatQ4K  Qwen38MTPTensorFormat = "Q4_K"
	Qwen38MTPFormatQ6K  Qwen38MTPTensorFormat = "Q6_K"
	Qwen38MTPFormatNone Qwen38MTPTensorFormat = ""
)

// Qwen38MTPTensorLayout is a read-only inventory derived from the model's real
// resident stores. TensorTypes is exact per-tensor evidence; Format selects the
// compatible execution family.
type Qwen38MTPTensorLayout struct {
	Format      Qwen38MTPTensorFormat `json:"format"`
	TensorTypes map[string]string     `json:"tensor_types"`
}

var qwen38MTPMatrixTensors = [...]string{
	"mtp.fc.weight",
	"mtp.layers.0.self_attn.q_proj.weight",
	"mtp.layers.0.self_attn.k_proj.weight",
	"mtp.layers.0.self_attn.v_proj.weight",
	"mtp.layers.0.self_attn.o_proj.weight",
	"mtp.layers.0.mlp.gate_proj.weight",
	"mtp.layers.0.mlp.up_proj.weight",
	"mtp.layers.0.mlp.down_proj.weight",
}

var qwen38MTPNormTensors = [...]string{
	"mtp.pre_fc_norm_embedding.weight",
	"mtp.pre_fc_norm_hidden.weight",
	"mtp.norm.weight",
	"mtp.layers.0.input_layernorm.weight",
	"mtp.layers.0.post_attention_layernorm.weight",
	"mtp.layers.0.self_attn.q_norm.weight",
	"mtp.layers.0.self_attn.k_norm.weight",
}

// qwen38MTPQ4KMArtifactMatrixTypes is the canonicalized tensor inventory of
// unsloth/Qwen3.8-27B-GGUF's Q4_K_M artifact. The aggregate layout remains
// Qwen38MTPFormatQ4K because that selects the mixed resident Q4_K session
// kernel; TensorTypes preserves the exact per-tensor storage evidence.
var qwen38MTPQ4KMArtifactMatrixTypes = map[string]Qwen38MTPTensorFormat{
	"mtp.fc.weight":                        Qwen38MTPFormatQ8,
	"mtp.layers.0.self_attn.q_proj.weight": Qwen38MTPFormatQ4K,
	"mtp.layers.0.self_attn.k_proj.weight": Qwen38MTPFormatQ4K,
	"mtp.layers.0.self_attn.v_proj.weight": Qwen38MTPFormatQ6K,
	"mtp.layers.0.self_attn.o_proj.weight": Qwen38MTPFormatQ4K,
	"mtp.layers.0.mlp.gate_proj.weight":    Qwen38MTPFormatQ4K,
	"mtp.layers.0.mlp.up_proj.weight":      Qwen38MTPFormatQ4K,
	"mtp.layers.0.mlp.down_proj.weight":    Qwen38MTPFormatQ6K,
}

func isQwen38MTPMatrixTensor(name string) bool {
	for _, candidate := range qwen38MTPMatrixTensors {
		if name == candidate {
			return true
		}
	}
	return false
}

// Qwen38MTPTensorLayout reports the actual retained MTP precision. It admits
// exactly two closed layout families:
//
//   - the compatibility layout whose matrices are uniformly F32 or BF16;
//   - the exact Qwen3.8-27B-Q4_K_M inventory: fc Q8_0; q/k/o/gate/up
//     Q4_K; v/down and the canonical lm_head (GGUF output.weight) Q6_K;
//     every MTP norm F32.
//
// An artifact label is deliberately irrelevant. Any other mixture, duplicate
// representation, or malformed resident span is precision_unsupported at the
// eligibility boundary.
func (m *Model) Qwen38MTPTensorLayout() (Qwen38MTPTensorLayout, error) {
	layout, present, err := m.qwen38MTPTensorLayout()
	if !present && err == nil {
		err = qwen35MTPStateError("weight lookup", "complete retained Qwen3.8 MTP tensor set", "missing")
	}
	return layout, err
}

func (m *Model) qwen38MTPTensorLayout() (Qwen38MTPTensorLayout, bool, error) {
	layout := Qwen38MTPTensorLayout{TensorTypes: make(map[string]string, len(qwen35MTPRequiredTensors)+1)}
	if m == nil {
		return layout, false, qwen35MTPStateError("model", "non-nil model", "nil")
	}
	if !m.Cfg.isQwen35TextFamily() || m.Cfg.NumMTPLayers() != 1 || m.Cfg.MTPUseDedicatedEmbeddings {
		return layout, qwen38MTPStoragePresent(m), qwen35MTPStateError(
			"model",
			"eligible one-layer shared-embedding Qwen3.8 MTP model",
			"ineligible config",
		)
	}
	expected, err := qwen35MTPExpectedShapes(m.Cfg)
	if err != nil {
		return layout, qwen38MTPStoragePresent(m), err
	}

	present := qwen38MTPStoragePresent(m)
	if !present {
		return layout, false, nil
	}

	matrixTypes := make(map[string]Qwen38MTPTensorFormat, len(qwen38MTPMatrixTensors))
	uniformFormat := Qwen38MTPFormatNone
	uniform := true
	for _, name := range qwen38MTPMatrixTensors {
		format, err := m.qwen38MTPMatrixFormat(name, expected[name])
		if err != nil {
			return layout, true, err
		}
		matrixTypes[name] = format
		layout.TensorTypes[name] = string(format)
		if uniformFormat == Qwen38MTPFormatNone {
			uniformFormat = format
		} else if uniformFormat != format {
			uniform = false
		}
	}

	exactQ4KM := true
	for name, want := range qwen38MTPQ4KMArtifactMatrixTypes {
		if matrixTypes[name] != want {
			exactQ4KM = false
			break
		}
	}
	compatibility := uniform && (uniformFormat == Qwen38MTPFormatF32 || uniformFormat == Qwen38MTPFormatBF16)
	if !compatibility && !exactQ4KM {
		for _, name := range qwen38MTPMatrixTensors {
			want := qwen38MTPQ4KMArtifactMatrixTypes[name]
			if matrixTypes[name] != want {
				return layout, true, &Qwen35MTPForwardError{
					Stage:  "weight precision",
					Tensor: name,
					Want:   string(want) + " (exact Qwen3.8-27B-Q4_K_M inventory)",
					Got:    string(matrixTypes[name]),
				}
			}
		}
	}

	for _, name := range qwen38MTPNormTensors {
		wantShape := expected[name]
		meta, ok := m.manifest[name]
		if !ok {
			return layout, true, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: name, Want: "F32 norm", Got: "missing"}
		}
		allowBF16 := compatibility
		if !strings.EqualFold(meta.Dtype, "F32") && !(allowBF16 && strings.EqualFold(meta.Dtype, "BF16")) {
			want := "F32 norm"
			if allowBF16 {
				want = "F32 or BF16 norm"
			}
			return layout, true, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: want, Got: meta.Dtype}
		}
		if err := validateQwen38MTPF32OrBF16Meta(m, name, meta, wantShape); err != nil {
			return layout, true, err
		}
		if m.q4kw[name] != nil || m.q8w[name] != nil || m.kqw[name] != nil {
			return layout, true, &Qwen35MTPForwardError{Stage: "weight precision", Tensor: name, Want: "one F32 or BF16 norm representation", Got: "duplicate or quantized norm representation"}
		}
		layout.TensorTypes[name] = strings.ToUpper(meta.Dtype)
	}

	if exactQ4KM {
		headName := "lm_head.weight" // canonical form of the artifact's output.weight
		headFormat, err := m.qwen38MTPMatrixFormat(headName, []int{m.Cfg.VocabSize, m.Cfg.HiddenSize})
		if err != nil {
			return layout, true, err
		}
		layout.TensorTypes[headName] = string(headFormat)
		if headFormat != Qwen38MTPFormatQ6K {
			return layout, true, &Qwen35MTPForwardError{
				Stage:  "weight precision",
				Tensor: headName,
				Want:   "Q6_K (canonical output.weight in exact Qwen3.8-27B-Q4_K_M inventory)",
				Got:    string(headFormat),
			}
		}
		layout.Format = Qwen38MTPFormatQ4K
	} else {
		layout.Format = uniformFormat
	}
	return layout, true, nil
}

func qwen38MTPStoragePresent(m *Model) bool {
	if m == nil {
		return false
	}
	for name := range m.manifest {
		if strings.HasPrefix(name, "mtp.") {
			return true
		}
	}
	for _, store := range []map[string]*q4kTensor{m.q4kw} {
		for name := range store {
			if strings.HasPrefix(name, "mtp.") {
				return true
			}
		}
	}
	for name := range m.q8w {
		if strings.HasPrefix(name, "mtp.") {
			return true
		}
	}
	for name := range m.kqw {
		if strings.HasPrefix(name, "mtp.") {
			return true
		}
	}
	return false
}

func (m *Model) qwen38MTPMatrixFormat(name string, wantShape []int) (Qwen38MTPTensorFormat, error) {
	meta, hasF32 := m.manifest[name]
	q4 := m.q4kw[name]
	q8 := m.q8w[name]
	kq := m.kqw[name]

	count := 0
	if hasF32 {
		count++
	}
	if q4 != nil {
		count++
	}
	if q8 != nil {
		count++
	}
	if kq != nil {
		count++
	}
	if count == 0 {
		return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: name, Want: "F32, BF16, Q8_0, Q4_K, or Q6_K", Got: "missing"}
	}
	if count != 1 {
		return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight precision", Tensor: name, Want: "one retained representation", Got: "mixed or duplicate representations"}
	}

	switch {
	case hasF32:
		if strings.EqualFold(meta.Dtype, "BF16") {
			if err := validateQwen38MTPBF16Meta(m, name, meta, wantShape); err != nil {
				return Qwen38MTPFormatNone, err
			}
			return Qwen38MTPFormatBF16, nil
		}
		if !strings.EqualFold(meta.Dtype, "F32") {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32, BF16, or Q4_K", Got: meta.Dtype}
		}
		if err := validateQwen38MTPF32Meta(m, name, meta, wantShape); err != nil {
			return Qwen38MTPFormatNone, err
		}
		return Qwen38MTPFormatF32, nil
	case q4 != nil:
		if q4.out != wantShape[0] || q4.in != wantShape[1] || q4.nblk != wantShape[1]/qkK {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprintf("[%d %d]", q4.out, q4.in)}
		}
		wantBytes := q4.out * q4.nblk * q4kBlockBytes
		residentOK := len(q4.raw) == wantBytes
		lazyOK := q4.lazy != nil && q4.lazy.Reader != nil && q4.lazy.Bytes == wantBytes
		if !residentOK && !lazyOK {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{
				Stage:  "weight storage",
				Tensor: name,
				Want:   fmt.Sprintf("%d resident or checkpoint-backed Q4_K bytes", wantBytes),
				Got:    fmt.Sprintf("resident=%d lazy=%v", len(q4.raw), q4.lazy != nil),
			}
		}
		return Qwen38MTPFormatQ4K, nil
	case q8 != nil:
		wantNblk := wantShape[1] / qBlk
		if wantShape[1]%qBlk != 0 || q8.out != wantShape[0] || q8.in != wantShape[1] || q8.nblk != wantNblk {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprintf("[%d %d]", q8.out, q8.in)}
		}
		if len(q8.q) != q8.out*q8.in || len(q8.d) != q8.out*q8.nblk {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{
				Stage:  "weight storage",
				Tensor: name,
				Want:   fmt.Sprintf("%d Q8_0 codes and %d scales", q8.out*q8.in, q8.out*q8.nblk),
				Got:    fmt.Sprintf("codes=%d scales=%d", len(q8.q), len(q8.d)),
			}
		}
		return Qwen38MTPFormatQ8, nil
	default:
		if kq.kind != kindQ6K {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32, BF16, Q8_0, Q4_K, or Q6_K", Got: kq.kind.String()}
		}
		wantNblk := wantShape[1] / kindQ6K.blockWeights()
		if wantShape[1]%kindQ6K.blockWeights() != 0 || kq.out != wantShape[0] || kq.in != wantShape[1] || kq.nblk != wantNblk {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprintf("[%d %d]", kq.out, kq.in)}
		}
		wantBytes := kq.out * kq.nblk * kindQ6K.blockBytes()
		if len(kq.raw) != wantBytes {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{
				Stage:  "weight storage",
				Tensor: name,
				Want:   fmt.Sprintf("%d resident Q6_K bytes", wantBytes),
				Got:    fmt.Sprintf("resident=%d", len(kq.raw)),
			}
		}
		return Qwen38MTPFormatQ6K, nil
	}
}

func validateQwen38MTPF32Meta(m *Model, name string, meta tensorMeta, wantShape []int) error {
	if !sameIntShape(meta.Shape, wantShape) {
		return &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprint(meta.Shape)}
	}
	wantBytes, ok := qwen35MTPF32Bytes(wantShape)
	if !ok || meta.Nbytes != wantBytes || meta.Offset < 0 || meta.Offset > len(m.raw)-meta.Nbytes {
		return &Qwen35MTPForwardError{
			Stage:  "weight storage",
			Tensor: name,
			Want:   fmt.Sprintf("%d bytes inside model payload", wantBytes),
			Got:    fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(m.raw)),
		}
	}
	return nil
}

func validateQwen38MTPBF16Meta(m *Model, name string, meta tensorMeta, wantShape []int) error {
	if !sameIntShape(meta.Shape, wantShape) {
		return &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprint(meta.Shape)}
	}
	elems, err := tensorShapeElems(name, wantShape)
	if err != nil {
		return err
	}
	wantBytes := elems * 2
	if meta.Nbytes != wantBytes && meta.Nbytes != elems*4 {
		return &Qwen35MTPForwardError{
			Stage:  "weight storage",
			Tensor: name,
			Want:   fmt.Sprintf("%d or %d bytes inside model payload", wantBytes, elems*4),
			Got:    fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(m.raw)),
		}
	}
	if meta.Offset < 0 || meta.Offset > len(m.raw)-meta.Nbytes {
		return &Qwen35MTPForwardError{
			Stage:  "weight storage",
			Tensor: name,
			Want:   "valid offset inside model payload",
			Got:    fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(m.raw)),
		}
	}
	return nil
}

func validateQwen38MTPF32OrBF16Meta(m *Model, name string, meta tensorMeta, wantShape []int) error {
	if strings.EqualFold(meta.Dtype, "BF16") {
		return validateQwen38MTPBF16Meta(m, name, meta, wantShape)
	}
	return validateQwen38MTPF32Meta(m, name, meta, wantShape)
}

func (f *Qwen35MTPForward) qwen38MTPFuse(priorHidden, currentEmbedding []float32) ([]float32, error) {
	if f == nil || f.target == nil || f.draft == nil {
		return nil, qwen35MTPStateError("forward state", "initialized Qwen35MTPForward", "nil or incomplete")
	}
	if f.tensorFormat != Qwen38MTPFormatQ4K {
		return f.target.Qwen35MTPFuse(priorHidden, currentEmbedding)
	}
	h := f.target.Cfg.HiddenSize
	if len(priorHidden) != h {
		return nil, qwen35MTPStateError("prior hidden shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(priorHidden)))
	}
	if len(currentEmbedding) != h {
		return nil, qwen35MTPStateError("current embedding shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(currentEmbedding)))
	}
	hiddenNorm, err := f.target.qwen35MTPF32Tensor("mtp.pre_fc_norm_hidden.weight", []int{h})
	if err != nil {
		return nil, err
	}
	embeddingNorm, err := f.target.qwen35MTPF32Tensor("mtp.pre_fc_norm_embedding.weight", []int{h})
	if err != nil {
		return nil, err
	}
	eps := float32(f.target.Cfg.RMSNormEps)
	fusedInput := make([]float32, 0, 2*h)
	normedEmbedding := rmsnormCfg(currentEmbedding, embeddingNorm, eps, f.target.Cfg)
	normedHidden := rmsnormCfg(priorHidden, hiddenNorm, eps, f.target.Cfg)
	fusedInput = append(fusedInput, normedEmbedding...)
	fusedInput = append(fusedInput, normedHidden...)
	return f.mat.mul("mtp.fc.weight", fusedInput, h, 2*h), nil
}
