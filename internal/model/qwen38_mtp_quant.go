package model

import (
	"fmt"
	"strings"
)

// Qwen38MTPTensorFormat names the actual retained storage used by every MTP
// projection. Norm vectors remain F32 in both admitted layouts.
type Qwen38MTPTensorFormat string

const (
	Qwen38MTPFormatF32  Qwen38MTPTensorFormat = "F32"
	Qwen38MTPFormatQ4K  Qwen38MTPTensorFormat = "Q4_K"
	Qwen38MTPFormatNone Qwen38MTPTensorFormat = ""
)

// Qwen38MTPTensorLayout is a read-only inventory derived from the model's real
// resident stores. TensorTypes is diagnostic evidence; admission keys on Format.
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

func isQwen38MTPMatrixTensor(name string) bool {
	for _, candidate := range qwen38MTPMatrixTensors {
		if name == candidate {
			return true
		}
	}
	return false
}

// Qwen38MTPTensorLayout reports the actual retained MTP precision. It admits
// exactly two closed layouts:
//
//   - every required tensor is F32;
//   - every matrix is resident raw Q4_K while every norm remains F32.
//
// A Q4_K_M artifact label is deliberately irrelevant. Mixed stores, Q8/K-quant
// substitutions, duplicate representations, or malformed resident spans are
// precision_unsupported at the eligibility boundary.
func (m *Model) Qwen38MTPTensorLayout() (Qwen38MTPTensorLayout, error) {
	layout, present, err := m.qwen38MTPTensorLayout()
	if !present && err == nil {
		err = qwen35MTPStateError("weight lookup", "complete retained Qwen3.8 MTP tensor set", "missing")
	}
	return layout, err
}

func (m *Model) qwen38MTPTensorLayout() (Qwen38MTPTensorLayout, bool, error) {
	layout := Qwen38MTPTensorLayout{TensorTypes: make(map[string]string, len(qwen35MTPRequiredTensors))}
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

	matrixFormat := Qwen38MTPFormatNone
	for _, name := range qwen35MTPRequiredTensors {
		wantShape := expected[name]
		if isQwen38MTPMatrixTensor(name) {
			format, err := m.qwen38MTPMatrixFormat(name, wantShape)
			if err != nil {
				return layout, true, err
			}
			layout.TensorTypes[name] = string(format)
			if matrixFormat == Qwen38MTPFormatNone {
				matrixFormat = format
			} else if matrixFormat != format {
				return layout, true, &Qwen35MTPForwardError{
					Stage:  "weight precision",
					Tensor: name,
					Want:   string(matrixFormat),
					Got:    string(format) + " (mixed MTP projection layout)",
				}
			}
			continue
		}

		meta, ok := m.manifest[name]
		if !ok {
			return layout, true, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: name, Want: "F32 norm", Got: "missing"}
		}
		if !strings.EqualFold(meta.Dtype, "F32") {
			return layout, true, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32 norm", Got: meta.Dtype}
		}
		if err := validateQwen38MTPF32Meta(m, name, meta, wantShape); err != nil {
			return layout, true, err
		}
		if m.q4kw[name] != nil || m.q8w[name] != nil || m.kqw[name] != nil {
			return layout, true, &Qwen35MTPForwardError{Stage: "weight precision", Tensor: name, Want: "one F32 norm representation", Got: "duplicate or quantized norm representation"}
		}
		layout.TensorTypes[name] = "F32"
	}
	layout.Format = matrixFormat
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
		return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: name, Want: "F32 or resident Q4_K", Got: "missing"}
	}
	if count != 1 {
		return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight precision", Tensor: name, Want: "one retained representation", Got: "mixed or duplicate representations"}
	}

	switch {
	case hasF32:
		if !strings.EqualFold(meta.Dtype, "F32") {
			return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32 or Q4_K", Got: meta.Dtype}
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
		return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32 or Q4_K", Got: "Q8_0"}
	default:
		return Qwen38MTPFormatNone, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32 or Q4_K", Got: kq.kind.String()}
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
	qt := f.draft.M.q4kw["mtp.fc.weight"]
	if qt == nil {
		return nil, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: "mtp.fc.weight", Want: "resident Q4_K", Got: "missing"}
	}
	return f.draft.q4kMatRowsDispatch("mtp.fc.weight", qt, fusedInput), nil
}
