package compute

// Qwen35SequencePrefillPath is the capability identity for a backend that owns an
// entire Qwen3.5/3.8 hybrid prompt panel, KV cache, and recurrent state for one call.
const Qwen35SequencePrefillPath = "qwen35-hybrid-sequence-prefill-v1"

// Qwen35SequenceState is persistent per-layer GDN state. Backends update both
// tensors in place; replacing either handle is a contract violation.
type Qwen35SequenceState struct {
	Conv      Tensor
	Recurrent Tensor
}

// Qwen35SequenceLayer contains the resident weights for one hybrid block. Linear
// layers consume GDN; attention layers consume Q/K/V/O and append to KV.
type Qwen35SequenceLayer struct {
	Linear bool

	InputNorm Tensor
	PostNorm  Tensor

	Q, K, V, O   Tensor
	QNorm, KNorm Tensor

	GDNInQKV, GDNInZ           Tensor
	GDNInB, GDNInA             Tensor
	GDNConv, GDNALog           Tensor
	GDNDTBias, GDNNorm, GDNOut Tensor

	Gate, Up, Down Tensor
}

// Qwen35SequencePrefillRequest is a complete, compute-owned description of one
// prompt prefill. Keeping this type outside model avoids a package cycle while
// allowing optional backends to implement the operation structurally.
type Qwen35SequencePrefillRequest struct {
	Path     string
	TokenIDs []int
	StartPos int

	TokenEmbedding Tensor
	OutputNorm     Tensor
	Output         Tensor
	Layers         []Qwen35SequenceLayer
	States         []Qwen35SequenceState
	KV             KVStore

	Hidden, Intermediate int
	NumHeads, NumKVHeads int
	HeadDim, RotaryDim   int
	NumKeyHeads          int
	NumValueHeads        int
	KeyHeadDim           int
	ValueHeadDim         int
	ConvKernel           int
	RMSNormEpsilon       float32
	RoPEThetaForLayer    []float64
	NeedLogits           bool
}

// Qwen35SequencePrefillResult returns only resident products. KV and recurrent
// state are mutated in place and therefore are not replaceable result values.
type Qwen35SequencePrefillResult struct {
	LastHidden Tensor
	Logits     Tensor
	Tokens     int
}
