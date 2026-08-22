package compute

import "fmt"

// Qwen35SequencePrefillPath is the capability identity for a backend that owns an
// entire Qwen3.5/3.8 hybrid prompt panel, KV cache, and recurrent state for one call.
const Qwen35SequencePrefillPath = "qwen35-hybrid-sequence-prefill-v1"

// Qwen35SequenceParityCosineMin is the bounded device/reference acceptance
// floor for the complete deterministic sequence witness.
const Qwen35SequenceParityCosineMin = 0.999

// The production Qwen3.8 dense checkpoint carries one trailing MTP metadata
// block after the 64 blocks executed by the text forward. The loader excludes
// that sidecar before constructing Layers; these constants keep that distinction
// explicit at the compute seam.
const (
	Qwen35DenseMainLayers     = 64
	Qwen35DenseMetadataLayers = 1
	Qwen35DenseHidden         = 5120
	Qwen35DenseIntermediate   = 17408
	Qwen35DenseQueryHeads     = 24
	Qwen35DenseKVHeads        = 4
	Qwen35DenseHeadDim        = 256
	Qwen35DenseGDNInner       = 6144
	Qwen35DenseGDNState       = 128
	Qwen35DenseGDNConv        = 4
	Qwen35DenseGDNGroups      = 16
	Qwen35DenseGDNRank        = 48
)

// Qwen35SequenceError is the fail-closed verdict for the whole resident path.
// Layer is -1 for request-wide stages. Cause preserves a CUDA/GDN typed error
// when execution, rather than request validation, failed.
type Qwen35SequenceError struct {
	Stage  string
	Layer  int
	Reason string
	Cause  error
}

func (e *Qwen35SequenceError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 resident sequence error"
	}
	where := e.Stage
	if e.Layer >= 0 {
		where = fmt.Sprintf("layer %d %s", e.Layer, e.Stage)
	}
	reason := e.Reason
	if e.Cause != nil {
		reason = e.Cause.Error()
	}
	return fmt.Sprintf("compute: Qwen3.5 resident sequence failed closed at %s: %s", where, reason)
}

func (e *Qwen35SequenceError) Unwrap() error { return e.Cause }

// Qwen35SequenceTransferCounters are deltas measured inside one sequence call.
// H2DBytes includes the small token-id control upload used by embedding gather.
// Activation* counts only traffic after that gather; both fields must remain zero
// for a resident layer stack and output head.
type Qwen35SequenceTransferCounters struct {
	H2DBytes           uint64
	D2HBytes           uint64
	ActivationH2DBytes uint64
	ActivationD2HBytes uint64
}

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
	Transfers  Qwen35SequenceTransferCounters
}
