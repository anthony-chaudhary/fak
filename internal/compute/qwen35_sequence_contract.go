package compute

import "fmt"

// Qwen35SequencePrefillPath is the capability identity for a backend that owns an
// entire Qwen3.5/3.8 hybrid prompt panel, KV cache, and recurrent state for one call.
const Qwen35SequencePrefillPath = "qwen35-hybrid-sequence-prefill-v1"

// Qwen35SequenceEmbeddingRowsPath identifies the optional extension which accepts
// an already-gathered [tokens, hidden] embedding panel. Backends must advertise
// this exact identity before model may replace the full [vocab, hidden] table.
const Qwen35SequenceEmbeddingRowsPath = "qwen35-hybrid-sequence-embedding-rows-v1"

// Qwen35SequenceAllLogitsPath identifies the optional final-projection extension
// which returns one device-resident logits row per input token.
const Qwen35SequenceAllLogitsPath = "qwen35-hybrid-sequence-all-logits-v1"

// Qwen35SequencePrefixReplayPath identifies the optional transaction extension
// which retains the GDN projections needed to commit a causal prefix without
// rerunning the complete model for its accepted tokens.
const Qwen35SequencePrefixReplayPath = "qwen35-hybrid-sequence-prefix-replay-v1"

// Qwen35SequencePrefixReplay owns request-detached projection tensors until its
// transaction either commits a prefix or closes. Before is borrowed from the
// caller's intact pre-round snapshot and remains caller-owned.
type Qwen35SequencePrefixReplay interface {
	CommitPrefix(accepted int, before []Qwen35SequenceState) error
	Close()
}

// Qwen35SequencePrefixReplayBackend prevents callers from requesting retained
// projections from sequence implementations without the matching lifecycle.
type Qwen35SequencePrefixReplayBackend interface {
	Qwen35SequencePrefixReplayPath() string
}

// Qwen35SequenceAllLogitsBackend prevents callers from setting NeedAllLogits on
// sequence implementations that predate the all-row result contract.
type Qwen35SequenceAllLogitsBackend interface {
	Qwen35SequenceAllLogitsPath() string
}

// Qwen35SequenceEmbeddingRowsBackend is the structural marker for bounded prompt
// embedding panels. The operation itself remains Qwen35SequencePrefill; the marker
// prevents an older backend from interpreting a row panel as a vocabulary table.
type Qwen35SequenceEmbeddingRowsBackend interface {
	Qwen35SequenceEmbeddingRowsPath() string
}

// qwen35SequenceEmbeddingContract resolves the mutually exclusive embedding
// representations used by sequence prefill. Table mode derives vocabulary from
// [vocab, hidden] and forbids an override; row mode requires [tokens, hidden]
// plus the original vocabulary used to validate token IDs and size logits.
func qwen35SequenceEmbeddingContract(req Qwen35SequencePrefillRequest) (vocab int, shape []int, err error) {
	if req.TokenEmbeddingRows {
		if req.TokenEmbeddingVocab <= 0 {
			return 0, nil, &Qwen35SequenceError{Stage: "tensor-preflight", Layer: -1, Reason: "token embedding row panel requires a positive vocabulary size"}
		}
		return req.TokenEmbeddingVocab, []int{len(req.TokenIDs), req.Hidden}, nil
	}
	if req.TokenEmbeddingVocab != 0 {
		return 0, nil, &Qwen35SequenceError{Stage: "tensor-preflight", Layer: -1, Reason: "token embedding vocabulary override requires row-panel mode"}
	}
	if len(req.TokenEmbedding.Shape) != 2 || req.TokenEmbedding.Shape[0] <= 0 {
		return 0, nil, &Qwen35SequenceError{Stage: "tensor-preflight", Layer: -1, Reason: fmt.Sprintf("token embedding shape %v, want [vocab,%d]", req.TokenEmbedding.Shape, req.Hidden)}
	}
	vocab = req.TokenEmbedding.Shape[0]
	return vocab, []int{vocab, req.Hidden}, nil
}

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

	TokenEmbedding      Tensor
	TokenEmbeddingRows  bool
	TokenEmbeddingVocab int
	OutputNorm          Tensor
	Output              Tensor
	Layers              []Qwen35SequenceLayer
	States              []Qwen35SequenceState
	KV                  KVStore

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
	// NeedAllLogits requests one output-logit row for every input token. The
	// rows stay device-resident; callers explicitly decide whether to read them.
	NeedAllLogits bool
	// CapturePrefixReplay retains only the GDN projection panels required to
	// repair a partially accepted causal prefix. It is valid for 1..4 tokens
	// together with NeedAllLogits.
	CapturePrefixReplay bool
}

// Qwen35SequencePrefillResult returns only resident products. KV and recurrent
// state are mutated in place and therefore are not replaceable result values.
type Qwen35SequencePrefillResult struct {
	LastHidden   Tensor
	Logits       Tensor
	LogitsRows   Tensor
	PrefixReplay Qwen35SequencePrefixReplay
	Tokens       int
	Transfers    Qwen35SequenceTransferCounters
}
