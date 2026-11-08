package gateway

// embedding_worker.go — model-worker-backed /v1/embeddings, ADAPTED from
// TensorFold's embeddings reply contract (Apache-2.0): ashhart/TensorFold
// zig/src/server/embeddings.zig:32 + embedding_request.zig
// @9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/server/embeddings.zig
//
// BORROWED CONTRACT, re-expressed in idiomatic Go (no upstream code copied):
//   - a configured ENCODER WORKER is invoked only after capability and input
//     validation: the worker declares its pooling mode, hidden dimension,
//     encoding formats and vocabulary; a request whose dimensions differ from
//     the declared hidden dimension, or whose encoding_format is outside the
//     declared set, refuses 400 with a typed sentinel BEFORE any encoder work
//     runs;
//   - text/token batches preserve sequence boundaries under per-input limits:
//     at most 8 inputs, each text 1..4096 bytes, each token row 1..256 ids,
//     total batch <= 1024 tokens; an oversized/mixed/heterogeneous input
//     refuses with a typed error and produces zero data rows (closed values
//     mirror the admitted causal contract so one wire contract covers both
//     backends);
//   - unexpected tokens or a non-length finish fail: a token id outside the
//     declared vocabulary, or a worker finish reason other than length/eos,
//     returns a typed refusal — the response is never 200 carrying data for
//     that input;
//   - borrowed storage survives cancellation to completion: the preallocated
//     output vector buffer is written through to the end even when ctx cancels
//     mid-batch; the cancelled request answers 408-class with zero data rows,
//     and the buffer returns to its pool exactly once;
//   - unwritten vectors or generated tokens cannot become successful
//     embeddings: any row not fully written fails the whole call; usage counts
//     only fully-written input tokens; float form is a JSON number array,
//     base64 form is little-endian float32 std-base64 (the encodeEmbedding
//     contract at embeddings.go).
//
// SHAPE — a package-level Set/Get accessor pair (SetEmbeddingEncoderWorker /
// GetEmbeddingEncoderWorker) consulted by handleEmbeddings ahead of the causal
// and hashing fallbacks; inert by default: no live worker leaves the request
// path byte-for-byte historical. The worker carries a bounded output-vector
// pool so a cancelled request completes its write without a second allocation.
//
// HONEST FENCE — what this is NOT. Today the Encode seam is pinned by fake
// encoders in tests: software witness only, no throughput or hardware claim.
// A real model runtime attaches to the same seam later.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Closed admission bounds, mirroring the admitted causal contract so one wire
// contract covers both backends. The per-row token bound is the shared
// model.CausalEmbeddingMaxTokens constant, not a duplicate.
const (
	embedWorkerMaxInputs      = 8
	embedWorkerMaxTextBytes   = 4096
	embedWorkerMaxBatchTokens = 1024
	embedWorkerMaxInputBytes  = 65536
)

// EmbeddingFinish is the worker's terminal completion class. Only Length and
// EOS are clean terminals: anything else (or a generated-token event) refuses
// the whole call.
type EmbeddingFinish int

const (
	EmbeddingFinishLength   EmbeddingFinish = iota // clean stop at the admitted length
	EmbeddingFinishEOS                             // terminal end-of-sequence
	EmbeddingFinishCanceled                        // engine canceled before completion
	EmbeddingFinishError                           // engine failure
)

// ErrEmbeddingWorker is the parent sentinel every encoder-worker refusal
// satisfies: a caller that only needs "the encoder worker said no" classifies
// against it; a caller that needs the class uses one of the specific sentinels
// below.
var ErrEmbeddingWorker = errors.New("gateway embedding encoder worker refused")

// The closed refusal classes. Each is distinct; all satisfy
// errors.Is(err, ErrEmbeddingWorker).
var (
	ErrEmbeddingWorkerModel      = errors.New("embedding model differs from the configured encoder worker")
	ErrEmbeddingWorkerDimensions = errors.New("embedding dimensions differ from the worker's declared hidden dimension")
	ErrEmbeddingWorkerFormat     = errors.New("embedding encoding format is not declared by the worker")
	ErrEmbeddingWorkerInput      = errors.New("embedding input violates the admitted batch shape")
	ErrEmbeddingWorkerVocab      = errors.New("embedding token ID outside the worker vocabulary")
	ErrEmbeddingWorkerCompletion = errors.New("embedding worker did not reach a clean terminal completion")
	ErrEmbeddingWorkerUnwritten  = errors.New("embedding worker left a vector row unwritten")
	ErrEmbeddingWorkerNonFinite  = errors.New("embedding worker vector is non-finite in the requested encoding")
)

// embeddingRefusal wraps one class sentinel plus the parent so errors.Is
// matches both — classification by class, never by error prose.
type embeddingRefusal struct{ cause error }

func (e *embeddingRefusal) Error() string { return e.cause.Error() }

func (e *embeddingRefusal) Unwrap() []error {
	return []error{e.cause, ErrEmbeddingWorker}
}

func refuseEmbedding(cause error) error { return &embeddingRefusal{cause: cause} }

// EmbeddingEncoderConfig declares the worker's closed capability set. Formats
// must be a non-empty subset of {"float","base64"}; Encode is the seam a real
// model runtime (or a test fake) fills.
type EmbeddingEncoderConfig struct {
	ModelName string
	Pooling   string
	HiddenDim int
	Formats   []string
	VocabSize int
	// Encode runs one admitted batch: it writes each input's pooled vector into
	// out[i*HiddenDim:(i+1)*HiddenDim] and reports its terminal finish class plus
	// whether it emitted any generated tokens. The contract is that it writes the
	// buffer through to the end even when ctx is done (borrowed storage survives
	// cancellation to completion); the gateway decides the reply afterwards.
	Encode func(ctx context.Context, rows [][]int, out []float64) (finish EmbeddingFinish, generated bool)
}

// EmbeddingEncoderWorker is the model-worker-backed /v1/embeddings backend.
// Build with NewEmbeddingEncoderWorker; install with SetEmbeddingEncoderWorker.
// Safe for concurrent use: the accessor pair is mutex-guarded and the pool is
// self-synchronizing.
type EmbeddingEncoderWorker struct {
	ModelName string
	Pooling   string
	HiddenDim int
	VocabSize int

	formats map[string]bool
	Encode  func(ctx context.Context, rows [][]int, out []float64) (finish EmbeddingFinish, generated bool)

	pool embeddingVectorPool
}

// NewEmbeddingEncoderWorker validates the declared capability set and refuses a
// worker that could not honor the wire contract: unknown pooling modes, zero
// dimensions/vocabulary, undeclared or empty format sets, and a missing Encode
// seam all fail construction rather than degrade at request time.
func NewEmbeddingEncoderWorker(cfg EmbeddingEncoderConfig) (*EmbeddingEncoderWorker, error) {
	if cfg.ModelName == "" {
		return nil, fmt.Errorf("gateway: encoder worker requires a served model name")
	}
	if cfg.Pooling != "mean" && cfg.Pooling != "last" {
		return nil, fmt.Errorf("gateway: encoder worker pooling must be mean or last")
	}
	if cfg.HiddenDim < 1 || cfg.VocabSize < 1 {
		return nil, fmt.Errorf("gateway: encoder worker requires positive hidden dimension and vocabulary")
	}
	formats := make(map[string]bool, len(cfg.Formats))
	for _, f := range cfg.Formats {
		if f != "float" && f != "base64" {
			return nil, fmt.Errorf("gateway: encoder worker encoding format must be float or base64")
		}
		formats[f] = true
	}
	if len(formats) == 0 {
		return nil, fmt.Errorf("gateway: encoder worker declares no encoding format")
	}
	if cfg.Encode == nil {
		return nil, fmt.Errorf("gateway: encoder worker requires an Encode seam")
	}
	return &EmbeddingEncoderWorker{
		ModelName: cfg.ModelName,
		Pooling:   cfg.Pooling,
		HiddenDim: cfg.HiddenDim,
		VocabSize: cfg.VocabSize,
		formats:   formats,
		Encode:    cfg.Encode,
		pool:      newEmbeddingVectorPool(embedWorkerMaxInputs),
	}, nil
}

// declares reports whether the worker declares the requested encoding form.
func (wk *EmbeddingEncoderWorker) declares(base64Encoded bool) bool {
	if base64Encoded {
		return wk.formats["base64"]
	}
	return wk.formats["float"]
}

// embeddingAdmission is the validated bounded batch an encoder worker consumes.
type embeddingAdmission struct {
	rows   [][]int
	b64    bool
	tokens int
}

// admit runs the capability guards and the sequence-boundary-preserving batch
// validation BEFORE any encoder work, returning typed sentinel refusals.
func (wk *EmbeddingEncoderWorker) admit(req EmbeddingsRequest) (*embeddingAdmission, error) {
	if req.Model != "" && req.Model != wk.ModelName {
		return nil, refuseEmbedding(ErrEmbeddingWorkerModel)
	}
	if req.Dimensions != 0 && req.Dimensions != wk.HiddenDim {
		return nil, refuseEmbedding(ErrEmbeddingWorkerDimensions)
	}
	b64, ok := embeddingEncoding(req.EncodingFormat)
	if !ok || !wk.declares(b64) {
		return nil, refuseEmbedding(ErrEmbeddingWorkerFormat)
	}
	texts, ids, err := workerSequences(req.Input)
	if err != nil {
		return nil, err
	}
	rows := make([][]int, 0, len(texts)+len(ids))
	for _, s := range texts {
		row := wk.encodeText(s)
		if len(row) == 0 {
			return nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
		rows = append(rows, row)
	}
	rows = append(rows, ids...)
	total := 0
	for _, row := range rows {
		if len(row) == 0 || len(row) > model.CausalEmbeddingMaxTokens {
			return nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
		for _, id := range row {
			if id < 0 || id >= wk.VocabSize {
				return nil, refuseEmbedding(ErrEmbeddingWorkerVocab)
			}
		}
		total += len(row)
	}
	if len(rows) == 0 || len(rows) > embedWorkerMaxInputs || total > embedWorkerMaxBatchTokens {
		return nil, refuseEmbedding(ErrEmbeddingWorkerInput)
	}
	return &embeddingAdmission{rows: rows, b64: b64, tokens: total}, nil
}

// workerSequences folds the OpenAI `input` field into text strings and explicit
// token-ID rows, preserving sequence boundaries and refusing mixed or
// heterogeneous shapes. The closed bounds mirror the admitted causal contract
// (embeddings_causal.go sequences()): a bare string, an array of strings, a
// single token-ID array, or a batch of token-ID arrays — nothing else.
func workerSequences(raw json.RawMessage) (texts []string, ids [][]int, err error) {
	if len(raw) == 0 || len(raw) > embedWorkerMaxInputBytes {
		return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
		return []string{s}, nil, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil || len(items) == 0 {
		return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
	}
	first := bytes.TrimSpace(items[0])
	if len(first) == 0 {
		return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
	}
	switch first[0] {
	case '"':
		if len(items) > embedWorkerMaxInputs {
			return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
		texts = make([]string, len(items))
		for i, item := range items {
			if err := json.Unmarshal(item, &texts[i]); err != nil {
				return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
			}
		}
	case '[':
		if len(items) > embedWorkerMaxInputs {
			return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
		ids = make([][]int, len(items))
		for i, item := range items {
			row, err := embeddingIDRow(item)
			if err != nil {
				return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
			}
			ids[i] = row
		}
	default:
		row, err := embeddingIDRow(raw)
		if err != nil {
			return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
		ids = [][]int{row}
	}
	for _, s := range texts {
		if len(s) == 0 || len(s) > embedWorkerMaxTextBytes {
			return nil, nil, refuseEmbedding(ErrEmbeddingWorkerInput)
		}
	}
	return texts, ids, nil
}

// encodeText maps a validated text input to deterministic token IDs inside the
// worker's declared vocabulary: the package word tokenizer feeds FNV-1a
// bucketing, so identical text always yields the same row and every id stays in
// [0, VocabSize).
func (wk *EmbeddingEncoderWorker) encodeText(s string) []int {
	toks := tokenizeForEmbedding(s)
	ids := make([]int, len(toks))
	for i, tok := range toks {
		ids[i] = int(fnv1a(tok) % uint64(wk.VocabSize))
	}
	return ids
}

// completionRefusal reports the typed refusal a terminal outcome produces: a
// generated-token event or any finish other than length/eos refuses the whole
// call; a clean terminal returns nil.
func completionRefusal(finish EmbeddingFinish, generated bool) error {
	if generated || (finish != EmbeddingFinishLength && finish != EmbeddingFinishEOS) {
		return refuseEmbedding(ErrEmbeddingWorkerCompletion)
	}
	return nil
}

// firstUnwrittenRow reports the index of the first vector row still holding a
// NaN component, or -1 when every row was fully written.
func firstUnwrittenRow(out []float64, inputs, dim int) int {
	for i := 0; i < inputs; i++ {
		for j := 0; j < dim; j++ {
			if math.IsNaN(out[i*dim+j]) {
				return i
			}
		}
	}
	return -1
}

// embeddingVectorPool hands out fixed-size float64 output buffers so the
// caller-owned vector stays alive through terminal completion without a second
// allocation. Every handout is NaN-filled so an unwritten row stays detectable;
// every borrow returns exactly once.
type embeddingVectorPool struct {
	mu      sync.Mutex
	free    [][]float64
	out     int
	maxFree int
}

func newEmbeddingVectorPool(maxFree int) embeddingVectorPool {
	return embeddingVectorPool{maxFree: maxFree}
}

// take hands out a NaN-filled buffer of exactly size floats, reusing a pooled
// buffer of the same size when one is free.
func (p *embeddingVectorPool) take(size int) []float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, v := range p.free {
		if len(v) == size {
			p.free = append(p.free[:i], p.free[i+1:]...)
			for j := range v {
				v[j] = math.NaN()
			}
			p.out++
			return v
		}
	}
	v := make([]float64, size)
	for j := range v {
		v[j] = math.NaN()
	}
	p.out++
	return v
}

// put returns a borrowed buffer to the pool. It is called exactly once per
// take, by the single deferred release on the serve path.
func (p *embeddingVectorPool) put(vec []float64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.out--
	if len(p.free) < p.maxFree {
		p.free = append(p.free, vec)
	}
}

// outstanding reports how many borrowed buffers have not yet returned.
func (p *embeddingVectorPool) outstanding() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.out
}

// borrowVectors hands out the preallocated output buffer for one admitted
// batch: one HiddenDim-wide row per input, NaN-filled so unwritten rows stay
// detectable.
func (wk *EmbeddingEncoderWorker) borrowVectors(inputs int) []float64 {
	return wk.pool.take(inputs * wk.HiddenDim)
}

// releaseVectors returns a borrowed buffer to the pool exactly once.
func (wk *EmbeddingEncoderWorker) releaseVectors(vec []float64) {
	wk.pool.put(vec)
}

// outstandingVectors reports how many borrowed buffers are still out.
func (wk *EmbeddingEncoderWorker) outstandingVectors() int {
	return wk.pool.outstanding()
}

var (
	embeddingEncoderMu sync.RWMutex
	embeddingEncoder   *EmbeddingEncoderWorker
)

// SetEmbeddingEncoderWorker installs (or, with nil, clears) the process-wide
// encoder worker consulted by handleEmbeddings ahead of the causal and hashing
// fallbacks. Inert by default: a nil worker leaves the request path
// byte-for-byte historical.
func SetEmbeddingEncoderWorker(wk *EmbeddingEncoderWorker) {
	embeddingEncoderMu.Lock()
	embeddingEncoder = wk
	embeddingEncoderMu.Unlock()
}

// GetEmbeddingEncoderWorker returns the live encoder worker, or nil when none
// is installed.
func GetEmbeddingEncoderWorker() *EmbeddingEncoderWorker {
	embeddingEncoderMu.RLock()
	defer embeddingEncoderMu.RUnlock()
	return embeddingEncoder
}

// handleEncoderEmbeddings serves POST /v1/embeddings over the configured
// encoder worker. The capability guards and batch validation run before any
// encoder work; the borrowed output buffer is released exactly once no matter
// which reply class the request ends in.
func (s *Server) handleEncoderEmbeddings(w http.ResponseWriter, r *http.Request, req EmbeddingsRequest, wk *EmbeddingEncoderWorker) {
	admitted, err := wk.admit(req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	out := wk.borrowVectors(len(admitted.rows))
	defer wk.releaseVectors(out)
	finish, generated := wk.Encode(r.Context(), admitted.rows, out)
	if r.Context().Err() != nil {
		// Borrowed storage survived cancellation to completion: the worker wrote
		// the buffer through to the end; this request answers 408-class with zero
		// data rows and the buffer returns to its pool exactly once.
		writeErr(w, http.StatusRequestTimeout, "embedding request canceled")
		return
	}
	if err := completionRefusal(finish, generated); err != nil {
		writeErr(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	if row := firstUnwrittenRow(out, len(admitted.rows), wk.HiddenDim); row >= 0 {
		writeErr(w, http.StatusUnprocessableEntity, refuseEmbedding(ErrEmbeddingWorkerUnwritten).Error())
		return
	}
	for _, value := range out {
		if math.IsInf(value, 0) || (admitted.b64 && math.IsInf(float64(float32(value)), 0)) {
			writeErr(w, http.StatusUnprocessableEntity, refuseEmbedding(ErrEmbeddingWorkerNonFinite).Error())
			return
		}
	}
	data := make([]EmbeddingData, len(admitted.rows))
	for i := range admitted.rows {
		vec := make([]float64, wk.HiddenDim)
		copy(vec, out[i*wk.HiddenDim:(i+1)*wk.HiddenDim])
		data[i] = EmbeddingData{Object: "embedding", Index: i, Embedding: encodeEmbedding(vec, admitted.b64)}
	}
	// The pooling identity accompanies every result, mirroring the causal
	// bindings.
	w.Header().Set("X-Fak-Embedding-Mode", "encoder-"+wk.Pooling)
	writeJSON(w, http.StatusOK, EmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  wk.ModelName,
		Usage:  EmbeddingUsage{PromptTokens: admitted.tokens, TotalTokens: admitted.tokens},
	})
}
