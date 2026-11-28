package model

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// -----------------------------------------------------------------------------
// Qwen3.8 QSA pluggable sparse-attention backend contract
// -----------------------------------------------------------------------------
//
// QSA (Qwen3.8 sparse attention) runs in two phases: an indexer kernel
// selects the top-k KV tiles for each query row, and a sparse core attends
// only over the selected tiles. TensorRT-LLM's PrimTS QSA backend wires both
// phases through a params/plan object pair and a pluggable backend
// registration so the engine can swap implementations without touching the
// model path.
//
// This component lands that contract on the fak model side as a pure-Go
// pluggable backend: the params/plan object shape, the
// indexer-selection-then-core-attention phase order, and a closed in-memory
// oracle over a fixed tile fixture. Structure and oracle only: no hardware
// run, no served-token or speed claim.
//
// Provenance (ADAPT): the contract is adapted from NVIDIA/TensorRT-LLM,
// PR #20714, at pin
// tensorrt_llm/_torch/attention/backends/sparse/qsa/prims_ts_core.py:1;
// tensorrt_llm/_torch/attention/backends/sparse/qsa/prims_ts_indexer/api.py:1
// upstream code is copied; the phase and registration facts are re-expressed
// as a Go contract with a closed host oracle.

// QSAPluggableBackendPath is the backend-neutral capability identity for the
// Qwen3.8 QSA pluggable sparse-attention backend.
const QSAPluggableBackendPath = "qwen38/qsa-pluggable-backend-v1"

// QSAPhase names one phase of the QSA two-phase program.
type QSAPhase int

const (
	// QSAPhaseIndexerSelection is the first phase: the indexer kernel
	// selects the top-k KV tiles for each query row.
	QSAPhaseIndexerSelection QSAPhase = iota
	// QSAPhaseCoreAttention is the second phase: the sparse core attends
	// only over the tiles the indexer selected.
	QSAPhaseCoreAttention
)

// String names the phase for reports and refusals.
func (p QSAPhase) String() string {
	switch p {
	case QSAPhaseIndexerSelection:
		return "indexer-selection"
	case QSAPhaseCoreAttention:
		return "core-attention"
	default:
		return fmt.Sprintf("qsa-phase-%d", int(p))
	}
}

// qsaPhaseOrder is the only legal phase order for a QSA plan.
var qsaPhaseOrder = []QSAPhase{QSAPhaseIndexerSelection, QSAPhaseCoreAttention}

// qsaPhaseOrderOK reports whether phases names exactly the
// indexer-selection-then-core-attention order.
func qsaPhaseOrderOK(phases []QSAPhase) bool {
	if len(phases) != len(qsaPhaseOrder) {
		return false
	}
	for i, want := range qsaPhaseOrder {
		if phases[i] != want {
			return false
		}
	}
	return true
}

// ErrQSAParams is the sentinel returned when a QSA params or fixture shape is
// unusable. Callers match it with errors.Is; the wrapped error carries the
// offending values.
var ErrQSAParams = errors.New("model: qsa params or fixture shape is invalid")

// ErrQSAUnregisteredBackend is the sentinel returned when a QSA backend path
// has no registered implementation. Callers match it with errors.Is; the
// wrapped error names the path.
var ErrQSAUnregisteredBackend = errors.New("model: qsa backend is not registered")

// ErrQSAPhaseOrder is the sentinel returned when a QSA plan does not name the
// indexer-selection-then-core-attention order.
var ErrQSAPhaseOrder = errors.New("model: qsa plan phase order is not indexer-selection then core-attention")

// QSAUnregisteredBackendError names the fail-closed refusal for a lookup of a
// path with no registered QSA backend.
type QSAUnregisteredBackendError struct {
	Path string
}

func (e *QSAUnregisteredBackendError) Error() string {
	return fmt.Sprintf("model: no QSA backend registered for path %q; refusing to run sparse attention", e.Path)
}

// Unwrap makes errors.Is(err, ErrQSAUnregisteredBackend) reachable across the
// typed refusal.
func (e *QSAUnregisteredBackendError) Unwrap() error { return ErrQSAUnregisteredBackend }

// QSAParams is the per-call parameter object handed to a QSA backend: the
// indexer top-k budget and the core-attention geometry.
type QSAParams struct {
	TopK         int     // top-k KV tiles selected per query row
	HeadDim      int     // per-tile vector dimension
	SoftmaxScale float32 // core-attention softmax scale
}

// QSAFixture is the closed in-memory tile fixture a QSA backend runs over:
// fixed query, key, and value tiles. No hardware, no allocation beyond the
// caller's slices.
type QSAFixture struct {
	QueryTiles [][]float32 // [queryRow][HeadDim]
	KeyTiles   [][]float32 // [kvTile][HeadDim]
	ValueTiles [][]float32 // [kvTile][HeadDim]
}

// QSAPlan is the execution plan object a backend derives from the params and
// the fixture: the tile budget and the ordered phase program.
type QSAPlan struct {
	Params    QSAParams
	QueryRows int
	TileCount int
	Phases    []QSAPhase
}

// valid reports whether the plan covers a positive tile budget and names the
// legal two-phase order.
func (p QSAPlan) valid() bool {
	return p.QueryRows > 0 && p.TileCount > 0 && qsaPhaseOrderOK(p.Phases)
}

// QSASelection is the indexer phase's output: the top-k KV tile indices for
// each query row, sorted by descending query-key score.
type QSASelection [][]int

// QSAOutput is the core-attention phase's output: one HeadDim vector per
// query row.
type QSAOutput [][]float32

// QSABackend is the pluggable QSA sparse-attention backend contract. A
// backend derives the plan, runs the indexer-selection phase, then the
// core-attention phase; the engine swaps implementations through the
// registry without touching the model path.
type QSABackend interface {
	// QSAPluggableBackendPath reports the capability path the backend is
	// registered under.
	QSAPluggableBackendPath() string
	// Plan derives the execution plan from the params and the fixture.
	Plan(params QSAParams, fixture QSAFixture) (QSAPlan, error)
	// SelectIndexer runs the indexer-selection phase: it returns the top-k
	// KV tile indices for each query row.
	SelectIndexer(plan QSAPlan, fixture QSAFixture) (QSASelection, error)
	// RunCore runs the core-attention phase over the selected tiles only.
	RunCore(plan QSAPlan, selection QSASelection, fixture QSAFixture) (QSAOutput, error)
}

// qsaValidateParams checks the params and fixture shapes a backend may rely
// on. It fails closed so a zero-sized budget cannot become the thing every
// phase derives from.
func qsaValidateParams(params QSAParams, fixture QSAFixture) error {
	if params.TopK <= 0 || params.HeadDim <= 0 || params.SoftmaxScale <= 0 {
		return fmt.Errorf("%w: top_k=%d head_dim=%d softmax_scale=%v",
			ErrQSAParams, params.TopK, params.HeadDim, params.SoftmaxScale)
	}
	if len(fixture.QueryTiles) == 0 || len(fixture.KeyTiles) == 0 || len(fixture.ValueTiles) == 0 {
		return fmt.Errorf("%w: empty fixture (query=%d key=%d value=%d tiles)",
			ErrQSAParams, len(fixture.QueryTiles), len(fixture.KeyTiles), len(fixture.ValueTiles))
	}
	if len(fixture.KeyTiles) != len(fixture.ValueTiles) {
		return fmt.Errorf("%w: key/value tile count mismatch (key=%d value=%d)",
			ErrQSAParams, len(fixture.KeyTiles), len(fixture.ValueTiles))
	}
	for i, row := range fixture.QueryTiles {
		if len(row) != params.HeadDim {
			return fmt.Errorf("%w: query tile %d has dim %d, want %d",
				ErrQSAParams, i, len(row), params.HeadDim)
		}
	}
	for i, row := range fixture.KeyTiles {
		if len(row) != params.HeadDim {
			return fmt.Errorf("%w: key tile %d has dim %d, want %d",
				ErrQSAParams, i, len(row), params.HeadDim)
		}
	}
	for i, row := range fixture.ValueTiles {
		if len(row) != params.HeadDim {
			return fmt.Errorf("%w: value tile %d has dim %d, want %d",
				ErrQSAParams, i, len(row), params.HeadDim)
		}
	}
	if params.TopK > len(fixture.KeyTiles) {
		return fmt.Errorf("%w: top_k=%d exceeds tile count %d",
			ErrQSAParams, params.TopK, len(fixture.KeyTiles))
	}
	return nil
}

// qsaDot returns the dot product of two equal-length vectors.
func qsaDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// qsaTopK returns the indices of the k largest scores, sorted by descending
// score with ties kept toward the lower tile index.
func qsaTopK(scores []float32, k int) []int {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	// Insertion sort: the fixture is small, and the strict comparison keeps
	// the order stable so ties keep the lower index first.
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0; j-- {
			a, b := idx[j], idx[j-1]
			if scores[a] > scores[b] {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			} else {
				break
			}
		}
	}
	return idx[:k]
}

// QSAOracle is the closed in-memory reference implementation of the QSA
// pluggable backend contract: deterministic top-k tile selection by
// query-key dot product, then sparse core attention over the selected tiles
// only. It is the host oracle a test pins the contract against; it makes no
// hardware claim.
type QSAOracle struct{}

// NewQSAOracle returns the closed in-memory QSA oracle.
func NewQSAOracle() *QSAOracle { return &QSAOracle{} }

// QSAPluggableBackendPath reports the oracle's capability path.
func (o *QSAOracle) QSAPluggableBackendPath() string { return QSAPluggableBackendPath }

// Plan derives the execution plan: the tile budget and the
// indexer-selection-then-core-attention phase order.
func (o *QSAOracle) Plan(params QSAParams, fixture QSAFixture) (QSAPlan, error) {
	if err := qsaValidateParams(params, fixture); err != nil {
		return QSAPlan{}, err
	}
	return QSAPlan{
		Params:    params,
		QueryRows: len(fixture.QueryTiles),
		TileCount: len(fixture.KeyTiles),
		Phases:    append([]QSAPhase(nil), qsaPhaseOrder...),
	}, nil
}

// SelectIndexer runs the indexer-selection phase: for each query row it
// scores every KV tile by query-key dot product and keeps the top-k.
func (o *QSAOracle) SelectIndexer(plan QSAPlan, fixture QSAFixture) (QSASelection, error) {
	if err := qsaValidateParams(plan.Params, fixture); err != nil {
		return nil, err
	}
	if !plan.valid() {
		return nil, fmt.Errorf("%w: plan has %d query rows, %d tiles, phases %v",
			ErrQSAParams, plan.QueryRows, plan.TileCount, plan.Phases)
	}
	sel := make(QSASelection, plan.QueryRows)
	for row, q := range fixture.QueryTiles {
		scores := make([]float32, plan.TileCount)
		for t, k := range fixture.KeyTiles {
			scores[t] = qsaDot(q, k)
		}
		sel[row] = qsaTopK(scores, plan.Params.TopK)
	}
	return sel, nil
}

// RunCore runs the core-attention phase: for each query row it re-scores the
// selected tiles, applies the scaled softmax, and returns the weighted sum of
// the selected value tiles. Unselected tiles contribute nothing.
func (o *QSAOracle) RunCore(plan QSAPlan, selection QSASelection, fixture QSAFixture) (QSAOutput, error) {
	if err := qsaValidateParams(plan.Params, fixture); err != nil {
		return nil, err
	}
	if !plan.valid() {
		return nil, fmt.Errorf("%w: plan has %d query rows, %d tiles, phases %v",
			ErrQSAParams, plan.QueryRows, plan.TileCount, plan.Phases)
	}
	if len(selection) != plan.QueryRows {
		return nil, fmt.Errorf("%w: selection has %d rows, plan has %d",
			ErrQSAParams, len(selection), plan.QueryRows)
	}
	out := make(QSAOutput, plan.QueryRows)
	for row, q := range fixture.QueryTiles {
		tiles := selection[row]
		if len(tiles) != plan.Params.TopK {
			return nil, fmt.Errorf("%w: selection row %d has %d tiles, want %d",
				ErrQSAParams, row, len(tiles), plan.Params.TopK)
		}
		scores := make([]float32, len(tiles))
		for i, t := range tiles {
			scores[i] = qsaDot(q, fixture.KeyTiles[t]) * plan.Params.SoftmaxScale
		}
		max := scores[0]
		for _, s := range scores[1:] {
			if s > max {
				max = s
			}
		}
		weights := make([]float32, len(tiles))
		var sum float32
		for i, s := range scores {
			weights[i] = float32(math.Exp(float64(s - max)))
			sum += weights[i]
		}
		out[row] = make([]float32, plan.Params.HeadDim)
		for i, t := range tiles {
			w := weights[i] / sum
			v := fixture.ValueTiles[t]
			for d := 0; d < plan.Params.HeadDim; d++ {
				out[row][d] += w * v[d]
			}
		}
	}
	return out, nil
}

// RunQSA executes the two-phase QSA program against a backend: it derives
// the plan, runs the indexer-selection phase, then the core-attention
// phase. It fails closed when the backend's plan does not name exactly that
// order, so a reversed or partial program never runs.
func RunQSA(backend QSABackend, params QSAParams, fixture QSAFixture) (QSAOutput, error) {
	if backend == nil {
		return nil, fmt.Errorf("%w: nil backend", ErrQSAParams)
	}
	plan, err := backend.Plan(params, fixture)
	if err != nil {
		return nil, err
	}
	if !qsaPhaseOrderOK(plan.Phases) {
		return nil, fmt.Errorf("%w: plan names phases %v, want %v",
			ErrQSAPhaseOrder, plan.Phases, qsaPhaseOrder)
	}
	if plan.QueryRows <= 0 || plan.TileCount <= 0 {
		return nil, fmt.Errorf("%w: plan has %d query rows, %d tiles",
			ErrQSAParams, plan.QueryRows, plan.TileCount)
	}
	selection, err := backend.SelectIndexer(plan, fixture)
	if err != nil {
		return nil, err
	}
	return backend.RunCore(plan, selection, fixture)
}

// QSARegistry is the pluggable registration table for QSA backends. The
// engine registers a backend under its capability path and looks it up at
// dispatch; a swap replaces the implementation for a path without touching
// the model path.
type QSARegistry struct {
	mu     sync.RWMutex
	byPath map[string]QSABackend
}

// NewQSARegistry returns an empty QSA backend registry.
func NewQSARegistry() *QSARegistry {
	return &QSARegistry{byPath: make(map[string]QSABackend)}
}

// Register adds a backend under its capability path. It fails closed on a
// nil backend or an empty path.
func (r *QSARegistry) Register(backend QSABackend) error {
	if backend == nil {
		return fmt.Errorf("%w: nil backend", ErrQSAParams)
	}
	path := backend.QSAPluggableBackendPath()
	if path == "" {
		return fmt.Errorf("%w: backend advertises an empty path", ErrQSAParams)
	}
	r.mu.Lock()
	r.byPath[path] = backend
	r.mu.Unlock()
	return nil
}

// Swap replaces the backend registered under path with backend. It fails
// closed when path has no current registration, so a swap cannot silently
// create a path.
func (r *QSARegistry) Swap(path string, backend QSABackend) error {
	if backend == nil {
		return fmt.Errorf("%w: nil backend", ErrQSAParams)
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrQSAParams)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.byPath[path]; !ok {
		return &QSAUnregisteredBackendError{Path: path}
	}
	r.byPath[path] = backend
	return nil
}

// Lookup returns the backend registered under path. It fails closed with
// ErrQSAUnregisteredBackend when no implementation is registered, so an
// unregistered path never runs sparse attention.
func (r *QSARegistry) Lookup(path string) (QSABackend, error) {
	r.mu.RLock()
	backend, ok := r.byPath[path]
	r.mu.RUnlock()
	if !ok {
		return nil, &QSAUnregisteredBackendError{Path: path}
	}
	return backend, nil
}
