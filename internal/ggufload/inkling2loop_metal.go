package ggufload

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// OpenInkling2LoopMetal binds the original GGUF matrix blocks to the Metal
// batcher. Embedding lookup, routing, attention and cache state belong to the
// native host forward.
func OpenInkling2LoopMetal(path string) (*Inkling2LoopModel, error) {
	data, mapping, ok, err := model.MmapOpen(path)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("inkling2_loop: cannot map %q", path)
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		mapping.Close()
		return nil, err
	}
	batch := &inkling2MetalBatch{store: store, mapping: mapping}
	m, err := bindInkling2LoopWith(data, batch.bind)
	if err != nil {
		batch.Close()
		return nil, err
	}
	m.Weights.Batcher = batch
	m.closer = batch
	return m, nil
}

type inkling2MetalBatch struct {
	store   *metalgemm.PackedGGUFStore
	mapping io.Closer
}

func (b *inkling2MetalBatch) Close() error {
	if b.mapping == nil {
		return nil
	}
	b.store.Close()
	err := b.mapping.Close()
	b.mapping = nil
	return err
}

func (b *inkling2MetalBatch) bind(t TensorInfo) (inkling2PackedWeight, error) {
	cols, rows := int(t.Dims[0]), int(t.Dims[1])
	experts := 1
	if len(t.Dims) == 3 {
		experts = int(t.Dims[2])
	}
	w, err := b.store.Matrix(int(t.FileOffset), uint32(t.Type), rows, cols, experts)
	if err != nil {
		return nil, err
	}
	return &inkling2MetalMatrix{store: b.store, weight: w, rows: rows, cols: cols, experts: experts}, nil
}

type inkling2MetalMatrix struct {
	store               *metalgemm.PackedGGUFStore
	weight              *metalgemm.PackedGGUFMatrix
	rows, cols, experts int
}

func (m *inkling2MetalMatrix) Rows() int    { return m.rows }
func (m *inkling2MetalMatrix) Cols() int    { return m.cols }
func (m *inkling2MetalMatrix) Experts() int { return m.experts }
func (m *inkling2MetalMatrix) MatVec(dst, x []float32) {
	m.ExpertMatVec(0, dst, x)
}
func (m *inkling2MetalMatrix) ExpertMatVec(expert int, dst, x []float32) {
	job := metalgemm.PackedGGUFJob{Matrix: m.weight, Expert: expert, X: x, Dst: dst}
	runInkling2MetalJobs(m.store, []metalgemm.PackedGGUFJob{job})
}

func (b *inkling2MetalBatch) MatVecBatch(jobs []model.Inkling2LoopMatVecJob) {
	packed := make([]metalgemm.PackedGGUFJob, len(jobs))
	for i, job := range jobs {
		var matrix *inkling2MetalMatrix
		if job.Matrix != nil {
			matrix = job.Matrix.(*inkling2MetalMatrix)
		} else {
			matrix = job.Experts.(*inkling2MetalMatrix)
		}
		packed[i] = metalgemm.PackedGGUFJob{Matrix: matrix.weight, Expert: job.Expert, X: job.X, Dst: job.Dst}
	}
	runInkling2MetalJobs(b.store, packed)
}

func runInkling2MetalJobs(store *metalgemm.PackedGGUFStore, jobs []metalgemm.PackedGGUFJob) {
	if err := store.Run(jobs); err != nil {
		panic(inkling2DispatchError{cause: fmt.Errorf("inkling2_loop Metal: %w", err)})
	}
}

func (b *inkling2MetalBatch) ExpertMLP(lw *model.Inkling2LoopLayerWeights, hp *model.Inkling2LoopHParams, selected []int, weights, x []float32) ([]float32, bool) {
	gu, okGate := lw.GateUp.(*inkling2MetalMatrix)
	down, okDown := lw.Down.(*inkling2MetalMatrix)
	if !okGate || !okDown {
		return nil, false
	}
	out := make([]float32, hp.Hidden)
	job := metalgemm.PackedExpertMLP{GateUp: gu.weight, Down: down.weight, Selected: selected, Weights: weights,
		GateUpScale: lw.GateUpScale, DownScale: lw.DownScale, Beta: hp.SituBeta, LinearBeta: hp.SituLinearBeta}
	if err := b.store.ExpertMLP(job, x, out); err != nil {
		panic(inkling2DispatchError{cause: fmt.Errorf("inkling2_loop Metal experts: %w", err)})
	}
	return out, true
}

func (b *inkling2MetalBatch) RoutedFFN(lw *model.Inkling2LoopLayerWeights, hp *model.Inkling2LoopHParams, x []float32) ([]float32, bool) {
	gate, okGate := lw.GateUp.(*inkling2MetalMatrix)
	down, okDown := lw.Down.(*inkling2MetalMatrix)
	router, okRouter := lw.Router.(*inkling2MetalMatrix)
	if !okRouter || !okGate || !okDown {
		return nil, false
	}
	out := make([]float32, hp.Hidden)
	job := metalgemm.PackedRoutedFFN{Router: router.weight, GateUp: gate.weight, Down: down.weight, Slots: hp.ExpertsUsed,
		Bias: lw.ExpProbsBias, GateUpScale: lw.GateUpScale, DownScale: lw.DownScale, Beta: hp.SituBeta,
		LinearBeta: hp.SituLinearBeta, WeightScale: hp.ExpertWeightsScale}
	ok, err := b.store.RoutedFFN(job, x, out)
	if err != nil {
		panic(inkling2DispatchError{cause: fmt.Errorf("inkling2_loop Metal routed FFN: %w", err)})
	}
	return out, ok
}
