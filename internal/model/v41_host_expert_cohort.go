package model

import (
	"fmt"
	"os"
)

// v41HostExpertProjection is private to the callback-free full host expert
// path. Its three calls are gate, up, and down, with all original publication
// and arithmetic boundaries between them retained by the scalar consumer.
type v41HostExpertProjection interface {
	project(w, x []float32, out, in int) []float32
	gateUpShape(w1, w3 []float32, out, in int) bool
	combinedHostStage() bool
}

// v41ContractHostExpertGroup batches only the pure gate/up contractions. Raw
// future-row projections may be computed early, but their BF16 publication,
// validation, activation, weighting and down contraction remain row ordered.
func (m *Model) v41ContractHostExpertGroup(layer int, stem string, rows []v41ExpertRow, inputs [][]float32, picks [][]routePick, outputs [][][]float32, cfg Config, st *v41ForwardState, triple v41RoutedTripleLoader, downWeight v41RoutedDownLoader, scratch *v41ProjScratch) (bool, error) {
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	if len(rows) < 2 || I <= 0 || H <= 0 || I > H/2 || H > v41RoutedInputMaxBytes/4/v41ExpertCohortRows {
		return false, nil
	}
	work, ok := checkedMulInt(I, H)
	if !ok {
		return false, nil
	}
	if st != nil && (st.routedExpert != nil || st.expertGateUp != nil || st.expertDown != nil || st.gatherOps.bf16PublishFunc() != nil) {
		return false, nil
	}
	// Identity publication is essential: the cohort reads the immutable
	// normalized layer rows, not a future callback's result. Refusal has no
	// effects and leaves every payload/error boundary on the original path.
	for _, row := range rows {
		if len(inputs[row.Token]) != H || !v41BF16Widened(inputs[row.Token]) || !finite32(picks[row.Token][row.Slot].weight) {
			return false, nil
		}
	}
	// A whole group must fit one cohort: otherwise row streaming would
	// repeat dequantization. Unsupported metadata keeps the original triple.
	var raw *v41HostRawGateUp
	if len(rows) <= v41ExpertCohortRows {
		raw = m.v41PlanHostRawGateUp(layer, stem, scratch, H, I)
	}
	if raw != nil {
		triple = func() ([]float32, []float32, []float32, error) {
			return raw.load(m, layer, scratch)
		}
	}
	// The gate/up intermediates occupy the already-required H-wide output
	// rows (2*I <= H). One I-wide copy preserves the down operand while its
	// output overwrites that row; no returned row aliases this reusable copy.
	var operand []float32
	for first := 0; first < len(rows); {
		last := min(first+v41ExpertCohortRows, len(rows))
		if len(rows)-last == 1 {
			last--
		}
		count := last - first
		cohort := v41HostExpertCohort{model: m, raw: raw, hidden: H, inner: I, count: count, parallel: work >= parThreshold, operand: operand}
		for r, row := range rows[first:last] {
			cohort.inputs[r] = inputs[row.Token]
		}
		for r, row := range rows[first:last] {
			cohort.row, cohort.stage = r, 0
			output, err := m.v41FullRoutedExpertProjected(layer, stem, inputs[row.Token], picks[row.Token][row.Slot].weight, cfg,
				nil, nil, nil, triple, downWeight, true, nil, &cohort)
			if err != nil {
				return true, err
			}
			outputs[row.Token][row.Slot] = output
		}
		operand = cohort.operand
		first = last
	}
	return true, nil
}

type v41HostExpertCohort struct {
	model                *Model
	raw                  *v41HostRawGateUp
	hidden, inner, count int
	row, stage           int
	parallel             bool
	gateReady, upReady   bool
	inputs, outputs      [v41ExpertCohortRows][]float32
	operand              []float32
	weightRows           []float32
}

func (c *v41HostExpertCohort) project(w, x []float32, out, in int) []float32 {
	stage := c.stage
	c.stage++
	if stage < 2 {
		if c.outputs[0] == nil {
			// Keep allocation after the original first-row input and triple
			// validation, at the first host projection boundary.
			storage := make([]float32, c.count*c.hidden)
			for r := 0; r < c.count; r++ {
				c.outputs[r] = storage[r*c.hidden : (r+1)*c.hidden : (r+1)*c.hidden]
			}
		}
		ready, offset := &c.gateReady, 0
		if stage == 1 {
			ready, offset = &c.upReady, c.inner
		}
		if !*ready {
			// Hold one weight row while evaluating every cohort input. Each
			// FDOT is the unchanged matRows/parMatRows reduction; neither the
			// arithmetic order within a dot nor the activation graph changes.
			var quant *kQuantTensor
			if c.raw != nil {
				quant = c.raw.weights[stage]
			}
			if quant == nil {
				c.run(out, func(lo, hi int) {
					for o := lo; o < hi; o++ {
						weight := w[o*in : (o+1)*in]
						for r := 0; r < c.count; r++ {
							c.outputs[r][offset+o] = fdot(weight, c.inputs[r])
						}
					}
				})
			} else {
				c.projectRawGateUp(quant, offset)
				// Count actual logical f32 values produced, not a retained panel.
				c.model.v41NoteExpertStreamedDequant(int64(out) * int64(in) * 4)
			}
			*ready = true
		}
		return c.outputs[c.row][offset : offset+out : offset+out]
	}
	// The projected gate became the weighted down operand in place. Detach
	// it before writing any down output element into the same output row.
	if c.operand == nil {
		c.operand = make([]float32, c.inner)
	}
	copy(c.operand, x)
	y := c.outputs[c.row]
	c.run(out, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			y[o] = fdot(w[o*in:(o+1)*in], c.operand)
		}
	})
	return y
}

// A task owns one H-wide row buffer and advances across 32 weight rows. The
// same task-indexed slab is reused for up after gate's workers have joined;
// dynamic parFor scheduling never changes ownership of a task's buffer.
const v41HostRawWeightRows = 32

func (c *v41HostExpertCohort) projectRawGateUp(q *kQuantTensor, offset int) {
	tasks := q.out / v41HostRawWeightRows
	if q.out%v41HostRawWeightRows != 0 {
		tasks++
	}
	if c.weightRows == nil {
		c.weightRows = make([]float32, tasks*q.in)
	}
	bb, rb := q.kind.blockBytes(), q.rowBytes()
	c.run(tasks, func(lo, hi int) {
		for task := lo; task < hi; task++ {
			weight := c.weightRows[task*q.in : (task+1)*q.in : (task+1)*q.in]
			first := task * v41HostRawWeightRows
			last := min(first+v41HostRawWeightRows, q.out)
			for o := first; o < last; o++ {
				rawRow := q.raw[o*rb : (o+1)*rb]
				for b := 0; b < q.nblk; b++ {
					kQuantDequantSuperBlock(weight[b*qkK:], rawRow[b*bb:(b+1)*bb], q.kind)
				}
				for r := 0; r < c.count; r++ {
					c.outputs[r][offset+o] = fdot(weight, c.inputs[r])
				}
			}
		}
	})
}

func (c *v41HostExpertCohort) run(out int, body func(int, int)) {
	if c.parallel {
		parFor(out, currentWorkerCount(), body)
	} else {
		body(0, out)
	}
}

// This distinct opt-in avoids changing default policy or silently widening a
// profiling path. It does not inherit the asynchronous-read opt-in.
const v41HostExpertRowStreamEnv = "FAK_V41_HOST_EXPERT_ROW_STREAM"

// v41HostRawGateUp carries immutable, fully faulted checkpoint bytes for one
// <=32-row expert group. No f32 gate/up panel is installed in model or scratch.
// The down projection keeps its existing one-materialization-per-group path.
type v41HostRawGateUp struct {
	names   [3]string
	entries [2]expertCheckpointEntry
	weights [2]*kQuantTensor
	down    []float32
	ready   bool
}

func (m *Model) v41PlanHostRawGateUp(layer int, stem string, scratch *v41ProjScratch, H, I int) *v41HostRawGateUp {
	if os.Getenv(v41HostExpertRowStreamEnv) != "1" || scratch == nil || m.expertCheckpoint == nil || H <= 0 || I <= 0 || H%qkK != 0 ||
		cap(scratch.exp1) != 0 || cap(scratch.exp3) != 0 {
		return nil
	}
	if _, ok := checkedMulInt(H, I); !ok {
		return nil
	}
	// Do not add raw lifetimes beside an already-retained pair of f32 scratch
	// panels. A fallback group may allocate those, disabling later streaming.
	plan := &v41HostRawGateUp{names: m.v41ExpertStemNames(layer, stem)}
	for _, name := range plan.names[:2] {
		if _, cached := scratch.layerExperts[name]; cached {
			return nil
		}
		if _, present := m.manifest[name]; present {
			return nil
		}
		if _, _, resident := m.residentShape(name); resident {
			return nil
		}
	}
	tier := m.expertCheckpoint
	tier.mu.Lock()
	defer tier.mu.Unlock()
	if tier.pool.Budget() != 0 {
		return nil
	}
	for i, name := range plan.names[:2] {
		e, ok := tier.index[name]
		if !ok || (e.quant != ExpertCheckpointQ2K && e.quant != ExpertCheckpointQ3K) || e.rows != I || e.cols != H || e.nblk != H/qkK || tier.overlays[e.fused] != nil {
			return nil
		}
		bb := q2kBlockBytes
		if e.quant == ExpertCheckpointQ3K {
			bb = q3kBlockBytes
		}
		if e.stride != int64(I)*int64(H/qkK)*int64(bb) {
			return nil
		}
		plan.entries[i] = e
	}
	return plan
}

func (p *v41HostRawGateUp) load(m *Model, layer int, scratch *v41ProjScratch) ([]float32, []float32, []float32, error) {
	if p.ready {
		return nil, nil, p.down, nil
	}
	fault := m.expertCheckpoint.fault
	if os.Getenv(v41HostExpertPrefetchEnv) == "1" {
		misses := p.names
		_, cached := scratch.layerExperts[misses[2]]
		_, present := m.manifest[misses[2]]
		_, _, resident := m.residentShape(misses[2])
		if cached || present || resident {
			misses[2] = ""
		}
		var join func()
		fault, join = m.expertCheckpoint.prefetchDemandJoined(misses)
		defer join()
	}
	for i, name := range p.names[:2] {
		opened := m.v41NowNanos()
		w, err := fault(name)
		if err != nil {
			cause := v41StageErr(v41StageMoE, layer, fmt.Errorf("%w: tensor %s: %w", ErrV41ForwardStage, name, err))
			return nil, nil, nil, v41StageErr(v41StageMoE, layer, cause)
		}
		if opened != 0 {
			m.v41NoteExpertFaultDoorNanos(m.v41NowNanos() - opened)
		}
		// Index construction plus a complete fault establishes all bounds
		// before any numeric projection. These concrete staged kinds have no
		// fallible/lazy dequant step: their existing superblock functions only
		// read this exact raw extent and write one proven-size row.
		e, q := p.entries[i], w.kq
		kind := kindQ2K
		if e.quant == ExpertCheckpointQ3K {
			kind = kindQ3K
		}
		if w.q4 != nil || q == nil || q.lazy != nil || q.kind != kind || q.out != e.rows || q.in != e.cols || q.nblk != e.nblk || int64(len(q.raw)) != e.stride {
			return nil, nil, nil, v41ExpertOperationErr(layer, "gate/up", errV41ExpertResult)
		}
		p.weights[i] = q
		// Raw resolution succeeded; dequant bytes are accounted only when
		// row scratch is actually produced by the projection below.
		m.v41NoteExpertTierFault(faultedRawBytes(w), 0)
	}
	if w, ok := scratch.v41LayerCacheGet(p.names[2]); ok {
		m.v41NoteExpertResidentHit()
		p.down = w
	} else {
		w, err := m.v41ExpertF32IntoNamedFault(layer, p.names[2], scratch.exp2, fault)
		if err != nil {
			return nil, nil, nil, v41StageErr(v41StageMoE, layer, err)
		}
		scratch.exp2, p.down = w, w
	}
	p.ready = true
	return nil, nil, p.down, nil
}

func (c *v41HostExpertCohort) gateUpShape(w1, w3 []float32, out, in int) bool {
	if c.raw == nil {
		return len(w1) == out*in && len(w3) == out*in
	}
	if !c.raw.ready || len(w1) != 0 || len(w3) != 0 {
		return false
	}
	for _, q := range c.raw.weights {
		if q == nil || q.out != out || q.in != in {
			return false
		}
	}
	return true
}

func (c *v41HostExpertCohort) combinedHostStage() bool {
	return c.raw != nil && !c.gateReady
}
