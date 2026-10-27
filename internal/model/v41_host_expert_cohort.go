package model

// v41HostExpertProjection is private to the callback-free full host expert
// path. Its three calls are gate, up, and down, with all original publication
// and arithmetic boundaries between them retained by the scalar consumer.
type v41HostExpertProjection func(w, x []float32, out, in int) []float32

// v41ContractHostExpertGroup batches only the pure gate/up contractions. Raw
// future-row projections may be computed early, but their BF16 publication,
// validation, activation, weighting and down contraction remain row ordered.
func (m *Model) v41ContractHostExpertGroup(layer int, stem string, rows []v41ExpertRow, inputs [][]float32, picks [][]routePick, outputs [][][]float32, cfg Config, st *v41ForwardState, triple v41RoutedTripleLoader, downWeight v41RoutedDownLoader) (bool, error) {
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
		cohort := v41HostExpertCohort{hidden: H, inner: I, count: count, parallel: work >= parThreshold, operand: operand}
		for r, row := range rows[first:last] {
			cohort.inputs[r] = inputs[row.Token]
		}
		for r, row := range rows[first:last] {
			cohort.row, cohort.stage = r, 0
			output, err := m.v41FullRoutedExpertProjected(layer, stem, inputs[row.Token], picks[row.Token][row.Slot].weight, cfg,
				nil, nil, nil, triple, downWeight, true, nil, cohort.project)
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
	hidden, inner, count int
	row, stage           int
	parallel             bool
	gateReady, upReady   bool
	inputs, outputs      [v41ExpertCohortRows][]float32
	operand              []float32
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
			body := func(lo, hi int) {
				for o := lo; o < hi; o++ {
					weight := w[o*in : (o+1)*in]
					for r := 0; r < c.count; r++ {
						c.outputs[r][offset+o] = fdot(weight, c.inputs[r])
					}
				}
			}
			c.run(out, body)
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

func (c *v41HostExpertCohort) run(out int, body func(int, int)) {
	if c.parallel {
		parFor(out, currentWorkerCount(), body)
	} else {
		body(0, out)
	}
}
