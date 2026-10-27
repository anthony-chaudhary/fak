package model

import "os"

// This adds up to 2*C*I float32 gate/up values to the host shared-expert live
// set. The byte cap is not a planner reservation or a whole-process/RSS bound.
// Operators must plan that headroom before opting in; the default stays scalar.
var v41HostSharedCohortEnabled = os.Getenv("FAK_V41_HOST_SHARED_COHORT") == "1"

// v41PlanHostSharedCohort selects only an immutable, callback-free full host
// path. Metadata refusal has no payload, publication, numeric-buffer or I/O effects.
// Lazy/NUMA/SDOT/adapted and all device/custom callback paths remain unchanged.
func (m *Model) v41PlanHostSharedCohort(in v41LayerMoEInput, panel v41LayerFFNPanel) *v41HostSharedCohort {
	if !v41HostSharedCohortEnabled || !in.full || in.resident != nil || in.mhcDevice != nil ||
		in.scratch == nil || m.lora != nil || m.prism != nil {
		return nil
	}
	scratch := in.scratch
	if scratch.denseProjection() != nil || scratch.sharedActivation != nil ||
		scratch.gatherOps.bf16PublishFunc() != nil || scratch.sharedAdd != nil {
		return nil
	}
	H, I, rows := m.Cfg.HiddenSize, m.Cfg.MoEIntermediateSize, len(in.x)
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full || H <= 0 || I <= 0 || in.H != H || rows < 2 || len(panel.inputs) != rows {
		return nil
	}
	count := min(rows, v41ExpertCohortRows)
	if I > v41RoutedInputMaxBytes/8/count {
		return nil
	}
	work, ok := checkedMulInt(I, H)
	if !ok {
		return nil
	}
	for _, row := range panel.inputs {
		// The unchanged scalar routine publishes a private copy of this row.
		// Only identity BF16 input lets future raw dots borrow it in advance.
		if len(row) != H || !v41BF16Widened(row) {
			return nil
		}
	}
	gate := m.v41HostSharedRawWeight(in.l, "ffn.shared_experts.w1.weight", I, H)
	up := m.v41HostSharedRawWeight(in.l, "ffn.shared_experts.w3.weight", I, H)
	if gate == nil || up == nil {
		return nil
	}
	return &v41HostSharedCohort{model: m, layer: in.l, hidden: H, inner: I, work: work,
		inputs: panel.inputs, weights: [2]*kQuantTensor{gate, up}}
}

func (m *Model) v41HostSharedRawWeight(layer int, leaf string, out, in int) *kQuantTensor {
	name := layerName(layer, leaf)
	// Match residentMatRowsBase precedence, not just presence in the k-quant map.
	if m.has(name) || m.q8w[name] != nil || m.q4w[name] != nil || m.q4kw[name] != nil {
		return nil
	}
	qt := m.kqw[name]
	if qt == nil || qt.out != out || qt.in != in || qt.lazy != nil || qt.denseBound != nil ||
		qt.numaPool != nil || qt.replicas != nil || qt.w3MLP || in%qkK != 0 || qt.nblk != in/qkK {
		return nil
	}
	switch qt.kind {
	case kindQ2K, kindQ3K, kindQ5K, kindQ6K:
	default:
		return nil
	}
	if kQuantSDOTEnabled(qt.kind) {
		return nil
	}
	stride, strideOK := checkedMulInt(qt.nblk, qt.kind.blockBytes())
	size, sizeOK := checkedMulInt(out, stride)
	if !strideOK || !sizeOK || len(qt.raw) != size {
		return nil
	}
	return qt
}

type v41HostSharedCohort struct {
	model                      *Model
	layer, hidden, inner, work int
	first, count, row          int
	inputs                     [][]float32
	weights                    [2]*kQuantTensor
	values                     [2][]float32
	ready                      [2]bool
}

// projection changes only the private raw projection supplier. Input/gate/up
// BF16, activation, down, shared add, trace and post-mix stay in their original
// per-row routine and call order. In particular up never runs before the first
// row's gate publication succeeds, and down is never prefetched or batched.
func (c *v41HostSharedCohort) projection(row int) v41DenseProjectionFunc {
	if row == c.first+c.count {
		c.first = row
		c.count = min(v41ExpertCohortRows, len(c.inputs)-row)
		if len(c.inputs)-row-c.count == 1 {
			c.count--
		}
		c.ready = [2]bool{}
	}
	c.row = row - c.first
	return c.project
}

func (c *v41HostSharedCohort) project(layer int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
	stage := -1
	switch leaf {
	case "ffn.shared_experts.w1.weight":
		stage = 0
	case "ffn.shared_experts.w3.weight":
		stage = 1
	}
	if stage < 0 || layer != c.layer || rows != 1 || out != c.inner || in != c.hidden || len(input) != in {
		return nil, v41ProjectionDeclined, nil
	}
	if !c.ready[stage] {
		c.projectStage(stage)
		c.ready[stage] = true
	}
	lo := c.row * c.inner
	// Original host BF16 publication copies this capped view before activation.
	// Neither the reusable panel nor a future row is exposed to any callback.
	return c.values[stage][lo : lo+c.inner : lo+c.inner], v41ProjectionHandled, nil
}

func (c *v41HostSharedCohort) projectStage(stage int) {
	opened, succeeded := c.model.v41NowNanos(), 0
	defer func() {
		// One actual host projection wave, C actually computed raw rows. No
		// cached-return call, device attribution or fabricated split timing.
		c.model.v41NoteDenseProjection(0, 1, 0, succeeded, 0, 0, opened)
	}()
	n := c.count * c.inner
	if cap(c.values[stage]) < n {
		c.values[stage] = make([]float32, n)
	} else {
		c.values[stage] = c.values[stage][:n]
	}
	qt, y := c.weights[stage], c.values[stage]
	bb, stride := qt.kind.blockBytes(), qt.rowBytes()
	inputs := c.inputs[c.first : c.first+c.count]
	// Match the original scalar route's worker budget AND work threshold;
	// batching must not silently replace its bounded decode-width dispatch.
	parForRangeWorkers(qt.out, c.work, kQuantDecodeWorkers(), func(lo, hi int) {
		var block [qkK]float32
		var acc [v41ExpertCohortRows]float32
		for o := lo; o < hi; o++ {
			for r := range inputs {
				acc[r] = 0
			}
			raw := qt.raw[o*stride : (o+1)*stride]
			for b := 0; b < qt.nblk; b++ {
				kQuantDequantSuperBlock(block[:], raw[b*bb:(b+1)*bb], qt.kind)
				for r, input := range inputs {
					x := input[b*qkK : (b+1)*qkK]
					// Exact kQuantMatRowsRangeRaw four-accumulator order,
					// followed by the original ascending super-block fold.
					var s0, s1, s2, s3 float32
					for i := 0; i < qkK; i += 4 {
						s0 += block[i] * x[i]
						s1 += block[i+1] * x[i+1]
						s2 += block[i+2] * x[i+2]
						s3 += block[i+3] * x[i+3]
					}
					acc[r] += (s0 + s1) + (s2 + s3)
				}
			}
			for r := range inputs {
				y[r*c.inner+o] = acc[r]
			}
		}
	})
	succeeded = c.count
}
