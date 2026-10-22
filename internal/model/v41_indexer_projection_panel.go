package model

// A cold prefill projects one index layer's query (indexer.wq_b) and head
// weights (indexer.weights_proj) as v41ProjPanelRowChunk-row panels, each just
// before its first row is scored. Each row is bit-identical to its per-row
// projection: the host panel is residentMatMulBatch, and a projection callback
// is offered multi-row work only for f32 manifest weights, where the device
// matmul's one-row and staged reductions share one operation order. Any other
// case keeps that leaf on the per-row path, including every case in which the
// per-row path would refuse, so each refusal keeps its site.

// v41IndexerPanelDisabled keeps both projections on the per-row path. Tests
// flip it to compare the two paths; production leaves it false.
var v41IndexerPanelDisabled bool

// v41IndexerPanelRow is one position's pre-projected raw rows. A nil field
// keeps that projection on the per-row path.
type v41IndexerPanelRow struct{ query, weights []float32 }

type v41IndexerProjectionPanel struct {
	m              *Model
	l              int
	project        v41DenseProjectionFunc
	qLat, hidden   [][]float32
	query, weights bool
	width, heads   int
	lo, hi         int
	qChunk, wChunk []float32
}

// v41IndexerProjectionPanels returns nil when neither leaf is admitted. It
// performs no projection; row projects each chunk on first use.
func (m *Model) v41IndexerProjectionPanels(l, seq int, qLatRows, preByPos [][]float32, scratch *v41ProjScratch) *v41IndexerProjectionPanel {
	cfg := m.Cfg
	heads, dim := cfg.IndexNHeads, cfg.IndexHeadDim
	width, valid := checkedMulInt(heads, dim)
	if v41IndexerPanelDisabled || seq < 2 || len(qLatRows) != seq || len(preByPos) != seq || scratch == nil || heads <= 0 || dim <= 0 || !valid {
		return nil
	}
	project := scratch.denseProjection()
	device := project != nil
	query := m.v41IndexerPanelAdmits(l, "indexer.wq_b.weight", qLatRows, width, cfg.QLoraRank, device) &&
		!m.v41IndexQueryMayPublish(scratch.projectionRoute, scratch.gatherOps, width)
	weights := m.v41IndexerPanelAdmits(l, "indexer.weights_proj.weight", preByPos, heads, cfg.HiddenSize, device)
	if !query && !weights {
		return nil
	}
	return &v41IndexerProjectionPanel{m: m, l: l, project: project, qLat: qLatRows, hidden: preByPos, query: query, weights: weights, width: width, heads: heads}
}

func (p *v41IndexerProjectionPanel) row(t int) (v41IndexerPanelRow, error) {
	var r v41IndexerPanelRow
	if p == nil {
		return r, nil
	}
	if t < p.lo || t >= p.hi {
		if err := p.projectChunk(t - t%v41ProjPanelRowChunk); err != nil {
			return r, err
		}
	}
	at := t - p.lo
	if p.query {
		r.query = p.qChunk[at*p.width : (at+1)*p.width : (at+1)*p.width]
	}
	if p.weights {
		r.weights = p.wChunk[at*p.heads : (at+1)*p.heads : (at+1)*p.heads]
	}
	return r, nil
}

func (p *v41IndexerProjectionPanel) projectChunk(lo int) error {
	cfg := p.m.Cfg
	hi := min(lo+v41ProjPanelRowChunk, len(p.qLat))
	p.lo, p.hi, p.qChunk, p.wChunk = 0, 0, nil, nil
	var qChunk, wChunk []float32
	var err error
	if p.query {
		qChunk, err = p.m.v41ProjPanelWithProjection(p.l, "indexer.wq_b.weight", v41IndexerPanelInput(p.qLat[lo:hi], cfg.QLoraRank), p.width, cfg.QLoraRank, hi-lo, p.project)
		if err != nil {
			return err
		}
	}
	if p.weights {
		wChunk, err = p.m.v41ProjPanelWithProjection(p.l, "indexer.weights_proj.weight", v41IndexerPanelInput(p.hidden[lo:hi], cfg.HiddenSize), p.heads, cfg.HiddenSize, hi-lo, p.project)
		if err != nil {
			return err
		}
	}
	p.lo, p.hi, p.qChunk, p.wChunk = lo, hi, qChunk, wChunk
	return nil
}

// A selected device callback validates finiteness as a session-closing
// payload fault, so a non-finite row stays on its own per-row call.
func (m *Model) v41IndexerPanelAdmits(l int, leaf string, rows [][]float32, out, in int, device bool) bool {
	name := layerName(l, leaf)
	wr, wc, present := m.residentShape(name)
	if !present || wr != out || wc != in || in <= 0 {
		return false
	}
	if _, ok := checkedMulInt(v41ProjPanelRowChunk, in); !ok {
		return false
	}
	if _, ok := checkedMulInt(v41ProjPanelRowChunk, out); !ok {
		return false
	}
	if device && (!m.has(name) || m.lora != nil || m.prism != nil) {
		return false
	}
	for _, row := range rows {
		if len(row) != in {
			return false
		}
		if device {
			for _, v := range row {
				if !finite32(v) {
					return false
				}
			}
		}
	}
	return true
}

// v41IndexQueryMayPublish is a superset of v41IndexQueryProjection's fused
// projection-and-publication selection; those queries stay one row per call.
func (m *Model) v41IndexQueryMayPublish(b *v41ProjectionRouteBinding, ops *v41DeviceGatherOps, width int) bool {
	return b != nil && ops != nil && ops.bf16Publish != nil && b.nativeProjection.core != nil && b.nativeProjection.owner != nil &&
		width == 8192 && m.Cfg.IndexNHeads == 64 && m.Cfg.IndexHeadDim == 128 && m.lora == nil && m.prism == nil
}

func v41IndexerPanelInput(rows [][]float32, in int) []float32 {
	flat := make([]float32, len(rows)*in)
	for t, row := range rows {
		copy(flat[t*in:(t+1)*in], row)
	}
	return flat
}
