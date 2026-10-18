package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A row's lane is its original request index, not its compacted survivor index.
// Input remains borrowed and stable until Run returns. No Session is copied.
type v41GroupedOutputCohortRow struct {
	Lane    int
	Session *Session
	Input   []float32
}

type v41GroupedOutputCohortLane struct {
	row     v41GroupedOutputCohortRow
	epoch   uint64
	forward *v41ForwardState
}

// v41GroupedOutputCohort owns a single synchronous cross-request projection.
// Its caller already owns the live backend scope and exclusively owns every
// Session, exactly as forwardV41StepCohortOwned requires. Epoch checks detect
// stale state; they are not locks and do not make concurrent Session use safe.
//
// Immutable handles are borrowed from the first session's cache only during
// Run. cachedImmutableWeight either pins them through the model residency and
// the session's weight hold, or stores private handles in that session's halW.
// No participant may close/restore while this operation runs. Tile activation
// buffers are drained before selected failure retires the participating owners.
// Every output row is copied before buffers are freed; no loan escapes Run.
type v41GroupedOutputCohort struct {
	scope                                           *backendExecutionScope
	model                                           *Model
	backend                                         compute.Backend
	lanes                                           [4]v41GroupedOutputCohortLane
	count, layer, heads, headDim, groups, rank, dim int
	ran, closed, selected, arithmeticSelected       bool
}

func prepareV41GroupedOutputCohort(scope *backendExecutionScope, layer int, rows []v41GroupedOutputCohortRow) (*v41GroupedOutputCohort, v41DenseProjectionOutcome, error) {
	if scope == nil || !scope.live || scope.owner == nil || (len(rows) != 2 && len(rows) != 4) {
		return nil, v41ProjectionDeclined, nil
	}
	leader := rows[0].Session
	if leader == nil || leader.M == nil || leader.Backend == nil || !backendHasComparableIdentity(leader.Backend) || leader.M.lora != nil || leader.M.prism != nil {
		return nil, v41ProjectionDeclined, nil
	}
	if backendExecutionOwner(leader.Backend) != scope.owner {
		return nil, v41ProjectionDeclined, nil
	}
	cfg := leader.M.Cfg
	if layer < 0 || layer >= cfg.NumLayers {
		return nil, v41ProjectionDeclined, nil
	}
	width, ok := checkedMulInt(cfg.NumHeads, cfg.HeadDim)
	if !ok || cfg.NumHeads <= 0 || cfg.HeadDim <= 0 || width <= 0 || cfg.OGroups <= 0 || cfg.OLoraRank <= 0 || cfg.HiddenSize <= 0 || cfg.NumHeads%cfg.OGroups != 0 {
		return nil, v41ProjectionDeclined, nil
	}
	// Bound every packed row geometry and its byte size before allocating the
	// operation's input panel or selecting a backend operation.
	joined, joinedOK := checkedMulInt(cfg.OGroups, cfg.OLoraRank)
	if !joinedOK || joined <= 0 {
		return nil, v41ProjectionDeclined, nil
	}
	for _, rowWidth := range []int{width, joined, cfg.OLoraRank, cfg.HiddenSize} {
		elements, valid := checkedMulInt(len(rows), rowWidth)
		if !valid {
			return nil, v41ProjectionDeclined, nil
		}
		if _, valid = checkedMulInt(elements, 4); !valid {
			return nil, v41ProjectionDeclined, nil
		}
	}
	owner := &v41GroupedOutputCohort{scope: scope, model: leader.M, backend: leader.Backend, count: len(rows), layer: layer, heads: cfg.NumHeads, headDim: cfg.HeadDim, groups: cfg.OGroups, rank: cfg.OLoraRank, dim: cfg.HiddenSize}
	last := -1
	for i, row := range rows {
		s := row.Session
		if row.Lane <= last || s == nil || s.M != owner.model || s.Backend != owner.backend || s.backendExecutionScope != scope || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || len(row.Input) != width {
			return nil, v41ProjectionDeclined, nil
		}
		for j := 0; j < i; j++ {
			if owner.lanes[j].row.Session == s {
				return nil, v41ProjectionDeclined, nil
			}
		}
		for _, v := range row.Input {
			if !finite32(v) {
				return nil, v41ProjectionDeclined, nil
			}
		}
		owner.lanes[i] = v41GroupedOutputCohortLane{row: row, epoch: s.cacheGeometryEpoch, forward: s.v41Forward}
		last = row.Lane
	}
	return owner, v41ProjectionHandled, nil
}

func (o *v41GroupedOutputCohort) valid() bool {
	if o == nil || o.closed || o.scope == nil || !o.scope.live || o.scope.owner == nil || o.model == nil {
		return false
	}
	cfg := o.model.Cfg
	if cfg.NumHeads != o.heads || cfg.HeadDim != o.headDim || cfg.OGroups != o.groups || cfg.OLoraRank != o.rank || cfg.HiddenSize != o.dim || o.model.lora != nil || o.model.prism != nil {
		return false
	}
	for i := 0; i < o.count; i++ {
		lane := &o.lanes[i]
		s := lane.row.Session
		if s.M != o.model || s.Backend != o.backend || s.backendExecutionScope != o.scope || s.cacheGeometryEpoch != lane.epoch || s.v41Forward != lane.forward || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld {
			return false
		}
	}
	return true
}

func (o *v41GroupedOutputCohort) close() {
	if o != nil {
		o.closed = true
	}
}

func (o *v41GroupedOutputCohort) Run() (result [][]float32, outcome v41DenseProjectionOutcome, err error) {
	if o == nil || o.ran || o.closed {
		return nil, v41ProjectionError, fmt.Errorf("%w: grouped output cohort is not pending", ErrV41ForwardStage)
	}
	o.ran = true
	complete := false
	defer func() {
		if o.selected && (!complete || err != nil) {
			cause := err
			if cause == nil {
				cause = o.lanes[0].row.Session.halFailure
			}
			if cause == nil {
				cause = errors.New("model: incomplete selected grouped output cohort")
			}
			failure := o.retire(cause)
			if err != nil {
				err = failure
				result = nil
				outcome = v41ProjectionError
			}
		}
		o.close()
	}()
	result, outcome, err = func() ([][]float32, v41DenseProjectionOutcome, error) {
		if !o.valid() {
			return nil, v41ProjectionDeclined, nil
		}
		width := o.heads * o.headDim
		panel := make([]float32, 0, o.count*width)
		for i := 0; i < o.count; i++ {
			row := o.lanes[i].row.Input
			if len(row) != width {
				return nil, v41ProjectionDeclined, nil
			}
			for _, v := range row {
				if !finite32(v) {
					return nil, v41ProjectionDeclined, nil
				}
			}
			panel = append(panel, row...)
		}
		result, outcome, err = o.lanes[0].row.Session.v41GroupedOutputPanelOwned(o, o.layer, panel, o.count, o.heads, o.headDim, o.groups, o.rank, o.dim)
		if err != nil {
			return nil, outcome, err
		}
		if !o.valid() {
			if o.selected {
				return nil, v41ProjectionError, o.retire(errors.New("model: grouped output cohort changed during selected operation"))
			}
			return nil, v41ProjectionDeclined, nil
		}
		if outcome == v41ProjectionDeclined && o.arithmeticSelected {
			return nil, v41ProjectionError, errors.New("model: selected grouped arithmetic cannot decline")
		}
		// A normal capability decline after immutable weight staging is permitted:
		// no activation or arithmetic has been selected and no operation failed.
		return result, outcome, nil
	}()
	complete = true
	return result, outcome, err
}

// The nil-owner arm preserves the historical single-session selected boundary.
// The cohort arm extends terminal failure to every participant after the tile
// has drained its call-local resources. It never retries the scalar projector.
func v41GroupedPanelSelectedCall[T any](o *v41GroupedOutputCohort, s *Session, layer int, path string, run func(*string) (T, error)) (result T, err error) {
	if o == nil {
		return v41SelectedDeviceCall(s, layer, path, run)
	}
	var zero T
	o.selected = true
	if path == "v41-grouped-output-panel" {
		o.arithmeticSelected = true
	}
	complete := false
	defer func() {
		if !complete {
			cause := s.halFailure
			if cause == nil {
				cause = errors.New("model: incomplete grouped output cohort operation")
			}
			o.retire(cause)
		}
	}()
	result, err = v41SelectedDeviceCall(s, layer, path, func(stage *string) (T, error) {
		if !o.valid() {
			return zero, errors.New("model: grouped output cohort loan is stale")
		}
		value, cause := run(stage)
		if cause == nil && !o.valid() {
			return zero, errors.New("model: grouped output cohort loan changed during operation")
		}
		return value, cause
	})
	if err != nil {
		result = zero
		err = o.retire(err)
	}
	complete = true
	return result, err
}

func (o *v41GroupedOutputCohort) retire(cause error) error {
	failure := cause
	var forward *BackendForwardOperationError
	if !errors.As(cause, &forward) {
		failure = &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-grouped-output-cohort", Layer: o.layer, Stage: "selected projection", Cause: cause}
	}
	for i := 0; i < o.count; i++ {
		s := o.lanes[i].row.Session
		if s.halFailure == nil {
			s.halFailure = failure
		}
		if !s.halClosed {
			func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
		}
	}
	return failure
}

// Cache keys and model identity select the immutable weights; tensor metadata
// must still name the exact backend and logical geometry before arithmetic.
func (o *v41GroupedOutputCohort) validWeights(w v41GroupedPanelWeights, groupIn int) bool {
	valid := func(t compute.Tensor, dt compute.Dtype, out, in int) bool {
		return t.Buf() != nil && t.Backend() == o.backend && t.Dtype == dt && t.Layout == compute.RowMajor && len(t.Shape) == 2 && t.Shape[0] == out && t.Shape[1] == in
	}
	if len(w.a) != o.groups || !valid(w.b, w.bDtype, o.dim, o.groups*o.rank) {
		return false
	}
	for _, a := range w.a {
		if !valid(a, w.aDtype, o.rank, groupIn) {
			return false
		}
	}
	return true
}
