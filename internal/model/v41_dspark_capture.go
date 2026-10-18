package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var errV41DSparkCapture = errors.New("model: invalid V4.1 DSpark target capture")
var errV41DSparkMeanRangeUnavailable = errors.New("model: V4.1 DSpark bounded resident mean is unavailable")

// v41DSparkMeanRangeBackend is a structural contract, NOT a claim that a real
// backend implements it. A separate HAL leaf and hardware witness are required.
// The input is borrowed F32/BF16-valued [sourceRows,4,H] device storage. Only
// [first,first+count) may be materialized; 1 <= count <= 128. Return fresh owned
// F32 [count,H], or a partial fresh output plus error which the caller releases.
// For each element, add streams 0,1,2,3 in that order in F32 starting at +0,
// multiply the final sum by .25, then round BF16 nearest-even (the reviewed native scalar reference order,
// not a claim of GPU reduction-order identity). Refuse nonfinite
// inputs/intermediates/results. Quarter-weighted V41MHCPre is not equivalent
// when the unscaled sum overflows. No full-panel read, alias, or hidden host
// reconstruction is permitted. Supports must be side-effect-free.
// This operation is independent of target NVFP4 versus draft MXFP4 weights.
type v41DSparkMeanRangeBackend interface {
	SupportsV41DSparkMeanRange(compute.Tensor, int, int) bool
	V41DSparkMeanRange(compute.Tensor, int, int) (compute.Tensor, error)
}

// v41DSparkCapture is operation-local, exclusively owned Session scratch. It is
// never snapshot data, a scheduler setting, or an automatically enabled capture.
// The model pointer/held weight owner, state pointer, complete prefix digest,
// copied tap IDs and exact retained-round pointer bind every row to one lineage.
// Physical checkpoint digests remain a separate admission requirement.
// Close must be deferred by the adapter, including after a panic or cancellation.
type v41DSparkCapture struct {
	ctx                                 context.Context
	owner                               *Session
	model                               *Model
	state, run                          *v41ForwardState
	round                               *v41StepRound
	taps                                [3]int
	width, layers, base, start          int
	end                                 int
	baseHash, finalHash                 [32]byte
	tokens                              []int
	rows                                []v41DSparkContextRow
	seen                                []uint8
	cold, coldSucceeded, failed, closed bool
}

// appended is the expected new target input. tail selects the last at most 128
// absolute positions of the prospective prefix, including old history if needed.
// A retained round instead captures its <=6 verification inputs from base onward.
// No target state or draft ring is advanced by construction.
func newV41DSparkCapture(ctx context.Context, owner *Session, appended []int, tail bool, round *v41StepRound) (*v41DSparkCapture, error) {
	if ctx == nil || owner == nil || owner.M == nil || !owner.modelWeightsHeld || owner.halClosed ||
		owner.v41Forward == nil || owner.v41Forward.callbackOwner != owner {
		return nil, errV41DSparkCapture
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m, st := owner.M, owner.v41Forward
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full || !m.Cfg.IsDeepSeekV41() || m.Cfg.DeepSeekV41 == nil || m.Cfg.DeepSeekV41.DSpark == nil ||
		st.dsparkCapture != nil || st.activeStepRound != round || m.Cfg.HiddenSize < 1 || m.Cfg.HiddenSize > 5120 || m.Cfg.NumLayers > 40 {
		return nil, errV41DSparkCapture
	}
	taps := m.Cfg.DeepSeekV41.DSpark.TargetLayerIDs
	if len(taps) != 3 || taps[0] < 0 || taps[0] >= taps[1] || taps[1] >= taps[2] || taps[2] >= m.Cfg.NumLayers {
		return nil, errV41DSparkCapture
	}
	base := len(st.history)
	if len(appended) > int(^uint(0)>>1)-base || (!tail && (len(appended) < 1 || len(appended) > v41WindowSize)) {
		return nil, errV41DSparkCapture
	}
	end, start := base+len(appended), base
	if tail {
		start = max(0, end-v41WindowSize)
	}
	if end <= start || (round != nil && (tail || len(appended) > 6 || !round.ownsState() || round.model != m ||
		round.state != st || round.baseHistoryLen != base || round.pending != nil || len(round.entries) != 0 || round.capacity < len(appended))) {
		return nil, errV41DSparkCapture
	}
	for _, ids := range [][]int{st.history, appended} {
		for _, id := range ids {
			if id < 0 || id >= m.Cfg.VocabSize {
				return nil, errV41DSparkCapture
			}
		}
	}
	c := &v41DSparkCapture{ctx: ctx, owner: owner, model: m, state: st, round: round,
		taps: [3]int{taps[0], taps[1], taps[2]}, width: m.Cfg.HiddenSize, layers: m.Cfg.NumLayers, base: base, start: start, end: end,
		baseHash: v41DSparkPrefix(st.history), finalHash: v41DSparkCapturePrefix(st.history, appended),
		tokens: make([]int, end-start), rows: make([]v41DSparkContextRow, end-start), seen: make([]uint8, end-start)}
	for i := range c.tokens {
		pos := start + i
		if pos < base {
			c.tokens[i] = st.history[pos]
		} else {
			c.tokens[i] = appended[pos-base]
		}
		c.rows[i] = v41DSparkContextRow{Position: pos, Token: c.tokens[i], Layers: append([]int(nil), c.taps[:]...), Hidden: make([][]float32, 3)}
	}
	st.dsparkCapture = c
	return c, nil
}

// Hash the whole lineage without allocating/copying the whole token prefix.
func v41DSparkCapturePrefix(parts ...[]int) [32]byte {
	h := sha256.New()
	var b [8]byte
	for _, ids := range parts {
		for _, id := range ids {
			binary.LittleEndian.PutUint64(b[:], uint64(id))
			_, _ = h.Write(b[:])
		}
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func (c *v41DSparkCapture) check() error {
	if c == nil || c.closed || c.failed || c.owner == nil || c.owner.M != c.model || !c.owner.modelWeightsHeld ||
		c.owner.halClosed || c.owner.v41Forward != c.state || c.state.dsparkCapture != c || c.state.callbackOwner != c.owner ||
		len(c.state.history) < c.base || v41DSparkPrefix(c.state.history[:c.base]) != c.baseHash ||
		c.model.Cfg.HiddenSize != c.width || c.model.Cfg.NumLayers != c.layers || c.model.Cfg.DeepSeekV41 == nil || c.model.Cfg.DeepSeekV41.DSpark == nil ||
		!slices.Equal(c.model.Cfg.DeepSeekV41.DSpark.TargetLayerIDs, c.taps[:]) {
		return errV41DSparkCapture
	}
	return c.ctx.Err()
}

func (c *v41DSparkCapture) fail(err error) error {
	c.failed = true
	return err
}

func (c *v41DSparkCapture) close() {
	if c == nil || c.closed {
		return
	}
	c.closed = true
	if c.state != nil && c.state.dsparkCapture == c {
		c.state.dsparkCapture = nil
	}
	if c.run != nil && c.run.dsparkCapture == c {
		c.run.dsparkCapture = nil
	}
	c.run, c.rows, c.seen, c.tokens = nil, nil, nil, nil
	c.owner, c.model, c.state, c.round, c.ctx = nil, nil, nil, nil, nil
}

// bindCold is called only after the existing history-rollback defer is armed.
func (c *v41DSparkCapture) bindCold(run *v41ForwardState) error {
	if err := c.check(); err != nil {
		return c.fail(err)
	}
	if c.round != nil || c.cold || c.run != nil || run == nil || run == c.state || run.dsparkCapture != nil ||
		run.callbackOwner != c.owner || len(run.history) != c.end || v41DSparkPrefix(run.history) != c.finalHash {
		return c.fail(errV41DSparkCapture)
	}
	for _, seen := range c.seen {
		if seen != 0 {
			return c.fail(errV41DSparkCapture)
		}
	}
	c.cold, c.run, run.dsparkCapture = true, run, c
	return nil
}

// runCold records success only after the entire target invocation returns
// normally. Panic (including nil), Goexit and deferred error all leave failure.
// This fences capture publication; it does not undo target state published by
// the body before a late cleanup fault. The adapter must recover or discard it.
func (c *v41DSparkCapture) runCold(body func() (*Activations, error)) (act *Activations, err error) {
	returned := false
	defer func() { c.finishCold(returned && err == nil) }()
	act, err = body()
	returned = true
	return act, err
}

func (c *v41DSparkCapture) finishCold(success bool) {
	if c == nil || c.closed {
		return
	}
	bound := c.run != nil && c.run.dsparkCapture == c
	if bound {
		c.run.dsparkCapture = nil
	}
	c.run = nil
	c.coldSucceeded = success && bound && c.cold && !c.failed
	if !c.coldSucceeded {
		c.failed = true
	}
}

func (c *v41DSparkCapture) tap(layer int) int {
	for i, l := range c.taps {
		if layer == l {
			return i
		}
	}
	return -1
}

// captureCold is deliberately between Engram and attention mHC pre-collapse.
func (c *v41DSparkCapture) captureCold(st *v41ForwardState, layer int, tokens []int, streams [][][]float32, resident *v41MHCResident, full bool) error {
	tap := c.tap(layer)
	if tap < 0 {
		return nil
	}
	if err := c.check(); err != nil {
		return c.fail(err)
	}
	if !full || !c.cold || c.run != st || st.dsparkCapture != c || c.round != nil || c.state.activeStepRound != nil ||
		len(tokens) != c.end || !slices.Equal(tokens[c.start:c.end], c.tokens) {
		return c.fail(errV41DSparkCapture)
	}
	return c.capture(layer, tap, 0, c.start, len(c.tokens), len(tokens), streams, resident)
}

func (c *v41DSparkCapture) captureStep(st *v41ForwardState, layer, token, position int, streams [][]float32, resident *v41MHCResident, full bool) error {
	tap := c.tap(layer)
	if tap < 0 {
		return nil
	}
	if err := c.check(); err != nil {
		return c.fail(err)
	}
	if !full || c.cold || c.run != nil || st != c.state || st.activeStepRound != c.round || position != len(st.history) ||
		position < c.start || position >= c.end || token != c.tokens[position-c.start] {
		return c.fail(errV41DSparkCapture)
	}
	if c.round != nil {
		r := c.round
		if !r.ownsState() || r.failed || r.poisoned || r.pending == nil || r.pending.token != token ||
			r.pending.historyLen != position || r.pending.ordinal != position-c.base {
			return c.fail(errV41DSparkCapture)
		}
	}
	return c.capture(layer, tap, position-c.start, 0, 1, 1, [][][]float32{streams}, resident)
}

func (c *v41DSparkCapture) capture(layer, tap, destination, first, count, sourceRows int, streams [][][]float32, resident *v41MHCResident) error {
	for i := destination; i < destination+count; i++ {
		if c.seen[i] != uint8((1<<tap)-1) {
			return c.fail(errV41DSparkCapture)
		}
	}
	var means [][]float32
	var err error
	if resident != nil {
		// Host streams can be shape-only placeholders when this panel is resident.
		means, err = c.residentMeans(layer, resident, first, count, sourceRows)
	} else {
		if c.owner.executionPolicy == ExecutionPolicyDeviceOnly || len(streams) != sourceRows {
			return c.fail(errV41DSparkCapture)
		}
		means = make([][]float32, count)
		for i := range means {
			means[i], err = v41DSparkMeanStreams(streams[first+i], c.width)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		return c.fail(err)
	}
	if err := c.ctx.Err(); err != nil {
		return c.fail(err)
	}
	for i, mean := range means {
		c.rows[destination+i].Hidden[tap] = mean
		c.seen[destination+i] |= 1 << tap
	}
	return nil
}

func (c *v41DSparkCapture) residentMeans(layer int, r *v41MHCResident, first, count, sourceRows int) ([][]float32, error) {
	if r.d == nil || r.d.s != c.owner || r.streams == nil || r.rows != sourceRows || r.hidden != c.width ||
		r.streams.Rows() != sourceRows || r.streams.Hidden() != c.width || count < 1 || count > v41WindowSize ||
		first < 0 || first > sourceRows-count || !v41DeviceStreamsShapeOK(c.owner.Backend, r.streams.Tensor()) {
		return nil, errV41DSparkCapture
	}
	input := r.streams.Tensor()
	op, ok := c.owner.Backend.(v41DSparkMeanRangeBackend)
	if !ok || !op.SupportsV41DSparkMeanRange(input, first, count) {
		// This is before selection. Never read the full panel or host placeholders.
		return nil, errV41DSparkMeanRangeUnavailable
	}
	var out [][]float32
	var primary error
	returned, bodyReturned := false, false
	defer func() {
		if returned || c.owner.halClosed {
			return
		}
		// Goexit cannot return through guard. Retire this incomplete selected
		// owner and preserve a known dispatch error before cleanup can unwind.
		cause := primary
		if cause == nil {
			cause = errV41MHCResult
		}
		if c.owner.halFailure == nil {
			c.owner.halFailure = &BackendForwardOperationError{Backend: c.owner.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-mhc-dspark-mean-range", Layer: layer, Stage: "incomplete", Cause: cause}
		}
		defer func() { c.owner.halClosed = true; _ = recover() }()
		c.owner.closeBackendOwned()
	}()
	err := r.d.guard(layer, "dspark-mean-range", func(stage *string) error {
		cause := func() (cause error) {
			var t compute.Tensor
			complete := false
			defer func() {
				primary = cause
				// Partial/error and malformed fresh outputs remain ours. A
				// secondary cleanup panic cannot replace an earlier error/unwind.
				if t.Buf() != nil && t.Backend() == c.owner.Backend && t.Buf() != input.Buf() {
					if !complete || cause != nil {
						defer func() { _ = recover() }()
					}
					c.owner.Backend.Free(t)
				}
			}()
			cause = func() error {
				*stage = "mean-range"
				var err error
				t, err = op.V41DSparkMeanRange(input, first, count)
				if err != nil {
					return err
				}
				if !v41MHCFreshOutput(c.owner, t, []int{count, c.width}, input) {
					return errV41MHCResult
				}
				*stage = "mean-range readback"
				flat, valid := compute.ReadV41MHCCollapse(c.owner.Backend, t, count*c.width)
				if !valid {
					return errV41MHCResult
				}
				for _, value := range flat {
					if math.Float32bits(value)&0xffff != 0 {
						return errV41MHCResult
					}
				}
				out = make([][]float32, count)
				for i := range out {
					out[i] = flat[i*c.width : (i+1)*c.width : (i+1)*c.width]
				}
				return nil
			}()
			complete = true
			return cause
		}()
		bodyReturned = true
		return cause
	})
	if err == nil && !bodyReturned {
		// The existing guard can recover legacy panic(nil). Keep that unwind
		// and retire the selected owner instead of reporting successful rows.
		_ = r.d.fail(layer, "dspark-mean-range", "incomplete", errV41MHCResult)
		panic(nil)
	}
	if err != nil {
		out = nil
	}
	returned = true
	return out, err
}

// The final cancellation/completeness check runs before target publication,
// including no-logits steps. No capture work is installed on the ordinary path.
func (c *v41DSparkCapture) beforeCommit(position int) error {
	if err := c.check(); err != nil {
		return c.fail(err)
	}
	if c.cold {
		if c.run == nil || len(c.state.history) != c.end || v41DSparkPrefix(c.state.history) != c.finalHash {
			return c.fail(errV41DSparkCapture)
		}
		for _, seen := range c.seen {
			if seen != 7 {
				return c.fail(errV41DSparkCapture)
			}
		}
	} else if position != len(c.state.history) || position < c.start || position >= c.end ||
		c.seen[position-c.start] != 7 || c.state.activeStepRound != c.round {
		return c.fail(errV41DSparkCapture)
	}
	return nil
}

// committed transfers owned rows only after the complete expected target prefix
// succeeded. Later head failure, restore, cancellation, or open rounds refuse.
func (c *v41DSparkCapture) committed() ([]v41DSparkContextRow, error) {
	if err := c.check(); err != nil {
		return nil, err
	}
	if c.round != nil || c.run != nil || (c.cold && !c.coldSucceeded) || c.state.activeStepRound != nil || len(c.state.history) != c.end ||
		v41DSparkPrefix(c.state.history) != c.finalHash {
		return nil, errV41DSparkCapture
	}
	for _, seen := range c.seen {
		if seen != 7 {
			return nil, errV41DSparkCapture
		}
	}
	rows := c.rows
	c.rows = nil
	c.close()
	return rows, nil
}

// staged lends immutable rows exclusively to draft PREPARATION while the target
// journal still owns the whole speculative suffix. This is not a public commit.
// The adapter must prepare every projection/hash, finish(keep), then perform only
// infallible draft assignments. A failed/poisoned drain never publishes the draft.
func (c *v41DSparkCapture) staged(round *v41StepRound, keep int) ([]v41DSparkContextRow, error) {
	if err := c.check(); err != nil {
		return nil, err
	}
	if round == nil || c.round != round || !round.ownsState() || round.failed || round.poisoned || round.pending != nil ||
		!round.atCommittedBoundary() || len(c.state.history) != c.end || v41DSparkPrefix(c.state.history) != c.finalHash ||
		keep < 1 || keep > len(c.rows) || round.committedCount() != len(c.rows) {
		return nil, errV41DSparkCapture
	}
	for i, seen := range c.seen {
		entry := round.entries[i]
		if seen != 7 || !entry.committed || !entry.closed || !entry.cleanupOK || entry.token != c.tokens[i] {
			return nil, fmt.Errorf("%w: incomplete retained row %d", errV41DSparkCapture, i)
		}
	}
	return c.rows[:keep:keep], nil
}
