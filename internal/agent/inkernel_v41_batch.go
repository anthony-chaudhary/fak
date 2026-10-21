package agent

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// The production V4.1 coordinator owns each request Session until this driver
// returns, preserves coalescedDecode's no-request-retry boundary, and delivers
// lane.err individually instead of broadcasting a peer's failure to every request.
// The planner gate may surround this call; the backend mutex must not, because
// StepV41Batch and public Session.Step acquire their own recording-domain owner.
func inKernelDecodeV41Lanes(ctx context.Context, lanes []*decodeLane, m *model.Model) (inKernelV41DecodeReceipt, error) {
	return inKernelDecodeV41LanesWithOps(ctx, lanes, inKernelV41DecodeOps{
		batch: m.StepV41Batch,
		serial: func(s *model.Session, token int) (logits []float32, err error) {
			defer func() {
				if primary := recover(); primary != nil {
					var handled bool
					err, handled = recoverDevicePanic(primary)
					if !handled {
						panic(primary)
					}
				}
			}()
			return s.Step(token), nil
		},
		wait: func(ctx context.Context) error {
			// Mutex ownership has no notification API. A bounded sleep avoids
			// spinning; every retry rechecks each request's cancellation.
			timer := time.NewTimer(time.Millisecond)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timer.C:
				return nil
			}
		},
	})
}

type inKernelV41DecodeReceipt struct {
	sharedPanels int
	sharedRows   int
}

// Test substitution is confined to this private driver. It is not an admission
// capability, and scripted results cannot qualify model arithmetic or serving.
type inKernelV41DecodeOps struct {
	batch  func([]model.V41BatchLane) (model.V41BatchResult, error)
	serial func(*model.Session, int) ([]float32, error)
	wait   func(context.Context) error
}

type inKernelV41Admission func() ([]*decodeLane, error)

type inKernelV41AdmissionContextKey struct{}

type inKernelV41Pending struct {
	lane    *decodeLane
	id      uint64
	token   int
	pending bool
}

func inKernelV41LaneContext(ctx context.Context, lane *decodeLane) context.Context {
	if lane.ctx != nil {
		return lane.ctx
	}
	return ctx
}

// lane.err is authoritative for each request. terminal retains an operation's
// cause for diagnosis; it must not be broadcast over successful peer results.
func inKernelDecodeV41LanesWithOps(ctx context.Context, lanes []*decodeLane, ops inKernelV41DecodeOps) (receipt inKernelV41DecodeReceipt, terminal error) {
	if len(lanes) == 0 {
		return receipt, nil
	}
	if len(lanes) > 4 || ops.batch == nil || ops.serial == nil || ops.wait == nil {
		return receipt, fmt.Errorf("agent: invalid V4.1 decode driver admission")
	}
	pending := make([]*inKernelV41Pending, 0, 4)
	seen := make(map[*model.Session]bool, 4)
	nextID := uint64(1)
	appendLanes := func(candidates []*decodeLane) error {
		if len(pending)+len(candidates) > 4 {
			return fmt.Errorf("agent: invalid V4.1 decode driver admission")
		}
		for _, lane := range candidates {
			if lane == nil || lane.s == nil || seen[lane.s] {
				return fmt.Errorf("agent: nil or aliased V4.1 decode lane")
			}
			seen[lane.s] = true
			pending = append(pending, &inKernelV41Pending{lane: lane, id: nextID})
			nextID++
		}
		return nil
	}
	if err := appendLanes(lanes); err != nil {
		return receipt, err
	}
	failPending := func(err error) {
		for _, p := range pending {
			if p.pending {
				p.lane.err, p.lane.done, p.pending = err, true, false
			}
		}
	}
	admit, _ := ctx.Value(inKernelV41AdmissionContextKey{}).(inKernelV41Admission)
	admitAtBoundary := false
	for {
		if admitAtBoundary && admit != nil && len(pending) < 4 {
			admitAtBoundary = false
			admitted, err := admit()
			if err != nil {
				return receipt, err
			}
			if err := appendLanes(admitted); err != nil {
				failPending(err)
				return receipt, err
			}
		}
		input := make([]model.V41BatchLane, 0, len(pending))
		byID := make(map[uint64]*inKernelV41Pending, len(pending))
		for _, p := range pending {
			laneCtx := inKernelV41LaneContext(ctx, p.lane)
			if p.pending {
				if err := laneCtx.Err(); err != nil {
					p.lane.err, p.lane.done, p.pending = err, true, false
				}
			} else if !p.lane.done && p.lane.gen < p.lane.maxNew {
				p.token, p.pending = p.lane.decodeOne(laneCtx)
			}
			if p.pending {
				input = append(input, model.V41BatchLane{ID: p.id, Context: laneCtx, Session: p.lane.s, Token: p.token})
				byID[p.id] = p
			}
		}
		if len(input) == 0 {
			return receipt, terminal
		}
		var traceNow func() time.Time
		for _, lane := range input {
			if measurement := byID[lane.ID].lane.measurement; measurement != nil && measurement.traceNow != nil {
				traceNow = measurement.traceNow
				break
			}
		}
		var traceStart time.Time
		if traceNow != nil {
			traceStart = traceNow()
		}
		started := time.Now()
		out, batchErr := ops.batch(input)
		elapsed := time.Since(started)
		var traceDuration time.Duration
		if traceNow != nil {
			traceDuration = traceNow().Sub(traceStart)
		}
		if err := inKernelV41ValidateResult(input, out); err != nil {
			failPending(err)
			return receipt, err
		}
		if !out.Selected && (errors.Is(batchErr, model.ErrV41BatchBusy) || errors.Is(batchErr, model.ErrV41BatchUnavailable)) {
			// Refusal is sufficient for fallback only with a completely untouched
			// result. A contradictory partial result can never authorize replay.
			for _, row := range out.Lanes {
				if row.Err != nil {
					failPending(batchErr)
					return receipt, batchErr
				}
			}
			if errors.Is(batchErr, model.ErrV41BatchBusy) {
				waitStart := time.Now()
				err := ops.wait(ctx)
				enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(waitStart))
				if err != nil {
					failPending(err)
					return receipt, err
				}
				continue // Keep the chosen token, RNG, counts, emit and gen unchanged.
			}
			committed := 0
			for _, lane := range input {
				p := byID[lane.ID]
				if err := lane.Context.Err(); err != nil {
					p.lane.err, p.lane.done, p.pending = err, true, false
					continue
				}
				var traceStart time.Time
				if traceNow != nil {
					traceStart = traceNow()
				}
				started := time.Now()
				logits, err := ops.serial(p.lane.s, p.token)
				elapsed := time.Since(started)
				var traceDuration time.Duration
				if traceNow != nil {
					traceDuration = traceNow().Sub(traceStart)
				}
				if err != nil {
					p.lane.err, p.lane.done, p.pending = err, true, false
					terminal = err
					continue
				}
				row := model.V41BatchLaneResult{ID: p.id, Logits: logits, Committed: true}
				inKernelV41ApplyRow(p, row, nil, NativeForwardSessionStep, 1, traceDuration)
				committed++
				enginestep.Default.ObserveDecodeStep(enginestep.PathSerial, 1, elapsed)
			}
			admitAtBoundary = committed > 0
			continue
		}
		if batchErr != nil {
			terminal = batchErr
		}
		receipt.sharedPanels += out.SharedHeadPanels
		receipt.sharedRows += out.SharedHeadRows
		committed := 0
		for _, row := range out.Lanes {
			if row.Committed {
				committed++
			}
			inKernelV41ApplyRow(byID[row.ID], row, batchErr, NativeForwardV41BatchStep, len(input), traceDuration)
		}
		enginestep.Default.ObserveDecodeStep(enginestep.PathBatched, committed, elapsed)
		admitAtBoundary = committed > 0
	}
}

func inKernelV41ValidateResult(input []model.V41BatchLane, out model.V41BatchResult) error {
	invalid := func() error { return fmt.Errorf("agent: invalid V4.1 batch result; no replay permitted") }
	if len(out.Lanes) != len(input) || out.SharedHeadPanels < 0 || out.SharedHeadRows < 0 ||
		(!out.Selected && (out.SharedHeadPanels != 0 || out.SharedHeadRows != 0)) {
		return invalid()
	}
	ids := make(map[uint64]bool, len(input))
	for _, lane := range input {
		ids[lane.ID] = true
	}
	for _, row := range out.Lanes {
		if !ids[row.ID] || (!out.Selected && (row.Committed || len(row.Logits) != 0)) {
			return invalid()
		}
		delete(ids, row.ID)
	}
	return nil
}

func inKernelV41ApplyRow(p *inKernelV41Pending, row model.V41BatchLaneResult, batchErr error, kind string, active int, duration time.Duration) {
	ln := p.lane
	p.pending = false
	if row.Committed {
		if ln.forwarded != nil {
			ln.forwarded = append(ln.forwarded, p.token)
		}
		ln.logits = row.Logits
		if err := ln.measurement.recordForwardTiming(ln.gen, kind, active, duration); err != nil && row.Err == nil {
			row.Err = err
		}
		if len(row.Logits) == 0 && row.Err == nil {
			row.Err = fmt.Errorf("agent: committed V4.1 lane has no logits; no replay permitted")
		}
	} else if row.Err == nil {
		row.Err = model.ErrV41BatchAborted
		if batchErr == nil {
			row.Err = fmt.Errorf("agent: V4.1 lane neither committed nor failed; no replay permitted")
		}
	}
	if row.Err != nil {
		ln.err, ln.done = row.Err, true
	}
}
