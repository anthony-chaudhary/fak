package agent

import (
	"context"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type iterationLaneResult struct {
	Trace             []NativeDecodeTraceEvent
	Tokens, Forwarded []int
	Gen               int
	Stopped, Done     bool
	Error             string
	Logits            []float32
	GreedyReady       bool
	GreedyNext, Vocab int
}

// fak-test:runtime fast est=1s
func TestInKernelDecodeIterationMatchesOriginal(t *testing.T) {
	for _, count := range []int{0, 1, 4} {
		for _, mode := range []string{"drain", "trace", "clock_error"} {
			run := func(original bool) ([]iterationLaneResult, int, int64) {
				cfg := tinyCfg()
				cfg.EOSTokenID = -1
				m := model.NewSynthetic(cfg)
				lanes := make([]*decodeLane, count)
				tokens := make([][]int, count)
				for i := range lanes {
					s := m.NewSession()
					defer s.Close()
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					logits := s.Prefill([]int{1, 2, 3 + i})
					laneIndex := i
					lanes[i] = &decodeLane{s: s, logits: logits, rng: rand.New(rand.NewSource(int64(17 + i))), temp: 0.7, topP: 0.9, maxNew: 5 + i, stops: map[int]bool{}, ctx: ctx, forwarded: make([]int, 0), emit: func(id int) bool {
						tokens[laneIndex] = append(tokens[laneIndex], id)
						if laneIndex == 1 && len(tokens[laneIndex]) == 2 {
							cancel()
						}
						return laneIndex == 2 && len(tokens[laneIndex]) == 3
					}}
					if mode == "trace" || mode == "clock_error" {
						calls := 0
						measurement := &nativeInferenceMeasurement{inferenceDisabled: true, traceNow: func() time.Time {
							calls++
							offset := calls
							if mode == "clock_error" && calls == 4 {
								offset = 2
							}
							return time.Unix(100, int64(offset))
						}}
						measurement.startDecodeTrace()
						lanes[i].measurement = measurement
					}
					if i == 3 {
						for id := 0; id < cfg.VocabSize; id++ {
							lanes[i].stops[id] = true
						}
					}
				}
				var panels int
				var macs int64
				if original {
					panels, macs = originalIterationDrain(context.Background(), lanes, m, false)
				} else {
					panels, macs = inKernelDecodeLanesBatched(context.Background(), lanes, m, false)
				}
				out := make([]iterationLaneResult, count)
				for i, ln := range lanes {
					out[i] = iterationLaneResult{Tokens: tokens[i], Forwarded: ln.forwarded, Gen: ln.gen, Stopped: ln.stopped, Done: ln.done, Logits: append([]float32(nil), ln.logits...), GreedyReady: ln.greedyReady, GreedyNext: ln.greedyNext, Vocab: ln.vocab}
					if ln.measurement != nil {
						out[i].Trace = ln.measurement.traceEvents
					}
					if ln.err != nil {
						out[i].Error = ln.err.Error()
					}
				}
				return out, panels, macs
			}
			want, wp, wm := run(true)
			got, gp, gm := run(false)
			if mode == "clock_error" && count > 0 && want[0].Error != "native decode forward clock moved backwards at token 1" {
				t.Fatalf("clock error fixture did not reach forward observer: %+v", want[0])
			}
			if !reflect.DeepEqual(got, want) || gp != wp || gm != wm {
				t.Fatalf("lanes=%d mode=%s: got %+v panels=%d macs=%d, want %+v panels=%d macs=%d", count, mode, got, gp, gm, want, wp, wm)
			}
		}
	}
}

// Literal pre-extraction driver from public 64ee7b08, retained as an independent
// oracle so the wrapper cannot validate itself through the new step method.
func originalIterationDrain(ctx context.Context, lanes []*decodeLane, m *model.Model, quant bool) (sharedPanels int, sharedMACs int64) {
	if len(lanes) == 0 {
		return
	}
	seqs := make([]*model.Session, len(lanes))
	for i, ln := range lanes {
		seqs[i] = ln.s
	}
	bs := &model.BatchSession{M: m, Seqs: seqs, Quant: quant}
	serialSteps := 0
	bs.ObserveSerialStep = func(reason string, d time.Duration) {
		serialSteps++
		enginestep.Default.ObserveSerialFallback(reason, d)
	}
	ids := make([]int, len(lanes))
	active := make([]bool, len(lanes))
	greedy := make([]bool, len(lanes))
	for {
		anyActive := false
		for i, ln := range lanes {
			active[i] = false
			if ln.done || ln.gen >= ln.maxNew {
				continue // finished lane: dropped from the active set, never re-stepped.
			}
			laneCtx := ctx
			if ln.ctx != nil {
				laneCtx = ln.ctx
			}
			next, advance := ln.decodeOne(laneCtx)
			if !advance {
				continue
			}
			ids[i] = next
			active[i] = true
			anyActive = true
		}
		if !anyActive {
			return
		}
		activeLanes := 0
		var forwardNow func() time.Time
		for i, isActive := range active {
			if !isActive {
				continue
			}
			activeLanes++
			if forwardNow == nil && lanes[i].measurement != nil {
				forwardNow = lanes[i].measurement.traceNow
			}
		}
		var forwardStarted time.Time
		if forwardNow != nil {
			forwardStarted = forwardNow()
		}
		stepStart := time.Now()
		serialSteps = 0
		var next []int
		out, panels, macs, probed := runQwenSharedReceiptProbe(bs, ids, active)
		if !probed {
			anyGreedy := false
			for i, ln := range lanes {
				greedy[i] = active[i] && ln.deviceGreedy()
				if greedy[i] {
					ln.retainGreedyLogits()
				}
				anyGreedy = anyGreedy || greedy[i]
			}
			if anyGreedy {
				out, next = bs.StepBatchActiveGreedy(ids, active, greedy)
			} else {
				out = bs.StepBatchActive(ids, active)
			}
			panels = bs.LastStepSharedPanels()
			macs = bs.LastStepMACs()
		}
		if serialSteps == 0 {
			enginestep.Default.ObserveDecodeStep(enginestep.PathBatched, activeLanes, time.Since(stepStart))
		}
		if forwardNow != nil {
			// This trace preserves outer call provenance and duration even when
			// the execution recorder above reports individual scalar fallbacks.
			duration := forwardNow().Sub(forwardStarted)
			for i, isActive := range active {
				if !isActive {
					continue
				}
				if err := lanes[i].measurement.recordForwardTiming(lanes[i].gen, NativeForwardStepBatchActive, activeLanes, duration); err != nil {
					lanes[i].err, lanes[i].done = err, true
				}
			}
		}
		sharedPanels += panels
		sharedMACs += macs
		for i, ln := range lanes {
			if active[i] {
				if ln.logits != nil {
					ln.vocab = len(ln.logits)
				}
				ln.logits = out[i]
				ln.greedyReady = next != nil && greedy[i] && out[i] == nil
				if ln.greedyReady {
					ln.greedyNext = next[i]
				}
				if lanes[i].forwarded != nil {
					lanes[i].forwarded = append(lanes[i].forwarded, ids[i])
				}
			}
		}
	}
}
