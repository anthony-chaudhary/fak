package agent

import (
	"context"
	"errors"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func v41DriverLanes(n, maxNew int) ([]*decodeLane, []*batchLaneSink) {
	lanes, sinks := make([]*decodeLane, n), make([]*batchLaneSink, n)
	for i := range lanes {
		sinks[i] = &batchLaneSink{}
		lanes[i] = &decodeLane{
			s: &model.Session{}, logits: []float32{0, 10, 0}, counts: make([]int32, 3),
			rng: rand.New(rand.NewSource(13)), emit: sinks[i].emit, maxNew: maxNew,
			forwarded: make([]int, 0, maxNew),
		}
	}
	return lanes, sinks
}

func v41DriverRefusal(input []model.V41BatchLane) model.V41BatchResult {
	out := model.V41BatchResult{Lanes: make([]model.V41BatchLaneResult, len(input))}
	for i, lane := range input {
		out.Lanes[i].ID = lane.ID
	}
	return out
}

// These deterministic scripts prove driver state transitions, not arithmetic or
// serving reachability. Actual model/shared-head CPU evidence belongs to Packet B.
// fak-test:runtime fast est=20ms lane=default
func TestInKernelV41DriverStateTransitions(t *testing.T) {
	t.Run("busy retains tokens and shuffled IDs scatter correctly", func(t *testing.T) {
		lanes, sinks := v41DriverLanes(2, 2)
		var first []model.V41BatchLane
		calls, waits, serial := 0, 0, 0
		clock := time.Unix(1, 0)
		now := func() time.Time { clock = clock.Add(time.Nanosecond); return clock }
		for _, lane := range lanes {
			lane.measurement = &nativeInferenceMeasurement{inferenceDisabled: true, traceNow: now}
			lane.measurement.startDecodeTrace()
		}
		ops := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				if calls == 1 {
					first = append([]model.V41BatchLane(nil), input...)
					return v41DriverRefusal(input), model.ErrV41BatchBusy
				}
				if calls != 2 || !reflect.DeepEqual(first, input) {
					t.Fatalf("retry changed token, identity, context or membership: %+v", input)
				}
				return model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: 2, Lanes: []model.V41BatchLaneResult{
					{ID: input[1].ID, Committed: true, Logits: []float32{0, 0, 10}},
					{ID: input[0].ID, Committed: true, Logits: []float32{10, 0, 0}},
				}}, nil
			},
			serial: func(*model.Session, int) ([]float32, error) {
				serial++
				return nil, errors.New("unexpected scalar replay")
			},
			wait: func(context.Context) error {
				waits++
				for i, lane := range lanes {
					if lane.gen != 1 || lane.counts[1] != 1 || !reflect.DeepEqual(sinks[i].toks, []int{1}) || len(lane.forwarded) != 0 {
						t.Fatal("busy path sampled, emitted or committed twice")
					}
				}
				return nil
			},
		}
		got, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, ops)
		if err != nil || calls != 2 || waits != 1 || serial != 0 || got.sharedPanels != 1 || got.sharedRows != 2 || !reflect.DeepEqual(sinks[0].toks, []int{1, 0}) || !reflect.DeepEqual(sinks[1].toks, []int{1, 2}) {
			t.Fatalf("busy/scatter receipt=%+v err=%v calls/waits/serial=%d/%d/%d sinks=%v/%v", got, err, calls, waits, serial, sinks[0].toks, sinks[1].toks)
		}
		for _, lane := range lanes {
			if lane.err != nil || !reflect.DeepEqual(lane.forwarded, []int{1}) || len(lane.measurement.traceEvents) != 2 {
				t.Fatal("driver lost per-lane commit or trace")
			}
			forward := lane.measurement.traceEvents[0].Forward
			if forward == nil || forward.Kind != NativeForwardV41BatchStep || forward.ActiveLanes != 2 || forward.DurationNS != 1 || lane.measurement.traceEvents[1].Forward != nil {
				t.Fatalf("trace mislabeled execution or included busy wait: %+v", forward)
			}
		}
	})

	t.Run("healthy peer survives mixed cancellation and partial commit error", func(t *testing.T) {
		lanes, sinks := v41DriverLanes(4, 3)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		lanes[1].ctx = ctx
		fault := errors.New("selected lane failure")
		calls, serial := 0, 0
		ops := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				if calls == 2 {
					if len(input) != 1 || input[0].ID != 1 || input[0].Token != 2 {
						t.Fatalf("failed peer replayed or healthy token lost: %+v", input)
					}
					return v41DriverRefusal(input), model.ErrV41BatchUnavailable
				}
				if calls != 1 || len(input) != 4 {
					t.Fatal("unexpected batch attempt")
				}
				cancel()
				return model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: 4, Lanes: []model.V41BatchLaneResult{
					{ID: 4, Committed: true, Err: fault},
					{ID: 2, Err: context.Canceled},
					{ID: 1, Committed: true, Logits: []float32{0, 0, 10}},
					{ID: 3, Err: fault},
				}}, fault
			},
			serial: func(s *model.Session, token int) ([]float32, error) {
				serial++
				if s != lanes[0].s || token != 2 {
					t.Fatal("selected or committed failed row replayed")
				}
				return []float32{10, 0, 0}, nil
			},
			wait: func(context.Context) error { t.Fatal("unexpected busy wait"); return nil },
		}
		got, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, ops)
		if err != fault || got.sharedPanels != 1 || got.sharedRows != 4 || calls != 2 || serial != 1 || lanes[0].err != nil || !reflect.DeepEqual(sinks[0].toks, []int{1, 2, 0}) || !reflect.DeepEqual(lanes[0].forwarded, []int{1, 2}) {
			t.Fatalf("healthy peer lost: receipt=%+v err=%v emitted=%v forwarded=%v", got, err, sinks[0].toks, lanes[0].forwarded)
		}
		if !errors.Is(lanes[1].err, context.Canceled) || lanes[2].err != fault || lanes[3].err != fault || !reflect.DeepEqual(lanes[3].forwarded, []int{1}) {
			t.Fatal("cancel/fault/committed-error outcomes were not independently retained")
		}
		for i := 1; i < len(lanes); i++ {
			if !lanes[i].done || !reflect.DeepEqual(sinks[i].toks, []int{1}) {
				t.Fatal("failed lane emitted another token")
			}
		}
	})

	t.Run("cancellation while busy returns without resampling", func(t *testing.T) {
		lanes, sinks := v41DriverLanes(2, 3)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		for _, lane := range lanes {
			lane.ctx = ctx
		}
		calls := 0
		ops := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				return v41DriverRefusal(input), model.ErrV41BatchBusy
			},
			serial: func(*model.Session, int) ([]float32, error) { t.Fatal("busy scalar fallback"); return nil, nil },
			wait:   func(context.Context) error { cancel(); return nil },
		}
		got, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, ops)
		if err != nil || calls != 1 || got.sharedPanels != 0 {
			t.Fatalf("busy cancellation result=%+v err=%v calls=%d", got, err, calls)
		}
		for i, lane := range lanes {
			if !errors.Is(lane.err, context.Canceled) || lane.gen != 1 || len(lane.forwarded) != 0 || !reflect.DeepEqual(sinks[i].toks, []int{1}) {
				t.Fatal("cancellation changed emitted or forwarded state")
			}
		}
	})

	t.Run("all canceled after projection retains completed panel receipt", func(t *testing.T) {
		lanes, _ := v41DriverLanes(2, 3)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		for _, lane := range lanes {
			lane.ctx = ctx
		}
		ops := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				cancel()
				out := v41DriverRefusal(input)
				out.Selected, out.SharedHeadPanels, out.SharedHeadRows = true, 1, 2
				for i := range out.Lanes {
					out.Lanes[i].Err = context.Canceled
				}
				return out, nil
			},
			serial: func(*model.Session, int) ([]float32, error) { t.Fatal("canceled projection replayed"); return nil, nil },
			wait:   func(context.Context) error { t.Fatal("canceled projection retried"); return nil },
		}
		got, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, ops)
		if err != nil || got.sharedPanels != 1 || got.sharedRows != 2 {
			t.Fatalf("zero commits erased completed projection: %+v %v", got, err)
		}
		for _, lane := range lanes {
			if !errors.Is(lane.err, context.Canceled) || len(lane.forwarded) != 0 || lane.gen != 1 {
				t.Fatal("canceled projection changed lane commit accounting")
			}
		}
	})

	for _, fault := range []string{"unselected commit", "duplicate ID", "selected unavailable"} {
		t.Run(fault+" never falls back", func(t *testing.T) {
			lanes, _ := v41DriverLanes(2, 3)
			calls := 0
			ops := inKernelV41DecodeOps{
				batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
					calls++
					out := v41DriverRefusal(input)
					switch fault {
					case "unselected commit":
						out.Lanes[0].Committed = true
					case "duplicate ID":
						out.Lanes[1].ID = out.Lanes[0].ID
					case "selected unavailable":
						out.Selected = true
					}
					return out, model.ErrV41BatchUnavailable
				},
				serial: func(*model.Session, int) ([]float32, error) {
					t.Fatal("effected result authorized replay")
					return nil, nil
				},
				wait: func(context.Context) error { t.Fatal("effected result retried"); return nil },
			}
			_, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, ops)
			if err == nil || calls != 1 {
				t.Fatalf("bad result was not terminal: %v calls=%d", err, calls)
			}
			for _, lane := range lanes {
				if !lane.done || lane.err == nil {
					t.Fatal("invalid result left a lane runnable")
				}
			}
		})
	}

	t.Run("fatal panic retains original identity", func(t *testing.T) {
		lanes, sinks := v41DriverLanes(2, 3)
		primary := &struct{ marker string }{"original backend panic"}
		calls := 0
		ops := inKernelV41DecodeOps{
			batch:  func([]model.V41BatchLane) (model.V41BatchResult, error) { calls++; panic(primary) },
			serial: func(*model.Session, int) ([]float32, error) { t.Fatal("fatal panic replayed"); return nil, nil },
			wait:   func(context.Context) error { t.Fatal("fatal panic retried"); return nil },
		}
		var got any
		func() {
			defer func() { got = recover() }()
			_, _ = inKernelDecodeV41LanesWithOps(context.Background(), lanes, ops)
		}()
		if got != primary || calls != 1 || !reflect.DeepEqual(sinks[0].toks, []int{1}) || !reflect.DeepEqual(sinks[1].toks, []int{1}) {
			t.Fatalf("panic identity or one-emit boundary changed: %v calls=%d", got, calls)
		}
	})
}

// The real adapter calls Model.StepV41Batch, which refuses this dense fixture,
// then ordinary public Session.Step. This is negative-capability wiring proof;
// it does not establish a successful Flash shared-head serving route.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelV41DriverUnavailableUsesPublicStep(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	lanes, refs := make([]*decodeLane, 2), make([]*decodeLane, 2)
	sinks, wants := make([]*batchLaneSink, 2), make([]*batchLaneSink, 2)
	for i := range lanes {
		prompt := synthIDs(m.Cfg.VocabSize, 3+i, uint64(30+i))
		lanes[i], sinks[i] = newDecodeLaneOver(m, prompt, 3+i, 42)
		refs[i], wants[i] = newDecodeLaneOver(m, prompt, 3+i, 42)
		lanes[i].forwarded = make([]int, 0)
		t.Cleanup(lanes[i].s.Close)
		t.Cleanup(refs[i].s.Close)
		inKernelDecodeSerial(context.Background(), refs[i])
	}
	got, err := inKernelDecodeV41Lanes(context.Background(), lanes, m)
	if err != nil || got.sharedPanels != 0 || got.sharedRows != 0 {
		t.Fatalf("unsupported model falsely claimed shared work: %+v %v", got, err)
	}
	for i, lane := range lanes {
		if lane.err != nil || !reflect.DeepEqual(sinks[i].toks, wants[i].toks) || lane.s.Cache.Len() != refs[i].s.Cache.Len() || len(lane.forwarded) != lane.gen-1 {
			t.Fatalf("public scalar fallback drifted lane %d", i)
		}
	}
}
