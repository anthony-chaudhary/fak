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

// v41OracleLogits derives one request's scripted 16-wide logit row from its seed.
// The spread keeps every keyed draw genuinely stochastic while remaining fully
// deterministic, so cross-lane contamination moves a stream.
func v41OracleLogits(seed int) []float32 {
	out := make([]float32, 16)
	for j := range out {
		out[j] = float32((seed*7+j*11)%19) * 0.1
	}
	return out
}

// v41OracleScriptedLane builds a keyed decode lane over an empty Session for
// scripted V4.1 driver results: the script owns every forward, the lane pins the
// per-request sampling identity (key, base, temp > 0) only.
func v41OracleScriptedLane(key uint64, base, maxNew int, logits []float32) (*decodeLane, *batchLaneSink) {
	sink := &batchLaneSink{}
	ln := &decodeLane{
		s:         &model.Session{},
		logits:    append([]float32(nil), logits...),
		counts:    make([]int32, len(logits)),
		rng:       rand.New(rand.NewSource(13)),
		emit:      sink.emit,
		stops:     map[int]bool{},
		maxNew:    maxNew,
		temp:      1,
		sampleKey: key, sampleKeyed: true, sampleBase: base,
		forwarded: make([]int, 0, maxNew),
	}
	return ln, sink
}

// TestV41BatchCompositionInvariantAcrossNeighbors is the V4.1-specific
// singleton-versus-batch composition oracle adapted from DeepInfer's
// test_batch_invariant.py (oss-port-deepinfer-batch-invariance-oracle, leaf 1):
// keyed-sampling lanes driven through the live V4.1 driver seam must emit token
// streams identical to their singleton runs regardless of neighbor permutation or
// early retirement, preserving the transactional no-replay rule. The dense
// synthetic fixture refuses StepV41Batch, so the model-driven arm rides the
// driver's public scalar fallback; it pins driver composition, not Flash
// shared-head serving.
//
// fak-test:runtime fast est=1s lane=default
func TestV41BatchCompositionInvariantAcrossNeighbors(t *testing.T) {
	t.Run("keyed streams match singletons across composition", func(t *testing.T) {
		cfg := tinyCfg()
		m := model.NewSynthetic(cfg)
		sharedPrompt := synthIDs(cfg.VocabSize, 13, 9101)
		type request struct {
			prompt []int
			key    uint64
			maxNew int
		}
		requests := []request{
			{prompt: sharedPrompt, key: 101, maxNew: 12},
			{prompt: sharedPrompt, key: ^uint64(0), maxNew: 12},
			{prompt: synthIDs(cfg.VocabSize, 9, 9102), key: 303, maxNew: 8},
			{prompt: synthIDs(cfg.VocabSize, 17, 9103), key: 0, maxNew: 1},
		}
		run := func(order []int) map[int][]int {
			lanes := make([]*decodeLane, len(order))
			sinks := make([]*batchLaneSink, len(order))
			for batchIndex, requestIndex := range order {
				r := requests[requestIndex]
				lane, sink := newDecodeLaneOver(m, r.prompt, r.maxNew, 777)
				lane.temp, lane.topP, lane.topK = 5, 0.95, 16
				lane.sampleKey, lane.sampleKeyed, lane.sampleBase = r.key, true, len(r.prompt)
				lanes[batchIndex], sinks[batchIndex] = lane, sink
				t.Cleanup(lane.s.Close)
			}
			receipt, err := inKernelDecodeV41Lanes(context.Background(), lanes, m)
			if err != nil {
				t.Fatalf("composition %v failed: %v", order, err)
			}
			if receipt.sharedPanels != 0 || receipt.sharedRows != 0 {
				t.Fatalf("dense fixture claimed shared-head work: %+v", receipt)
			}
			got := make(map[int][]int, len(order))
			for batchIndex, requestIndex := range order {
				if lanes[batchIndex].err != nil {
					t.Fatalf("request %d decode failed: %v", requestIndex, lanes[batchIndex].err)
				}
				if len(sinks[batchIndex].toks) != requests[requestIndex].maxNew {
					t.Fatalf("request %d produced %d tokens, want %d", requestIndex, len(sinks[batchIndex].toks), requests[requestIndex].maxNew)
				}
				got[requestIndex] = sinks[batchIndex].toks
			}
			return got
		}
		singletons := make(map[int][]int, len(requests))
		for i := range requests {
			singletons[i] = run([]int{i})[i]
			if replay := run([]int{i})[i]; !eqInts(replay, singletons[i]) {
				t.Fatalf("request %d keyed replay diverged: got %v want %v", i, replay, singletons[i])
			}
		}
		if eqInts(singletons[0], singletons[1]) {
			t.Fatalf("distinct keys on one prompt did not activate stochastic sampling: %v", singletons[0])
		}
		assertMatchesSingleton := func(name string, got map[int][]int, ids ...int) {
			t.Helper()
			for _, id := range ids {
				if !eqInts(got[id], singletons[id]) {
					t.Fatalf("%s changed request %d:\n got =%v\n want=%v", name, id, got[id], singletons[id])
				}
			}
		}
		assertMatchesSingleton("original order", run([]int{0, 1, 2}), 0, 1, 2)
		assertMatchesSingleton("permuted order", run([]int{2, 0, 1}), 0, 1, 2)
		assertMatchesSingleton("early-retiring neighbor", run([]int{3, 0, 1, 2}), 0, 1, 2, 3)
	})

	t.Run("failing row isolates its lane", func(t *testing.T) {
		fault := errors.New("selected V4.1 lane failure")
		keys := []uint64{11, 22, 33, 44}
		bases := []int{40, 41, 42, 43}
		seeds := []int{1, 2, 3, 4}
		const maxNew = 4
		build := func() ([]*decodeLane, []*batchLaneSink, map[*model.Session]int) {
			lanes := make([]*decodeLane, 4)
			sinks := make([]*batchLaneSink, 4)
			bySession := make(map[*model.Session]int, 4)
			for i := range lanes {
				lanes[i], sinks[i] = v41OracleScriptedLane(keys[i], bases[i], maxNew, v41OracleLogits(seeds[i]))
				bySession[lanes[i].s] = i
			}
			return lanes, sinks, bySession
		}
		operations := func(bySession map[*model.Session]int) inKernelV41DecodeOps {
			rounds := make(map[*model.Session]int, 4)
			return inKernelV41DecodeOps{
				batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
					out := model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: len(input), Lanes: make([]model.V41BatchLaneResult, len(input))}
					for i, lane := range input {
						index := bySession[lane.Session]
						rounds[lane.Session]++
						if index == 1 && rounds[lane.Session] > 1 {
							out.Lanes[i] = model.V41BatchLaneResult{ID: lane.ID, Err: fault}
							continue
						}
						out.Lanes[i] = model.V41BatchLaneResult{ID: lane.ID, Committed: true, Logits: v41OracleLogits(seeds[index])}
					}
					return out, nil
				},
				serial: func(*model.Session, int) ([]float32, error) {
					return nil, errors.New("unexpected scalar replay")
				},
				wait: func(context.Context) error { return errors.New("unexpected busy wait") },
			}
		}
		// Singleton references: the same script drives each lane alone, so a healthy
		// peer's stream is pinned independently of batch composition.
		refs := make([][]int, 4)
		for i := 0; i < 4; i++ {
			lanes, sinks, bySession := build()
			_, err := inKernelDecodeV41LanesWithOps(context.Background(), []*decodeLane{lanes[i]}, operations(bySession))
			if i == 1 {
				if err != nil || lanes[i].err != fault {
					t.Fatalf("singleton failing lane terminal=%v laneErr=%v", err, lanes[i].err)
				}
			} else if err != nil || lanes[i].err != nil {
				t.Fatalf("singleton %d failed: terminal=%v laneErr=%v", i, err, lanes[i].err)
			}
			refs[i] = sinks[i].toks
		}
		// The fault row lands at round two's forward, so the failing lane emits two
		// tokens but commits only the first.
		if len(refs[1]) != 2 {
			t.Fatalf("scripted failure did not fire at round two: %v", refs[1])
		}
		lanes, sinks, bySession := build()
		terminal, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, operations(bySession))
		if err != nil {
			t.Fatalf("failing row broadcast over the terminal error: %v", err)
		}
		// Three forwards commit (rounds one to three); the fourth emission retires
		// the healthy lanes without a further forward, so receipts stop at 3x(1,3+4).
		if terminal.sharedPanels != 3 || terminal.sharedRows != 11 {
			t.Fatalf("healthy peers lost committed projection: %+v", terminal)
		}
		for i := 0; i < 4; i++ {
			if !eqInts(sinks[i].toks, refs[i]) || !lanes[i].done {
				t.Fatalf("lane %d drifted under the failing peer: emitted=%v want %v done=%t", i, sinks[i].toks, refs[i], lanes[i].done)
			}
		}
		if lanes[1].err != fault || lanes[1].gen != 2 || !eqInts(lanes[1].forwarded, refs[1][:1]) {
			t.Fatalf("failing lane outcome=%v gen=%d forwarded=%v", lanes[1].err, lanes[1].gen, lanes[1].forwarded)
		}
		for i := 0; i < 4; i++ {
			if i == 1 {
				continue
			}
			if lanes[i].err != nil || lanes[i].gen != maxNew || !eqInts(lanes[i].forwarded, refs[i][:maxNew-1]) {
				t.Fatalf("healthy lane %d lost: err=%v gen=%d forwarded=%v", i, lanes[i].err, lanes[i].gen, lanes[i].forwarded)
			}
		}
	})

	t.Run("busy refusal retries with chosen token and keyed state", func(t *testing.T) {
		keys := []uint64{101, 303}
		bases := []int{40, 41}
		seeds := []int{1, 2}
		const maxNew = 3
		buildPair := func() ([]*decodeLane, []*batchLaneSink, map[*model.Session]int) {
			lanes := make([]*decodeLane, 2)
			sinks := make([]*batchLaneSink, 2)
			bySession := make(map[*model.Session]int, 2)
			for i := range lanes {
				lanes[i], sinks[i] = v41OracleScriptedLane(keys[i], bases[i], maxNew, v41OracleLogits(seeds[i]))
				bySession[lanes[i].s] = i
			}
			return lanes, sinks, bySession
		}
		commitRows := func(input []model.V41BatchLane, bySession map[*model.Session]int) model.V41BatchResult {
			out := model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: len(input), Lanes: make([]model.V41BatchLaneResult, len(input))}
			for i, lane := range input {
				out.Lanes[i] = model.V41BatchLaneResult{ID: lane.ID, Committed: true, Logits: v41OracleLogits(seeds[bySession[lane.Session]])}
			}
			return out
		}
		controlLanes, controlSinks, controlBySession := buildPair()
		if _, err := inKernelDecodeV41LanesWithOps(context.Background(), controlLanes, inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				return commitRows(input, controlBySession), nil
			},
			serial: func(*model.Session, int) ([]float32, error) { return nil, errors.New("unexpected scalar replay") },
			wait:   func(context.Context) error { return errors.New("unexpected busy wait") },
		}); err != nil {
			t.Fatalf("control run failed: %v", err)
		}
		refs := [][]int{controlSinks[0].toks, controlSinks[1].toks}
		for i := range refs {
			if len(refs[i]) != maxNew {
				t.Fatalf("control lane %d produced %d tokens, want %d", i, len(refs[i]), maxNew)
			}
		}
		lanes, sinks, bySession := buildPair()
		var first []model.V41BatchLane
		calls, waits := 0, 0
		got, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				if calls == 1 {
					first = append([]model.V41BatchLane(nil), input...)
					return v41DriverRefusal(input), model.ErrV41BatchBusy
				}
				if calls == 2 && !reflect.DeepEqual(first, input) {
					t.Fatal("busy retry changed chosen token, identity, context or membership")
				}
				return commitRows(input, bySession), nil
			},
			serial: func(*model.Session, int) ([]float32, error) {
				return nil, errors.New("unexpected scalar replay")
			},
			wait: func(context.Context) error {
				waits++
				for i, lane := range lanes {
					want := refs[i][0]
					if lane.gen != 1 || lane.counts[want] != 1 || !eqInts(sinks[i].toks, []int{want}) || len(lane.forwarded) != 0 {
						t.Fatalf("busy wait resampled or re-emitted lane %d: gen=%d counts=%v emitted=%v", i, lane.gen, lane.counts, sinks[i].toks)
					}
				}
				return nil
			},
		})
		if err != nil || calls != 3 || waits != 1 || got.sharedPanels != 2 || got.sharedRows != 4 {
			t.Fatalf("busy retry receipt=%+v err=%v calls=%d waits=%d", got, err, calls, waits)
		}
		for i := range lanes {
			if !eqInts(sinks[i].toks, refs[i]) || !eqInts(lanes[i].forwarded, refs[i][:maxNew-1]) {
				t.Fatalf("busy refusal perturbed lane %d: emitted=%v forwarded=%v want %v", i, sinks[i].toks, lanes[i].forwarded, refs[i])
			}
		}
	})

	t.Run("unavailable refusal falls back serially without resampling", func(t *testing.T) {
		keys := []uint64{101, 303}
		bases := []int{40, 41}
		seeds := []int{1, 2}
		const maxNew = 3
		buildPair := func() ([]*decodeLane, []*batchLaneSink, map[*model.Session]int) {
			lanes := make([]*decodeLane, 2)
			sinks := make([]*batchLaneSink, 2)
			bySession := make(map[*model.Session]int, 2)
			for i := range lanes {
				lanes[i], sinks[i] = v41OracleScriptedLane(keys[i], bases[i], maxNew, v41OracleLogits(seeds[i]))
				bySession[lanes[i].s] = i
			}
			return lanes, sinks, bySession
		}
		commitRows := func(input []model.V41BatchLane, bySession map[*model.Session]int) model.V41BatchResult {
			out := model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: len(input), Lanes: make([]model.V41BatchLaneResult, len(input))}
			for i, lane := range input {
				out.Lanes[i] = model.V41BatchLaneResult{ID: lane.ID, Committed: true, Logits: v41OracleLogits(seeds[bySession[lane.Session]])}
			}
			return out
		}
		controlLanes, controlSinks, controlBySession := buildPair()
		if _, err := inKernelDecodeV41LanesWithOps(context.Background(), controlLanes, inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				return commitRows(input, controlBySession), nil
			},
			serial: func(*model.Session, int) ([]float32, error) { return nil, errors.New("unexpected scalar replay") },
			wait:   func(context.Context) error { return errors.New("unexpected busy wait") },
		}); err != nil {
			t.Fatalf("control run failed: %v", err)
		}
		refs := [][]int{controlSinks[0].toks, controlSinks[1].toks}
		lanes, sinks, bySession := buildPair()
		calls, serialCalls := 0, 0
		var serialTokens []int
		got, err := inKernelDecodeV41LanesWithOps(context.Background(), lanes, inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				if calls == 1 {
					return v41DriverRefusal(input), model.ErrV41BatchUnavailable
				}
				return commitRows(input, bySession), nil
			},
			serial: func(s *model.Session, token int) ([]float32, error) {
				serialCalls++
				serialTokens = append(serialTokens, token)
				return v41OracleLogits(seeds[bySession[s]]), nil
			},
			wait: func(context.Context) error { return errors.New("unexpected busy wait") },
		})
		if err != nil || calls != 2 || serialCalls != 2 || !eqInts(serialTokens, []int{refs[0][0], refs[1][0]}) {
			t.Fatalf("serial fallback resampled: calls=%d serial=%d tokens=%v want %v", calls, serialCalls, serialTokens, []int{refs[0][0], refs[1][0]})
		}
		if got.sharedPanels != 1 || got.sharedRows != 2 {
			t.Fatalf("serial fallback erased completed projections: %+v", got)
		}
		for i := range lanes {
			if !eqInts(sinks[i].toks, refs[i]) || !eqInts(lanes[i].forwarded, refs[i][:maxNew-1]) {
				t.Fatalf("unavailable refusal perturbed lane %d: emitted=%v forwarded=%v want %v", i, sinks[i].toks, lanes[i].forwarded, refs[i])
			}
		}
	})
}
