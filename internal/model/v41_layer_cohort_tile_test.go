package model

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The real StepV41Batch consumer uses the existing CPU arithmetic recorders.
// Panel methods reproduce scalar quantized math; this is an orchestration and
// lifetime witness, not native shader parity or a throughput measurement.
type v41LayerCohortTileBackend struct {
	*v41LayerCohortBackend
	tile                   *v41ExpertDownTileWitness
	tileFences             int
	packFault              string
	captureSums            bool
	sumInputs              [][]float32
	declineQ3At, q3Checks  int
	declinedBeforeDispatch bool
}

func (b *v41LayerCohortTileBackend) AddInPlace(dst, x compute.Tensor) {
	if b.captureSums && len(b.pending) > 0 {
		b.sumInputs = append(b.sumInputs, append([]float32(nil), b.Backend.Read(x)...))
	}
	b.Backend.AddInPlace(dst, x)
}

func (b *v41LayerCohortTileBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	if dt == compute.Q3_K && b.declineQ3At > 0 {
		b.q3Checks++
		if b.q3Checks == b.declineQ3At {
			b.declinedBeforeDispatch = len(b.tile.tileRows) == 0 && b.tile.activationCalls == 0 && b.publishes == 0
			return false
		}
	}
	return b.v41LayerCohortBackend.SupportsDeviceWeightDtype(dt)
}

func (b *v41LayerCohortTileBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if b.packFault == "gate/up" && site == "V4.1 same-expert token tile" {
		return compute.Tensor{}
	}
	return b.v41LayerCohortBackend.UploadClass(x, dt, class, site)
}

func (b *v41LayerCohortTileBackend) SupportsV41Q2KTokenPanel(w compute.Tensor, rows int) bool {
	return b.tile.SupportsV41Q2KTokenPanel(w, rows)
}
func (b *v41LayerCohortTileBackend) SupportsV41Q3KTokenPanel(w compute.Tensor, rows int) bool {
	return b.tile.SupportsV41Q3KTokenPanel(w, rows)
}
func (b *v41LayerCohortTileBackend) checkTileOwner() {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("token panel escaped its shared execution owner")
	}
}
func (b *v41LayerCohortTileBackend) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.checkTileOwner()
	return b.tile.V41Q2KTokenPanel(w, x, rows)
}
func (b *v41LayerCohortTileBackend) V41Q3KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.checkTileOwner()
	return b.tile.V41Q3KTokenPanel(w, x, rows)
}
func (b *v41LayerCohortTileBackend) V41RowLogAvailable() bool {
	return b.tile.tiles || b.tile.downTiles
}
func (b *v41LayerCohortTileBackend) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	return b.tile.V41RowLogAlloc(rows, width)
}
func (b *v41LayerCohortTileBackend) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	return b.tile.V41RowLogCopy(dst, dstRow, src, srcRow, rows)
}
func (b *v41LayerCohortTileBackend) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	b.checkTileOwner()
	if b.packFault == "down" && len(spans) > 1 {
		return compute.Tensor{}, errors.New("down activation packing failed")
	}
	return b.tile.V41RowLogGather(spans, width)
}
func (b *v41LayerCohortTileBackend) V41RoutedSwiGLUDeferred(g, u compute.Tensor, limit, weight float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	return b.tile.V41RoutedSwiGLUDeferred(g, u, limit, weight)
}
func (b *v41LayerCohortTileBackend) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16Publish(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16Publish(x, site)
}
func (b *v41LayerCohortTileBackend) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16PublishDeferred(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16PublishDeferred(x, site)
}
func (b *v41LayerCohortTileBackend) Free(x compute.Tensor) { b.tile.Free(x) }
func (b *v41LayerCohortTileBackend) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) > 0 && statuses[0].Record.Backend() == b {
		return b.ordinary.V41ReadDeferred(statuses, rows)
	}
	if len(b.tile.panels) > 0 {
		b.checkTileOwner()
		b.tileFences++
		for _, panel := range b.tile.panels {
			if !b.tile.ownedTiles[panel] {
				return nil, nil, errors.New("cohort tile released before the shared fence")
			}
		}
		b.tile.panels = nil
	}
	return b.v41LayerCohortBackend.V41ReadDeferred(statuses, rows)
}

// fak-test:runtime medium est=20s lane=default
func TestV41LayerCohortTokenTiles(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "1")
	m := v41PinnedSlateFixture(t, false)
	defer m.CloseWeights()
	newFixture := func(t *testing.T, count int) (*v41LayerCohortTileBackend, []V41BatchLane, []*Session) {
		t.Helper()
		scalar := &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}
		b := &v41LayerCohortTileBackend{
			v41LayerCohortBackend: &v41LayerCohortBackend{v41ScalarCohortWitness: scalar},
			tile:                  &v41ExpertDownTileWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: scalar, ownedTiles: map[compute.Buffer]bool{}}},
		}
		b.ordinary = newV41DeferredPublishRecorder()
		b.ordinary.self = b
		lanes, refs := make([]V41BatchLane, count), make([]*Session, count)
		for i := range lanes {
			var err error
			lanes[i].Session, err = m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(lanes[i].Session.Close)
			refs[i], err = m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(refs[i].Close)
			// Keep unrelated attention output on its scalar reference. This
			// recorder implements fresh expert gathers, not resident KV/logs.
			lanes[i].Session.v41State().groupedOutput = nil
			refs[i].v41State().groupedOutput = nil
			prefix := []int{1, 2}
			for j := 0; j < i; j++ {
				prefix = append(prefix, 3+j)
			}
			lanes[i].ID, lanes[i].Token = uint64(91-i*17), 4+i
			lanes[i].Session.Prefill(prefix)
			refs[i].Prefill(prefix)
		}
		return b, lanes, refs
	}
	startTiles := func(b *v41LayerCohortTileBackend) {
		b.resetCohort()
		b.publishes, b.fences, b.tile.activationCalls = 0, 0, 0
		b.tile.tiles, b.tile.downTiles = true, true
	}
	// Route selection alone is controlled: the real router still derives
	// normalized weights from each lane's logits, and every expert/continuation
	// uses real Q2_K/Q3_K arithmetic. Routes contain shared and singleton experts
	// in different pick orders, including groups led by a later lane.
	divergent := [][]int{{5, 1, 4, 0, 3, 2}, {6, 2, 5, 7, 0, 8}, {9, 3, 6, 1, 8, 10}, {11, 4, 7, 2, 9, 0}}
	setRoutes := func(t *testing.T, lanes []V41BatchLane, refs []*Session, routes [][]int) {
		t.Helper()
		for lane, experts := range routes {
			bias := make([]float32, m.Cfg.NumExperts)
			for slot, expert := range experts {
				bias[expert] = float32(1000 * (len(experts) - slot))
			}
			route := func(_ int, logits, _ []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
				picks, err := v41RouteForGeometry(logits, bias, cfg, full)
				if err == nil {
					for slot, pick := range picks {
						if pick.expert != experts[slot] {
							t.Fatal("controlled router failed to select the requested divergent route")
						}
					}
				}
				return picks, err
			}
			lanes[lane].Session.v41State().setRouterTopK(route)
			refs[lane].v41State().setRouterTopK(route)
		}
	}
	for _, count := range []int{2, 3, 4} {
		t.Run("divergent_"+itoa(count)+"_lanes", func(t *testing.T) {
			b, lanes, refs := newFixture(t, count)
			setRoutes(t, lanes, refs, divergent[:count])
			b.captureSums = true
			want := make([][]float32, count)
			for lane, s := range refs {
				want[lane] = append([]float32(nil), s.Step(lanes[lane].Token)...)
			}
			controlInputs := b.sumInputs
			b.sumInputs = nil
			startTiles(b)
			got, err := m.StepV41Batch(lanes)
			if err != nil || !got.Selected {
				t.Fatalf("divergent cohort selected=%v err=%v", got.Selected, err)
			}
			occurrences := map[int]int{}
			for _, route := range divergent[:count] {
				for _, expert := range route {
					occurrences[expert]++
				}
			}
			shared := 0
			for _, rows := range occurrences {
				if rows >= 2 {
					shared++
				}
			}
			if len(b.tile.tileRows) != 2*shared*m.Cfg.NumLayers || len(b.tile.downRows) != shared*m.Cfg.NumLayers || b.tileFences != m.Cfg.NumLayers {
				t.Fatalf("divergent reuse missing: gate/up=%v down=%v fences=%d", b.tile.tileRows, b.tile.downRows, b.tileFences)
			}
			k, layers := m.Cfg.NumExpertsPerTok, m.Cfg.NumLayers
			if len(b.sumInputs) != count*k*layers || len(controlInputs) != len(b.sumInputs) || len(b.expertFences) != layers {
				t.Fatalf("routed rows/fences=%d/%d %v", len(b.sumInputs), len(controlInputs), b.expertFences)
			}
			positive, negative := false, false
			for layer := 0; layer < layers; layer++ {
				if b.expertFences[layer] != 2*count*k || b.outputsPerFence[layer] != count {
					t.Fatal("divergent groups split the shared fence or lost a lane sum")
				}
				for lane := range lanes {
					for slot := 0; slot < k; slot++ {
						row := b.sumInputs[(layer*count+lane)*k+slot]
						if !v41SameBits(row, controlInputs[(lane*layers+layer)*k+slot]) {
							t.Fatalf("lane %d layer %d contribution %d changed scalar accumulation order", lane, layer, slot)
						}
						for _, value := range row {
							positive, negative = positive || value > 0, negative || value < 0
						}
					}
				}
			}
			if !positive || !negative {
				t.Fatal("parity witness lacks nonzero signed routed contributions")
			}
			for lane, result := range got.Lanes {
				if result.ID != lanes[lane].ID || !result.Committed || result.Err != nil || !v41SameBits(result.Logits, want[lane]) {
					t.Fatalf("divergent lane %d differs from its scalar result", lane)
				}
				if next, reference := lanes[lane].Session.Step(6), refs[lane].Step(6); !v41SameBits(next, reference) {
					t.Fatalf("divergent lane %d continuation changed", lane)
				}
			}
			if len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
				t.Fatal("divergent cohort retained status/panel/activation ownership")
			}
		})
	}
	t.Run("divergent_singleton_decline_is_atomic", func(t *testing.T) {
		b, lanes, refs := newFixture(t, 2)
		setRoutes(t, lanes, refs, divergent[:2])
		want := make([][]float32, len(refs))
		for lane, s := range refs {
			want[lane] = append([]float32(nil), s.Step(lanes[lane].Token)...)
		}
		startTiles(b)
		b.declineQ3At = 2 // Down admission of the singleton after the first shared expert.
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || !b.declinedBeforeDispatch || len(b.expertFences) != m.Cfg.NumLayers+1 || b.expertFences[0] != 12 || b.expertFences[1] != 12 {
			t.Fatalf("late preparation decline selected arithmetic: fences=%v err=%v", b.expertFences, err)
		}
		for lane, result := range got.Lanes {
			if !result.Committed || result.Err != nil || !v41SameBits(result.Logits, want[lane]) || lanes[lane].Session.BackendSessionClosed() {
				t.Fatalf("preselection decline damaged lane %d", lane)
			}
		}
		if len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
			t.Fatal("declined singleton retained prepared ownership")
		}
	})
	t.Run("divergent_singleton_verdict_precedes_later_tile_failure", func(t *testing.T) {
		b, lanes, refs := newFixture(t, 2)
		setRoutes(t, lanes, refs, divergent[:2])
		before := make([]*v41ForwardSnapshot, len(lanes))
		for lane := range lanes {
			before[lane] = captureV41ForwardSnapshot(lanes[lane].Session.v41State())
		}
		startTiles(b)
		b.publishFaultAt = map[int]bool{3: true} // First singleton follows the shared expert's two rows.
		b.tile.failTileAt = 3                    // Gate of the next shared expert, after two scalar singletons.
		b.tile.tileFailure = &compute.BackendError{Backend: "test", Site: "later divergent tile", Err: errors.New("selected tile failed")}
		got, err := m.StepV41Batch(lanes)
		var numeric *V41ExpertOperationError
		if !got.Selected || !errors.As(err, &numeric) || numeric.Stage != "down" || errors.Is(err, b.tile.tileFailure) {
			t.Fatalf("issued singleton verdict lost: selected=%v err=%v", got.Selected, err)
		}
		for lane, result := range got.Lanes {
			if result.Committed || !lanes[lane].Session.BackendSessionClosed() || !reflect.DeepEqual(before[lane], captureV41ForwardSnapshot(lanes[lane].Session.v41State())) {
				t.Fatalf("failed divergent participant %d was not retired and rolled back", lane)
			}
		}
		if len(b.tile.tileRows) != 3 || len(b.tile.downRows) != 1 || b.tile.activationCalls != 4 || b.tileFences != 1 || len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
			t.Fatal("selected divergent failure replayed or leaked ownership")
		}
	})
	t.Run("divergent_cancelled_lane_rolls_back", func(t *testing.T) {
		b, lanes, refs := newFixture(t, 2)
		setRoutes(t, lanes, refs, divergent[:2])
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		lanes[0].Context = ctx
		before := captureV41ForwardSnapshot(lanes[0].Session.v41State())
		want := append([]float32(nil), refs[1].Step(lanes[1].Token)...)
		startTiles(b)
		b.onCohort = cancel
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || !errors.Is(got.Lanes[0].Err, context.Canceled) || got.Lanes[0].Committed || !got.Lanes[1].Committed || !v41SameBits(got.Lanes[1].Logits, want) {
			t.Fatalf("divergent cancellation result=%+v err=%v", got, err)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].Session.v41State())) || b.tileFences != 1 || len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
			t.Fatal("divergent cancellation retained state, replayed tiles or leaked ownership")
		}
	})
	t.Run("disjoint_routes_share_scalar_fence", func(t *testing.T) {
		b, lanes, refs := newFixture(t, 2)
		setRoutes(t, lanes, refs, [][]int{divergent[0], {6, 7, 8, 9, 10, 11}})
		want := make([][]float32, len(refs))
		for lane, s := range refs {
			want[lane] = append([]float32(nil), s.Step(lanes[lane].Token)...)
		}
		startTiles(b)
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || len(b.expertFences) != m.Cfg.NumLayers || len(b.tile.tileRows) != 0 || len(b.tile.downRows) != 0 {
			t.Fatalf("disjoint routes lost shared scalar fence or selected native tiles: fences=%v err=%v", b.expertFences, err)
		}
		for lane, result := range got.Lanes {
			if !result.Committed || result.Err != nil || !v41SameBits(result.Logits, want[lane]) {
				t.Fatalf("disjoint route changed lane %d", lane)
			}
		}
	})
	for _, count := range []int{2, 3, 4} {
		t.Run(itoa(count)+"_lanes", func(t *testing.T) {
			b, lanes, refs := newFixture(t, count)
			want := make([][]float32, count)
			for i, r := range refs {
				want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
			}
			startTiles(b)
			got, err := m.StepV41Batch(lanes)
			if err != nil || !got.Selected || len(got.Lanes) != count {
				t.Fatalf("actual batch selected=%v err=%v", got.Selected, err)
			}
			groups := m.Cfg.NumLayers * m.Cfg.NumExpertsPerTok
			if len(b.tile.tileRows) != 2*groups || len(b.tile.downRows) != groups || b.tileFences != m.Cfg.NumLayers || len(b.expertFences) != m.Cfg.NumLayers {
				t.Fatalf("missing shared-expert tiles/fence: gate/up=%v down=%v tile fences=%d expert fences=%v", b.tile.tileRows, b.tile.downRows, b.tileFences, b.expertFences)
			}
			for _, rows := range append(append([]int(nil), b.tile.tileRows...), b.tile.downRows...) {
				if rows != v41ExpertTileSize(count) {
					t.Fatalf("tile crossed a prepared expert boundary: rows=%d lanes=%d", rows, count)
				}
			}
			for i, statuses := range b.expertFences {
				if statuses != 2*count*m.Cfg.NumExpertsPerTok || b.outputsPerFence[i] != count {
					t.Fatalf("lost per-lane sums or split final fence: statuses=%v outputs=%v", b.expertFences, b.outputsPerFence)
				}
			}
			for i, lane := range got.Lanes {
				if lane.ID != lanes[i].ID || !lane.Committed || lane.Err != nil || !v41SameBits(lane.Logits, want[i]) {
					t.Fatalf("lane %d differed from scalar continuation", i)
				}
				if next, reference := lanes[i].Session.Step(6), refs[i].Step(6); !v41SameBits(next, reference) {
					t.Fatalf("lane %d continuation diverged after tiled cohort", i)
				}
			}
			if len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
				t.Fatal("tiled cohort retained status/panel/activation ownership")
			}
		})
	}
	t.Run("cancel_after_shared_tile_fence", func(t *testing.T) {
		b, lanes, refs := newFixture(t, 2)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		lanes[0].Context = ctx
		before := captureV41ForwardSnapshot(lanes[0].Session.v41State())
		want := append([]float32(nil), refs[1].Step(lanes[1].Token)...)
		startTiles(b)
		b.onCohort = cancel
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || !errors.Is(got.Lanes[0].Err, context.Canceled) || got.Lanes[0].Committed || !got.Lanes[1].Committed || !v41SameBits(got.Lanes[1].Logits, want) {
			t.Fatalf("cancelled tiled cohort result=%+v err=%v", got, err)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].Session.v41State())) || b.tileFences != 1 || len(b.tile.tileRows) != 2*m.Cfg.NumExpertsPerTok || len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
			t.Fatal("cancellation retained state, replayed tiles or leaked ownership")
		}
	})
	t.Run("earlier_issued_fault_before_later_selected_tile_failure", func(t *testing.T) {
		b, lanes, _ := newFixture(t, 2)
		before := make([]*v41ForwardSnapshot, len(lanes))
		for i, lane := range lanes {
			before[i] = captureV41ForwardSnapshot(lane.Session.v41State())
		}
		startTiles(b)
		b.publishFaultAt = map[int]bool{1: true}
		b.tile.failTileAt = 3 // Gate projection of the second prepared expert.
		b.tile.tileFailure = &compute.BackendError{Backend: "test", Site: "later cohort tile", Err: errors.New("selected tile failed")}
		got, err := m.StepV41Batch(lanes)
		var numeric *V41ExpertOperationError
		if !got.Selected || !errors.As(err, &numeric) || numeric.Stage != "down" || errors.Is(err, b.tile.tileFailure) {
			t.Fatalf("earlier issued primary lost: selected=%v err=%v", got.Selected, err)
		}
		for i, lane := range lanes {
			if got.Lanes[i].Committed || !lane.Session.BackendSessionClosed() || !reflect.DeepEqual(before[i], captureV41ForwardSnapshot(lane.Session.v41State())) {
				t.Fatalf("failed tile participant %d was not retired and rolled back", i)
			}
		}
		if len(b.tile.tileRows) != 3 || len(b.tile.downRows) != 1 || b.tile.activationCalls != 2 || b.tileFences != 1 || len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
			t.Fatal("selected failure replayed or leaked tile/status ownership")
		}
	})
	for _, stage := range []string{"gate/up", "down"} {
		t.Run(stage+"_packing_does_not_count_dispatch", func(t *testing.T) {
			b, lanes, _ := newFixture(t, 2)
			startTiles(b)
			b.packFault = stage
			before := m.V41ExpertFaultAttribution().Decode
			got, err := m.StepV41Batch(lanes)
			after := m.V41ExpertFaultAttribution().Decode
			gateUp := 0
			if stage == "down" {
				gateUp = 1
			}
			if !got.Selected || err == nil || after.IncrementalDeviceGateUpCalls-before.IncrementalDeviceGateUpCalls != gateUp || after.IncrementalDeviceDownCalls != before.IncrementalDeviceDownCalls || after.Contractions != before.Contractions || len(b.tile.tileRows) != 2*gateUp || len(b.tile.downRows) != 0 {
				t.Fatalf("packing failure fabricated dispatch: stage=%s err=%v before=%+v after=%+v", stage, err, before, after)
			}
			for i, lane := range lanes {
				if got.Lanes[i].Committed || !lane.Session.BackendSessionClosed() {
					t.Fatal("failed packing kept a selected participant reusable")
				}
			}
			if len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
				t.Fatal("failed packing leaked panel/status ownership")
			}
		})
	}
}
