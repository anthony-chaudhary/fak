package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41LayerCohortBackend struct {
	*v41ScalarCohortWitness
	owner           sync.Mutex
	ordinary        *v41BF16PublishRecorder
	expertFences    []int
	outputsPerFence []int
	onCohort        func()
	cohortFailure   error
	routerEnabled   bool
	routerArmed     bool
	routerRoutes    [][]int
	routerCalls     int
	expertMatMuls   int
}

func (b *v41LayerCohortBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41LayerCohortBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if w.Dtype == compute.Q2_K || w.Dtype == compute.Q3_K {
		b.expertMatMuls++
	}
	return b.v41ScalarCohortWitness.MatMul(w, x)
}
func (b *v41LayerCohortBackend) V41RouterTopKAvailable() bool { return b.routerEnabled }
func (b *v41LayerCohortBackend) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, routeScale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	if !b.routerArmed {
		all := b.Backend.Read(logits)
		var correction []float32
		if hasBias {
			correction = b.Backend.Read(bias)
		}
		cfg := v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: routeScale}
		var out compute.V41RouterPicks
		for row := 0; row < rows; row++ {
			picks, err := v41RouteForGeometry(all[row*experts:(row+1)*experts], correction, cfg, full)
			if err != nil {
				return compute.V41RouterPicks{}, err
			}
			for _, pick := range picks {
				out.Experts = append(out.Experts, int32(pick.expert))
				out.Weights = append(out.Weights, pick.weight)
			}
		}
		return out, nil
	}
	if rows != 1 || len(b.routerRoutes) == 0 {
		return compute.V41RouterPicks{}, errors.New("invalid scripted cohort route")
	}
	route := b.routerRoutes[b.routerCalls%len(b.routerRoutes)]
	b.routerCalls++
	if len(route) != topK {
		return compute.V41RouterPicks{}, errors.New("scripted cohort route has wrong top-k")
	}
	out := compute.V41RouterPicks{Experts: make([]int32, topK), Weights: make([]float32, topK)}
	denominator := float32(topK * (topK + 1) / 2)
	for slot, expert := range route {
		out.Experts[slot] = int32(expert)
		out.Weights[slot] = routeScale * float32(slot+1) / denominator
	}
	return out, nil
}

// Ordinary publication requires the outer Session backend identity. Reuse its
// existing checked CPU recorder; expert tensors keep their borrowed weight's
// cpu-ref identity, as required by the scalar expert protocol witness.
func (b *v41LayerCohortBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 BF16 publication input" {
		return b.ordinary.UploadClass(x, dt, class, site)
	}
	return b.Backend.Upload(x, dt)
}
func (b *v41LayerCohortBackend) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16Publish(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16Publish(x, site)
}
func (b *v41LayerCohortBackend) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16PublishDeferred(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16PublishDeferred(x, site)
}

func (b *v41LayerCohortBackend) V41SharedHeadAvailable(dt compute.Dtype, out, in, rows int) bool {
	return dt == compute.F32 && out > 0 && in > 0 && rows >= 1 && rows <= 4
}
func (b *v41LayerCohortBackend) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) > 0 && statuses[0].Record.Backend() == b {
		return b.ordinary.V41ReadDeferred(statuses, rows)
	}
	if len(statuses) >= 12 {
		b.expertFences = append(b.expertFences, len(statuses))
		b.outputsPerFence = append(b.outputsPerFence, len(rows))
		if len(statuses) > 12 {
			if b.onCohort != nil {
				f := b.onCohort
				b.onCohort = nil
				f()
			}
			if b.cohortFailure != nil {
				// A deferred read consumes the recorded statuses even on a
				// transport verdict; preserve that contract in this witness.
				_, _, _ = b.v41ScalarCohortWitness.V41ReadDeferred(statuses, rows)
				return nil, nil, b.cohortFailure
			}
		}
	}
	return b.v41ScalarCohortWitness.V41ReadDeferred(statuses, rows)
}
func (b *v41LayerCohortBackend) resetCohort() { b.expertFences = nil; b.outputsPerFence = nil }

// fak-test:runtime medium est=6s lane=default
func TestV41LayerCohortTransactions(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	// Real block-aligned Q2_K gate/up + Q3_K down arithmetic, not fabricated
	// cohort results. The host-resident provider is no hardware qualification.
	m := v41PinnedSlateFixture(t, false)
	defer m.CloseWeights()
	newFixture := func(t *testing.T) (*v41LayerCohortBackend, []V41BatchLane, []*Session) {
		t.Helper()
		b := &v41LayerCohortBackend{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}}
		b.ordinary = newV41DeferredPublishRecorder()
		b.ordinary.self = b
		var lanes []V41BatchLane
		var refs []*Session
		for i, prefix := range [][]int{{1, 2}, {1, 2, 3}} {
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(s.Close)
			r, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(r.Close)
			s.Prefill(prefix)
			r.Prefill(prefix)
			lanes = append(lanes, V41BatchLane{ID: uint64(91 - i*74), Session: s, Token: 4 + i})
			refs = append(refs, r)
		}
		b.resetCohort()
		return b, lanes, refs
	}
	newScriptedFixtureWithRing := func(t *testing.T, count int, ringBytes int64) (*v41LayerCohortTileBackend, []V41BatchLane, []*Session) {
		t.Helper()
		scalar := &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}
		base := &v41LayerCohortBackend{v41ScalarCohortWitness: scalar, routerEnabled: true}
		b := &v41LayerCohortTileBackend{
			v41LayerCohortBackend: base,
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
			lanes[i].Session.ExpertRingBytes = ringBytes
			t.Cleanup(lanes[i].Session.Close)
			refs[i], err = m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(refs[i].Close)
			lanes[i].Session.v41State().groupedOutput = nil
			refs[i].v41State().groupedOutput = nil
			prefix := []int{1, 2}
			for j := 0; j < i; j++ {
				prefix = append(prefix, 3+j)
			}
			lanes[i].ID, lanes[i].Token = uint64(191-i*23), 4+i
			lanes[i].Session.Prefill(prefix)
			refs[i].Prefill(prefix)
		}
		return b, lanes, refs
	}
	newScriptedFixture := func(t *testing.T, count int) (*v41LayerCohortTileBackend, []V41BatchLane, []*Session) {
		return newScriptedFixtureWithRing(t, count, 0)
	}
	serialScripted := func(b *v41LayerCohortTileBackend, lanes []V41BatchLane, refs []*Session, routes [][]int) [][]float32 {
		want := make([][]float32, len(routes))
		for i, ref := range refs {
			b.routerRoutes, b.routerCalls, b.routerArmed = [][]int{routes[i]}, 0, true
			want[i] = append([]float32(nil), ref.Step(lanes[i].Token)...)
		}
		return want
	}
	startScripted := func(b *v41LayerCohortTileBackend, routes [][]int) {
		b.resetCohort()
		b.publishes, b.fences, b.tile.activationCalls, b.tileFences, b.expertMatMuls = 0, 0, 0, 0, 0
		b.tile.tiles, b.tile.downTiles = true, true
		b.routerRoutes, b.routerCalls, b.routerArmed = routes, 0, true
	}
	runScriptedRoutes := func(t *testing.T, routes [][]int) {
		t.Helper()
		b, lanes, refs := newScriptedFixture(t, len(routes))
		want := serialScripted(b, lanes, refs, routes)
		startScripted(b, routes)
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || len(got.Lanes) != len(routes) {
			t.Fatalf("scripted batch selected=%v err=%v", got.Selected, err)
		}
		for i, result := range got.Lanes {
			if result.ID != lanes[i].ID || !result.Committed || result.Err != nil || !v41SameBits(result.Logits, want[i]) {
				t.Fatalf("scripted lane %d differed from serial route: %+v", i, result)
			}
		}
		if len(b.expertFences) != m.Cfg.NumLayers {
			t.Fatalf("scripted route did not share one layer fence: %v", b.expertFences)
		}
		for i, statuses := range b.expertFences {
			if statuses != 2*len(routes)*m.Cfg.NumExpertsPerTok || b.outputsPerFence[i] != len(routes) {
				t.Fatalf("scripted fence statuses=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		counts, order := map[int]int{}, []int{}
		for _, route := range routes {
			for _, expert := range route {
				if counts[expert] == 0 {
					order = append(order, expert)
				}
				counts[expert]++
			}
		}
		var tileRows []int
		scalarRows := 0
		for _, expert := range order {
			if rows := v41ExpertTileSize(counts[expert]); rows > 0 {
				tileRows = append(tileRows, rows)
			}
			if counts[expert] == 1 || counts[expert] == 3 {
				scalarRows++
			}
		}
		if len(b.tile.tileRows) != 2*len(tileRows)*m.Cfg.NumLayers || len(b.tile.downRows) != len(tileRows)*m.Cfg.NumLayers {
			t.Fatalf("scripted token tiles gate/up=%v down=%v want groups=%v", b.tile.tileRows, b.tile.downRows, tileRows)
		}
		wantTileFences := 0
		if len(tileRows) > 0 {
			wantTileFences = m.Cfg.NumLayers
		}
		if b.tileFences != wantTileFences {
			t.Fatalf("scripted tile fences=%d want %d", b.tileFences, wantTileFences)
		}
		if want := 3 * scalarRows * m.Cfg.NumLayers; b.expertMatMuls != want {
			t.Fatalf("scripted scalar expert matmuls=%d want %d (participants=%v)", b.expertMatMuls, want, counts)
		}
		for i, rows := range b.tile.tileRows {
			if rows != tileRows[(i/2)%len(tileRows)] {
				t.Fatalf("gate/up tile rows=%v want repeated %v", b.tile.tileRows, tileRows)
			}
		}
		for i, rows := range b.tile.downRows {
			if rows != tileRows[i%len(tileRows)] {
				t.Fatalf("down tile rows=%v want repeated %v", b.tile.downRows, tileRows)
			}
		}
	}
	t.Run("one real layer fence and serial continuation parity", func(t *testing.T) {
		b, lanes, refs := newFixture(t)
		want := make([][]float32, len(refs))
		for i, r := range refs {
			want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
		}
		if len(b.expertFences) != 2*m.Cfg.NumLayers {
			t.Fatalf("serial expert fences=%v", b.expertFences)
		}
		for i, n := range b.expertFences {
			if n != 12 || b.outputsPerFence[i] != 1 {
				t.Fatalf("serial layer fence rows=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected {
			t.Fatalf("batch selected=%v err=%v", got.Selected, err)
		}
		if len(b.expertFences) != m.Cfg.NumLayers {
			t.Fatalf("cohort expert fences=%v", b.expertFences)
		}
		for i, n := range b.expertFences {
			if n != 24 || b.outputsPerFence[i] != 2 {
				t.Fatalf("combined layer fence rows=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		for i, result := range got.Lanes {
			if result.ID != lanes[i].ID || !result.Committed || result.Err != nil || !reflect.DeepEqual(result.Logits, want[i]) {
				t.Fatalf("lane%d id=%d commit=%v err=%v logits match=%v", i, result.ID, result.Committed, result.Err, reflect.DeepEqual(result.Logits, want[i]))
			}
			if next, ref := lanes[i].Session.Step(6), refs[i].Step(6); !reflect.DeepEqual(next, ref) {
				t.Fatalf("lane%d continuation diverged", i)
			}
		}
	})
	t.Run("prefill-only panel selection preserves decode cohort", func(t *testing.T) {
		t.Setenv(v41PrefillExpertPanelEnv, "1")
		b, lanes, refs := newFixture(t)
		want := make([][]float32, len(refs))
		for i, r := range refs {
			want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
		}
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || len(b.expertFences) != m.Cfg.NumLayers {
			t.Fatalf("prefill-only flag disabled decode cohort: selected=%v fences=%v err=%v", got.Selected, b.expertFences, err)
		}
		for i, n := range b.expertFences {
			if n != 24 || b.outputsPerFence[i] != 2 {
				t.Fatalf("prefill-only layer fence rows=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		for i, result := range got.Lanes {
			if result.ID != lanes[i].ID || !result.Committed || result.Err != nil || !reflect.DeepEqual(result.Logits, want[i]) {
				t.Fatalf("prefill-only lane%d id=%d commit=%v err=%v", i, result.ID, result.Committed, result.Err)
			}
			if next, ref := lanes[i].Session.Step(6), refs[i].Step(6); !reflect.DeepEqual(next, ref) {
				t.Fatalf("prefill-only lane%d continuation diverged", i)
			}
		}
	})
	for _, test := range []struct {
		name   string
		routes [][]int
	}{
		{name: "permuted same set", routes: [][]int{{0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}}},
		{name: "partial overlap and singletons", routes: [][]int{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 6, 7, 8}, {0, 1, 2, 9, 10, 11}}},
		{name: "disjoint", routes: [][]int{{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}}},
	} {
		t.Run(test.name, func(t *testing.T) { runScriptedRoutes(t, test.routes) })
	}
	t.Run("partial overlap cancellation stays lane local", func(t *testing.T) {
		routes := [][]int{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 6, 7, 8}, {0, 1, 2, 9, 10, 11}}
		b, lanes, refs := newScriptedFixture(t, len(routes))
		want := serialScripted(b, lanes, refs, routes)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		lanes[0].Context = ctx
		before := captureV41ForwardSnapshot(lanes[0].Session.v41State())
		startScripted(b, routes)
		b.onCohort = func() {
			cancel()
			b.routerRoutes, b.routerCalls = routes[1:], 0
		}
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || !errors.Is(got.Lanes[0].Err, context.Canceled) || got.Lanes[0].Committed ||
			!got.Lanes[1].Committed || !got.Lanes[2].Committed || !v41SameBits(got.Lanes[1].Logits, want[1]) || !v41SameBits(got.Lanes[2].Logits, want[2]) {
			t.Fatalf("partial-overlap cancellation result=%+v err=%v", got, err)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].Session.v41State())) || len(b.expertFences) != m.Cfg.NumLayers {
			t.Fatalf("partial-overlap cancellation replayed or retained state: fences=%v", b.expertFences)
		}
	})
	t.Run("disjoint selected failure is not replayed", func(t *testing.T) {
		routes := [][]int{{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}}
		b, lanes, _ := newScriptedFixture(t, len(routes))
		before := make([]*v41ForwardSnapshot, len(lanes))
		for i, lane := range lanes {
			before[i] = captureV41ForwardSnapshot(lane.Session.v41State())
		}
		primary := &compute.BackendError{Backend: "cpu-contract", Class: compute.VulkanClassExecutionFailed, Err: errors.New("selected disjoint cohort fault")}
		startScripted(b, routes)
		b.cohortFailure = primary
		got, err := m.StepV41Batch(lanes)
		if !got.Selected || !errors.Is(err, primary) || len(b.expertFences) != 1 {
			t.Fatalf("disjoint selected fault replayed: selected=%v fences=%v err=%v", got.Selected, b.expertFences, err)
		}
		for i, lane := range lanes {
			if got.Lanes[i].Committed || !lane.Session.BackendSessionClosed() || !reflect.DeepEqual(before[i], captureV41ForwardSnapshot(lane.Session.v41State())) {
				t.Fatalf("disjoint failed lane %d was not retired and rolled back", i)
			}
		}
	})
	t.Run("late union ring decline releases holds before whole lane fallback", func(t *testing.T) {
		routes := [][]int{{0, 1, 2, 3, 4, 5}, {0, 6, 7, 8, 9, 10}}
		stem := "ffn.experts.0"
		tripleBytes := kQuantResidentBytes(m.kqw[layerName(0, stem+".w1.weight")]) +
			kQuantResidentBytes(m.kqw[layerName(0, stem+".w3.weight")]) +
			kQuantResidentBytes(m.kqw[layerName(0, stem+".w2.weight")])
		if tripleBytes <= 0 {
			t.Fatal("fixture has no quantized expert triple")
		}
		b, lanes, refs := newScriptedFixtureWithRing(t, len(routes), 2*tripleBytes)
		want := serialScripted(b, lanes, refs, routes)
		before := m.V41ExpertFaultAttribution().Decode
		startScripted(b, routes)
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected {
			t.Fatalf("late ring decline fallback selected=%v err=%v", got.Selected, err)
		}
		for i, result := range got.Lanes {
			if !result.Committed || result.Err != nil || !v41SameBits(result.Logits, want[i]) {
				t.Fatalf("late ring decline lane %d differed from serial fallback: %+v", i, result)
			}
			stats := lanes[i].Session.ExpertRing()
			if !stats.Enabled || stats.BudgetBytes != 2*tripleBytes || stats.PeakBytes > stats.BudgetBytes || stats.Evictions == 0 {
				t.Fatalf("late ring decline lane %d ring=%+v", i, stats)
			}
		}
		if len(b.expertFences) != 0 || b.tileFences != 0 || len(b.tile.tileRows) != 0 || len(b.tile.downRows) != 0 || len(b.pending) != 0 || len(b.tile.ownedTiles) != 0 {
			t.Fatalf("late ring decline selected cohort arithmetic or leaked ownership: fences=%v tile=%v/%v", b.expertFences, b.tile.tileRows, b.tile.downRows)
		}
		after := m.V41ExpertFaultAttribution().Decode
		wantCalls := len(routes) * m.Cfg.NumLayers * m.Cfg.NumExpertsPerTok
		if after.IncrementalDeviceGateUpCalls-before.IncrementalDeviceGateUpCalls != wantCalls ||
			after.IncrementalDeviceDownCalls-before.IncrementalDeviceDownCalls != wantCalls ||
			after.Contractions-before.Contractions != wantCalls || b.expertMatMuls != 3*wantCalls {
			t.Fatalf("late ring decline replayed or skipped expert work: before=%+v after=%+v matmuls=%d want calls=%d", before, after, b.expertMatMuls, wantCalls)
		}
	})
	t.Run("explicit ID selection retains whole-lane route", func(t *testing.T) {
		b, lanes, refs := newFixture(t)
		t.Setenv(v41RoutedIDEnv, "1")
		want := make([][]float32, 2)
		for i, r := range refs {
			want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
		}
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || len(b.expertFences) != 2*m.Cfg.NumLayers {
			t.Fatalf("explicit route fences=%v err=%v", b.expertFences, err)
		}
		for i, r := range got.Lanes {
			if !r.Committed || r.Err != nil || !reflect.DeepEqual(r.Logits, want[i]) {
				t.Fatalf("fallback lane%d err=%v", i, r.Err)
			}
		}
	})
	t.Run("cancelled row rolls back while peer commits", func(t *testing.T) {
		b, lanes, refs := newFixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		lanes[0].Context = ctx
		before := captureV41ForwardSnapshot(lanes[0].Session.v41State())
		want := append([]float32(nil), refs[1].Step(lanes[1].Token)...)
		b.onCohort = cancel
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !errors.Is(got.Lanes[0].Err, context.Canceled) || got.Lanes[0].Committed || !got.Lanes[1].Committed || !reflect.DeepEqual(got.Lanes[1].Logits, want) {
			t.Fatalf("cancel result=%+v err=%v", got, err)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].Session.v41State())) {
			t.Fatal("cancelled lane retained prepared state")
		}
		if len(b.expertFences) == 0 || b.expertFences[0] != 24 {
			t.Fatal("cancellation never traversed selected expert cohort")
		}
	})
	t.Run("selected cohort failure retires and rolls back every lane", func(t *testing.T) {
		b, lanes, _ := newFixture(t)
		before := make([]*v41ForwardSnapshot, len(lanes))
		for i, lane := range lanes {
			before[i] = captureV41ForwardSnapshot(lane.Session.v41State())
		}
		primary := &compute.BackendError{Backend: "cpu-contract", Class: compute.VulkanClassExecutionFailed, Err: errors.New("selected cohort fault")}
		b.cohortFailure = primary
		got, err := m.StepV41Batch(lanes)
		if !got.Selected || !errors.Is(err, primary) {
			t.Fatalf("selected=%v err=%v", got.Selected, err)
		}
		for i, lane := range lanes {
			if got.Lanes[i].ID != lane.ID || got.Lanes[i].Committed || !lane.Session.halClosed || !reflect.DeepEqual(before[i], captureV41ForwardSnapshot(lane.Session.v41State())) {
				t.Fatalf("lane%d was not retired/rolled back", i)
			}
		}
		if len(b.expertFences) != 1 {
			t.Fatalf("selected operation replayed: fences=%v", b.expertFences)
		}
	})
}
