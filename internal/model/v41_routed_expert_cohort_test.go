package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ScalarCohortWitness struct {
	*v41RoutedLayerTestBackend
	enabled            bool
	fences             int
	matmulInputs       [][]float32
	matmulOutputs      [][]float32
	cohortRows         []int
	failMatmul         int
	failure            error
	publicationErrorAt int
	cleanupErrorAt     int
	publicationRows    map[compute.Buffer]int
	advance            func(int64)
	readFailure        error
}

func (b *v41ScalarCohortWitness) SupportsV41DeferredStatus(n int) bool { return b.enabled && n > 0 }
func (b *v41ScalarCohortWitness) SupportsV41BF16Publish(n int) bool    { return n > 0 }
func (b *v41ScalarCohortWitness) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	p, status, err := b.v41RoutedLayerTestBackend.V41BF16PublishDeferred(x, site)
	if b.publicationRows == nil {
		b.publicationRows = map[compute.Buffer]int{}
	}
	b.publicationRows[p.Buf()] = b.publishes
	if b.publishes == b.publicationErrorAt {
		err = b.failure
	}
	return p, status, err
}
func (b *v41ScalarCohortWitness) Free(x compute.Tensor) {
	delete(b.pending, x.Buf())
	b.Backend.Free(x)
	if row := b.publicationRows[x.Buf()]; row > 0 && row == b.cleanupErrorAt {
		panic(b.failure)
	}
}
func (b *v41ScalarCohortWitness) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	p, status, err := b.V41BF16PublishDeferred(x, site)
	if err != nil {
		return compute.Tensor{}, err
	}
	verdicts, _, err := b.V41ReadDeferred([]compute.V41DeferredStatus{status}, nil)
	if err != nil {
		b.Free(p)
		return compute.Tensor{}, err
	}
	if verdicts[0] != nil {
		b.Free(p)
		return compute.Tensor{}, verdicts[0]
	}
	return p, nil
}
func (b *v41ScalarCohortWitness) V41RoutedSwiGLU(g, u compute.Tensor, limit, weight float32) (compute.Tensor, error) {
	b.fences++
	return b.v41RoutedLayerTestBackend.V41RoutedSwiGLU(g, u, limit, weight)
}
func (b *v41ScalarCohortWitness) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.advance != nil {
		b.advance(3)
	}
	b.matmulInputs = append(b.matmulInputs, append([]float32(nil), b.Backend.Read(x)...))
	if len(b.matmulInputs) == b.failMatmul {
		panic(b.failure)
	}
	y := b.v41HalSeamBackend.MatMul(w, x)
	b.matmulOutputs = append(b.matmulOutputs, append([]float32(nil), b.Backend.Read(y)...))
	return y
}
func (b *v41ScalarCohortWitness) Read(x compute.Tensor) []float32 {
	b.fences++
	return b.v41RoutedLayerTestBackend.Read(x)
}
func (b *v41ScalarCohortWitness) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.fences++
	if len(st) > 1 {
		b.cohortRows = append(b.cohortRows, len(st)/2)
	}
	verdicts, values, err := b.v41RoutedLayerTestBackend.V41ReadDeferred(st, rows)
	if b.advance != nil {
		b.advance(11)
	}
	if b.readFailure != nil {
		return nil, nil, b.readFailure
	}
	return verdicts, values, err
}

// The clock advances only on issued operations and completion, independently
// of diagnostic clock reads. These CPU recorders make no device-time claim.
type v41TimedCohortTileWitness struct{ *v41ExpertDownTileWitness }

func (b *v41TimedCohortTileWitness) V41Q2KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.advance(3)
	return b.v41ExpertTileWitness.V41Q2KTokenPanel(w, x, rows)
}

func (b *v41TimedCohortTileWitness) V41Q3KTokenPanel(w, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.advance(3)
	return b.v41ExpertDownTileWitness.V41Q3KTokenPanel(w, x, rows)
}

type v41EntryPointCohortBatch struct {
	weights           []compute.Buffer
	statuses          []compute.V41DeferredStatus
	readRows          int
	opened, completed int64
}

// Operand identities and completion records are independent of the model's
// phase ledger. Only the pinned fixture's routed experts use Q2_K/Q3_K.
type v41EntryPointCohortWitness struct {
	*v41ScalarCohortWitness
	ordinary    *v41BF16PublishRecorder
	now, opened int64
	weights     []compute.Buffer
	batches     []v41EntryPointCohortBatch
}

// Ordinary publication belongs to the Session backend; routed expert tensors
// retain the borrowed weights' cpu-ref identity, as in v41LayerCohortBackend.
func (b *v41EntryPointCohortWitness) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 BF16 publication input" {
		return b.ordinary.UploadClass(x, dt, class, site)
	}
	return b.Backend.Upload(x, dt)
}

func (b *v41EntryPointCohortWitness) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16Publish(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16Publish(x, site)
}

func (b *v41EntryPointCohortWitness) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16PublishDeferred(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16PublishDeferred(x, site)
}

func (b *v41EntryPointCohortWitness) MatMul(w, x compute.Tensor) compute.Tensor {
	if w.Dtype == compute.Q2_K || w.Dtype == compute.Q3_K {
		if len(b.weights) == 0 {
			b.opened = b.now
		}
		b.weights = append(b.weights, w.Buf())
	}
	return b.v41ScalarCohortWitness.MatMul(w, x)
}

func (b *v41EntryPointCohortWitness) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(st) > 0 && st[0].Record.Backend() == b {
		return b.ordinary.V41ReadDeferred(st, rows)
	}
	// Resident ordinary publications also retain cpu-ref identity, but only
	// routed batches carry an activation verdict alongside the down verdict.
	if len(st) == 0 || st[0].Kind != compute.V41DeferredRoutedSwiGLU {
		return b.v41ScalarCohortWitness.V41ReadDeferred(st, rows)
	}
	batch := v41EntryPointCohortBatch{
		weights: append([]compute.Buffer(nil), b.weights...), statuses: append([]compute.V41DeferredStatus(nil), st...),
		readRows: len(rows), opened: b.opened,
	}
	verdicts, values, err := b.v41ScalarCohortWitness.V41ReadDeferred(st, rows)
	batch.completed = b.now
	b.batches = append(b.batches, batch)
	b.weights = nil
	return verdicts, values, err
}

// The actual caller selects cold multirow cohorts, then one incremental decode.
// The clock advances on operations and completion only, never on clock reads;
// this is CPU wiring evidence, not a physical-device timing measurement.
// fak-test:runtime medium est=10s lane=default
func TestV41ColdPrefillSingleDecodeDispatchAttribution(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "1")
	m, controlModel := v41PinnedSlateFixture(t, false), v41PinnedSlateFixture(t, false)
	b := &v41EntryPointCohortWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{
		v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true,
	}, now: 100}
	b.ordinary = newV41DeferredPublishRecorder()
	b.ordinary.self = b
	b.advance = func(n int64) { b.now += n }
	m.v41SetExpertTimeClock(func() int64 { return b.now })
	defer m.v41SetExpertTimeClock(nil)
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	serial := &v41EntryPointCohortWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{
		v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(),
	}}
	serial.ordinary = newV41DeferredPublishRecorder()
	serial.ordinary.self = serial
	control, err := controlModel.NewBackendSessionChecked(serial)
	if err != nil {
		t.Fatal(err)
	}
	defer control.Close()
	checkLogits := func(phase string, got, want []float32) {
		t.Helper()
		if len(got) == 0 || !v41SameBits(got, want) {
			t.Fatalf("%s logits differ from independent serial session", phase)
		}
		for _, v := range got {
			if !finite32(v) {
				t.Fatalf("%s returned nonfinite logits", phase)
			}
		}
	}
	checkDispatch := func(phase string, ledger V41ExpertFaultAttributionPhase, batches []v41EntryPointCohortBatch, cold bool) {
		t.Helper()
		roles := map[compute.Buffer]int{}
		for layer := 0; layer < m.Cfg.NumLayers; layer++ {
			for expert := 0; expert < m.Cfg.NumExperts; expert++ {
				for role, leaf := range []string{"w1.weight", "w3.weight", "w2.weight"} {
					name := layerName(layer, "ffn.experts."+itoa(expert)+"."+leaf)
					if weight, ok := s.halW["kquant-raw:"+name]; ok {
						if _, duplicate := roles[weight.Buf()]; duplicate {
							t.Fatal("ambiguous expert operand identity")
						}
						roles[weight.Buf()] = role
					}
				}
			}
		}
		groups, nanos := 0, int64(0)
		if len(batches) == 0 {
			t.Fatalf("%s selected no deferred expert work", phase)
		}
		for i, batch := range batches {
			rows := len(batch.weights) / 3
			if rows == 0 || len(batch.weights) != 3*rows || len(batch.statuses) != 2*rows {
				t.Fatalf("%s batch %d has unmatched operations/statuses: %d/%d", phase, i, len(batch.weights), len(batch.statuses))
			}
			if cold && (rows < 2 || batch.readRows != rows) || !cold && batch.readRows != 1 {
				t.Fatalf("%s batch %d has %d expert rows and %d output rows", phase, i, rows, batch.readRows)
			}
			for j, weight := range batch.weights {
				role, ok := roles[weight]
				if !ok || role != j%3 || cold && weight != batch.weights[j%3] {
					t.Fatalf("%s batch %d operation %d is not the selected gate/up/down group", phase, i, j)
				}
			}
			for row := 0; row < rows; row++ {
				activation, publication := batch.statuses[2*row], batch.statuses[2*row+1]
				if activation.Kind != compute.V41DeferredRoutedSwiGLU || publication.Kind != compute.V41DeferredBF16Publish || publication.Site != compute.V41BF16PublishRoutedDown {
					t.Fatalf("%s batch %d row %d did not fence routed activation and down publication", phase, i, row)
				}
			}
			elapsed := int64(3*len(batch.weights) + 11)
			if batch.completed-batch.opened != elapsed {
				t.Fatalf("%s batch %d completion interval=%d want %d", phase, i, batch.completed-batch.opened, elapsed)
			}
			groups += rows
			nanos += elapsed
		}
		if ledger.IncrementalDeviceGateUpCalls != groups || ledger.IncrementalDeviceDownCalls != groups || ledger.IncrementalDeviceDispatchNanos != nanos {
			t.Fatalf("%s ledger gate/up=%d down=%d elapsed=%d; actual groups=%d elapsed=%d", phase,
				ledger.IncrementalDeviceGateUpCalls, ledger.IncrementalDeviceDownCalls, ledger.IncrementalDeviceDispatchNanos, groups, nanos)
		}
	}
	before := m.V41ExpertFaultAttribution()
	prompt := []int{1, 2, 3}
	checkLogits("prefill", s.Prefill(prompt), control.Prefill(prompt))
	prefill := m.V41ExpertFaultAttribution()
	if prefill.Prefill.Tokens != len(prompt) || !reflect.DeepEqual(prefill.Decode, before.Decode) {
		t.Fatalf("cold prefill changed phase/token accounting: prefill=%d decode=%d", prefill.Prefill.Tokens, prefill.Decode.Tokens)
	}
	checkDispatch("prefill", prefill.Prefill, b.batches, true)
	if m.v41ExpertFaultPhase() != V41PhaseUnknown || len(b.pending) != 0 || len(b.weights) != 0 || !s.v41IncrementalEligible() {
		t.Fatal("prefill did not leave a completed, inert, incremental-ready session")
	}
	firstDecodeBatch := len(b.batches)
	incremental := 0
	prior := v41IncrementalStepProbe
	v41IncrementalStepProbe = func() { incremental++ }
	defer func() { v41IncrementalStepProbe = prior }()
	got := s.Step(4)
	if incremental != 1 {
		t.Fatalf("Step executed %d incremental forwards, want exactly one", incremental)
	}
	checkLogits("decode", got, control.Step(4))
	after := m.V41ExpertFaultAttribution()
	if after.Decode.Tokens != prefill.Decode.Tokens+1 || !reflect.DeepEqual(after.Prefill, prefill.Prefill) {
		t.Fatal("one decode changed the prefill snapshot or the decode token denominator")
	}
	checkDispatch("decode", after.Decode, b.batches[firstDecodeBatch:], false)
	if m.v41ExpertFaultPhase() != V41PhaseUnknown || len(b.pending) != 0 || len(b.weights) != 0 || s.BackendSessionClosed() || len(serial.cohortRows) != 0 || serial.calls == 0 {
		t.Fatal("completion, phase restoration, or independent serial control changed")
	}
	s.Close()
	control.Close()
	if len(b.pending) != 0 || len(serial.pending) != 0 || len(b.ordinary.deferredPending) != 0 || len(serial.ordinary.deferredPending) != 0 {
		t.Fatal("session teardown retained deferred records")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41ColdCohortDispatchAttribution(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs, weights := make([][]float32, 2), []float32{.25, .5}
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((r*3+i)%13-6) / 16
		}
	}
	control, err := m.NewBackendSessionChecked(&v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend()})
	if err != nil {
		t.Fatal(err)
	}
	defer control.Close()
	want := make([][]float32, len(inputs))
	picks := [][]routePick{{{expert: 0, weight: weights[0]}}, {{expert: 0, weight: weights[1]}}}
	state := control.v41State()
	state.routedSum = nil
	if err := m.v41ContractRoutedGrouped(0, inputs, picks, &v41ProjScratch{}, m.Cfg, want, state); err != nil {
		t.Fatal(err)
	}
	defer m.v41SetExpertTimeClock(nil)
	for _, tc := range []struct {
		name                 string
		tiles, decline       bool
		failMatmul, failRead bool
		gateUp, down, fences int
		nanos                int64
	}{
		{name: "scalar", gateUp: 2, down: 2, fences: 1, nanos: 29},
		{name: "token_tiles", tiles: true, gateUp: 1, down: 1, fences: 1, nanos: 20},
		{name: "unavailable", decline: true},
		{name: "first_dispatch_failure", failMatmul: true, gateUp: 1, nanos: 3},
		{name: "readback_failure", failRead: true, gateUp: 2, down: 2, fences: 1, nanos: 29},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := int64(100)
			b := &v41ScalarCohortWitness{
				v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: !tc.decline,
				advance: func(n int64) { now += n },
			}
			var backend compute.Backend = b
			var tiled *v41TimedCohortTileWitness
			if tc.tiles {
				tiled = &v41TimedCohortTileWitness{&v41ExpertDownTileWitness{
					v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: b, tiles: true, ownedTiles: map[compute.Buffer]bool{}},
					downTiles:            true,
				}}
				backend = tiled
			}
			s, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			failure := &compute.BackendError{Backend: "test", Site: tc.name, Err: errors.New("selected failure")}
			if tc.failMatmul {
				b.failMatmul, b.failure = 1, failure
			}
			if tc.failRead {
				b.readFailure = failure
			}
			m.v41SetExpertTimeClock(func() int64 { return now })
			m.v41SetExpertFaultPhase(V41PhasePrefill)
			before := m.V41ExpertFaultAttribution()
			got, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
			m.v41SetExpertFaultPhase(V41PhaseUnknown)
			after := m.V41ExpertFaultAttribution()
			pre, old := after.Prefill, before.Prefill
			if pre.IncrementalDeviceGateUpCalls-old.IncrementalDeviceGateUpCalls != tc.gateUp ||
				pre.IncrementalDeviceDownCalls-old.IncrementalDeviceDownCalls != tc.down ||
				pre.IncrementalDeviceDispatchNanos-old.IncrementalDeviceDispatchNanos != tc.nanos {
				t.Fatalf("dispatch gate/up=%d down=%d elapsed=%d; want %d/%d/%d",
					pre.IncrementalDeviceGateUpCalls-old.IncrementalDeviceGateUpCalls,
					pre.IncrementalDeviceDownCalls-old.IncrementalDeviceDownCalls,
					pre.IncrementalDeviceDispatchNanos-old.IncrementalDeviceDispatchNanos, tc.gateUp, tc.down, tc.nanos)
			}
			if !reflect.DeepEqual(after.Decode, before.Decode) || b.fences != tc.fences || len(b.pending) != 0 {
				t.Fatalf("decode attribution or ownership changed: fences=%d pending=%d", b.fences, len(b.pending))
			}
			switch {
			case tc.decline:
				if outcome != v41RoutedExpertDeclined || got != nil || err != nil || len(b.matmulInputs) != 0 || s.BackendSessionClosed() {
					t.Fatalf("decline dispatched or retired: outcome=%v error=%v", outcome, err)
				}
			case tc.failMatmul || tc.failRead:
				if outcome != v41RoutedExpertError || got != nil || !errors.Is(err, failure) || !s.BackendSessionClosed() {
					t.Fatalf("selected failure lost: outcome=%v error=%v", outcome, err)
				}
			default:
				if outcome != v41RoutedExpertHandled || err != nil || len(got) != len(want) || s.BackendSessionClosed() {
					t.Fatalf("cohort not handled: outcome=%v error=%v", outcome, err)
				}
				for r := range got {
					if !v41SameBits(got[r], want[r]) {
						t.Fatalf("row %d differs from serial control", r)
					}
				}
				if tiled != nil && (!reflect.DeepEqual(tiled.tileRows, []int{2, 2}) || !reflect.DeepEqual(tiled.downRows, []int{2}) || len(tiled.ownedTiles) != 0 || len(b.matmulInputs) != 0) {
					t.Fatal("actual tile groups or ownership changed")
				}
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41ExpertScalarCohortProtocol(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	const count = 33 // Causally requires a 31-row cohort and a two-row tail.
	m := v41PinnedSlateFixture(t, false)
	inputs, picks := make([][]float32, count), make([][]routePick, count)
	weights := make([]float32, count)
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((r+i)%13-6) / 16
		}
		weights[r] = float32(r+1) / 64
		picks[r] = []routePick{{expert: 0, weight: weights[r]}}
	}
	newSession := func(enabled bool) (*Session, *v41ScalarCohortWitness) {
		b := &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: enabled}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	run := func(s *Session) [][]float32 {
		out := make([][]float32, count)
		state := s.v41State()
		state.routedSum = nil
		err := m.v41ContractRoutedGrouped(0, inputs, picks, &v41ProjScratch{}, m.Cfg, out, state)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	serial, control := newSession(false)
	batched, candidate := newSession(true)
	want, got := run(serial), run(batched)
	for r := range got {
		if !v41SameBits(got[r], want[r]) {
			t.Fatalf("row %d bits differ", r)
		}
	}
	if len(candidate.matmulInputs) != 3*count || !reflect.DeepEqual(candidate.matmulInputs, control.matmulInputs) || !reflect.DeepEqual(candidate.matmulOutputs, control.matmulOutputs) {
		t.Fatal("scalar MatMul count, operands, outputs, or execution order changed")
	}
	if !reflect.DeepEqual(candidate.cohortRows, []int{31, 2}) || candidate.fences != 2 || control.fences != 3*count {
		t.Fatalf("cohort rows=%v candidate fences=%d serial fences=%d; want [31 2],2,99", candidate.cohortRows, candidate.fences, control.fences)
	}
	if len(candidate.pending) != 0 {
		t.Fatal("cohort leaked deferred statuses")
	}
	// An earlier row's checked down verdict must remain primary even when
	// recording a later row fails before the final read. The earlier-only
	// serial arm returns the same numeric stage; no output or host replay occurs.
	t.Run("earlier_verdict_before_later_recording_failure", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.failMatmul = 4
		b.failure = &compute.BackendError{Backend: "test", Site: "later gate", Err: errors.New("later recording failure")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		var numeric *V41ExpertOperationError
		if outcome != v41RoutedExpertError || out != nil || !errors.As(err, &numeric) || numeric.Stage != "down" || errors.Is(err, b.failure) {
			t.Fatalf("earlier down verdict lost: outcome=%v output=%v error=%v", outcome, out, err)
		}
		if !s.BackendSessionClosed() {
			t.Fatal("later selected recording failure left session reusable")
		}
		if len(b.pending) != 0 || b.fences != 1 {
			t.Fatalf("pending=%d fences=%d", len(b.pending), b.fences)
		}
	})
	t.Run("failing_operation_record_cannot_replace_error", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.publicationErrorAt = 1
		b.failure = &compute.BackendError{Backend: "test", Site: "publication", Err: errors.New("selected publication error")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		if outcome != v41RoutedExpertError || out != nil || !errors.Is(err, b.failure) || !s.BackendSessionClosed() || len(b.pending) != 0 {
			t.Fatalf("operation error replaced or ownership retained: outcome=%v error=%v pending=%d", outcome, err, len(b.pending))
		}
	})
	t.Run("earlier_verdict_still_retires_later_cleanup_failure", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.cleanupErrorAt = 2
		b.failure = errors.New("later cleanup panic")
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		var numeric *V41ExpertOperationError
		if outcome != v41RoutedExpertError || out != nil || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || len(b.pending) != 0 {
			t.Fatalf("earlier verdict or retirement lost: outcome=%v error=%v pending=%d", outcome, err, len(b.pending))
		}
	})
}
