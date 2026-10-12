package model

import (
	"errors"
	"math"
	"slices"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41DeferredLedger adds the deferred-status publication to a cpu-ref
// publication recorder. It rounds with the same integer RNE oracle and
// attributes the same lowest-index verdict the synchronous recorders return,
// so a session through it witnesses wiring, read order and ownership only,
// never a physical command. A nil ledger declines, which keeps every existing
// recorder on its synchronous status readback.
type v41DeferredLedger struct {
	owner             func() compute.Backend
	deferredPublishes int
	deferredFences    int
	deferredSites     []uint32
	deferredFaultAt   int
	deferredFault     error
	deferredPending   map[compute.Buffer]error
}

func newV41DeferredLedger(owner func() compute.Backend) *v41DeferredLedger {
	return &v41DeferredLedger{owner: owner, deferredPending: map[compute.Buffer]error{}}
}

func (l *v41DeferredLedger) SupportsV41DeferredStatus(n int) bool { return l != nil && n > 0 }

func (l *v41DeferredLedger) V41RoutedSwiGLUDeferred(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("the ordinary publication ledger issues no routed SwiGLU")
}

func (l *v41DeferredLedger) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if !compute.V41BF16PublishSite(site) || len(x.Shape) != 1 || x.Dtype != compute.F32 {
		panic("deferred publication operand")
	}
	l.deferredPublishes++
	l.deferredSites = append(l.deferredSites, site)
	values := append([]float32(nil), compute.Default().Read(x)...)
	var verdict error
	for i, v := range values {
		bits := math.Float32bits(v)
		rounded := v41BF16BoundaryOracle(bits)
		if verdict == nil && (bits&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000) {
			verdict = &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(rounded)
	}
	if l.deferredPublishes == l.deferredFaultAt {
		verdict = l.deferredFault
	}
	if verdict != nil {
		// The kernel still wrote a row; poison it so a row used past its
		// verdict cannot pass for a published one.
		for i := range values {
			values[i] = 7
		}
	}
	record := compute.NewF32(l.owner(), []int{4}, make([]float32, 4))
	l.deferredPending[record.Buf()] = verdict
	status := compute.V41DeferredStatus{Kind: compute.V41DeferredBF16Publish, Count: len(values), Site: site, Record: record}
	return compute.NewF32(l.owner(), []int{len(values)}, values), status, nil
}

func (l *v41DeferredLedger) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	l.deferredFences++
	verdicts := make([]error, len(statuses))
	for i, st := range statuses {
		verdict, ok := l.deferredPending[st.Record.Buf()]
		if !ok {
			return nil, nil, errors.New("deferred status read twice or never issued")
		}
		delete(l.deferredPending, st.Record.Buf())
		verdicts[i] = verdict
	}
	values := make([][]float32, len(rows))
	for i, row := range rows {
		values[i] = append([]float32(nil), compute.Default().Read(row)...)
	}
	return verdicts, values, nil
}

func newV41DeferredPublishRecorder() *v41BF16PublishRecorder {
	b := newV41BF16PublishRecorder()
	b.v41DeferredLedger = newV41DeferredLedger(b.owner)
	return b
}

func v41DeferredBitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: width %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s lane %d deferred %08x synchronous %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// v41DeferredPublishSites is every ordinary model boundary plus the two final
// norm sites, in a stable order.
func v41DeferredPublishSites(layers int) []struct {
	layer int
	name  string
	id    uint32
} {
	type site = struct {
		layer int
		name  string
		id    uint32
	}
	sites := []site{{-1, "model.norm.weight input", compute.V41BF16PublishFinalNormInput}, {-1, "model.norm.weight output", compute.V41BF16PublishFinalNormOutput}}
	names := make([]string, 0, len(v41BF16PublishSites))
	for name := range v41BF16PublishSites {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		sites = append(sites, site{layers - 1, name, v41BF16PublishSites[name]})
	}
	return sites
}

// Every ordinary site publishes the same bits through the deferred status as
// through the synchronous one, on the host and resident paths alike, with one
// fence per publication and no synchronous status readback; "0" in the A/B
// switch restores the synchronous form.
// fak-test:justify why=contract when=changed:internal/model/v41_bf16_publish*
// fak-test:runtime fast est=300ms lane=default
func TestV41BF16PublishDeferredMatchesSynchronous(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	H := m.Cfg.HiddenSize
	syncB, deferB := newV41BF16PublishRecorder(), newV41DeferredPublishRecorder()
	ss, sd := v41DenseTestSession(t, m, syncB), v41DenseTestSession(t, m, deferB)
	syncPublish, deferPublish := ss.v41BF16PublishFunc(), sd.v41BF16PublishFunc()
	if syncPublish == nil || deferPublish == nil {
		t.Fatal("ordinary publication not bound")
	}
	ledger := deferB.v41DeferredLedger
	sites := v41DeferredPublishSites(m.Cfg.NumLayers)
	ids := map[uint32]bool{}
	for i, st := range sites {
		ids[st.id] = true
		row := v41BF16PublishRow(H)
		row[i%H] = math.Float32frombits(0x7f7f7fff)
		want, err := syncPublish(st.layer, st.name, row)
		if err != nil {
			t.Fatalf("%s synchronous: %v", st.name, err)
		}
		calls, publishes, fences := deferB.calls, ledger.deferredPublishes, ledger.deferredFences
		got, err := deferPublish(st.layer, st.name, row)
		if err != nil {
			t.Fatalf("%s deferred: %v", st.name, err)
		}
		if deferB.calls != calls || ledger.deferredPublishes != publishes+1 || ledger.deferredFences != fences+1 ||
			ledger.deferredSites[len(ledger.deferredSites)-1] != st.id || len(ledger.deferredPending) != 0 {
			t.Fatalf("%s: synchronous=%d deferred=%d fences=%d pending=%d, want one deferred publication at site %d behind one fence",
				st.name, deferB.calls-calls, ledger.deferredPublishes-publishes, ledger.deferredFences-fences, len(ledger.deferredPending), st.id)
		}
		v41DeferredBitsEqual(t, st.name, got, want)
	}
	ordered := make([]uint32, 0, len(ids))
	for id := range ids {
		ordered = append(ordered, id)
	}
	slices.Sort(ordered)
	resident := v41BF16PublishRow(2 * H)
	for _, id := range ordered {
		x := compute.NewF32(compute.Default(), []int{2, H}, append([]float32(nil), resident...))
		p, ok, err := ss.v41PublishResidentBF16(x, id)
		if !ok || err != nil {
			t.Fatalf("site %d synchronous resident: ok=%t err=%v", id, ok, err)
		}
		want := append([]float32(nil), ss.Backend.Read(p)...)
		ss.Backend.Free(p)
		calls, fences := deferB.calls, ledger.deferredFences
		dp, status, deferred, err := sd.v41PublishResidentBF16Deferred(x, id)
		if !deferred || err != nil {
			t.Fatalf("site %d deferred resident: deferred=%t err=%v", id, deferred, err)
		}
		got, verdict, err := sd.v41ReadResidentBF16(dp, status)
		sd.Backend.Free(dp)
		if verdict != nil || err != nil || deferB.calls != calls || ledger.deferredFences != fences+1 || len(ledger.deferredPending) != 0 {
			t.Fatalf("site %d resident read: verdict=%v err=%v synchronous=%d fences=%d", id, verdict, err, deferB.calls-calls, ledger.deferredFences-fences)
		}
		v41DeferredBitsEqual(t, "resident", got, want)
	}

	t.Setenv(v41DeferredBF16PublishEnv, "0")
	row := v41BF16PublishRow(H)
	calls, publishes := deferB.calls, ledger.deferredPublishes
	got, err := deferPublish(0, "attn.kv_norm.weight projection", row)
	want, wantErr := syncPublish(0, "attn.kv_norm.weight projection", row)
	if err != nil || wantErr != nil || deferB.calls != calls+1 || ledger.deferredPublishes != publishes {
		t.Fatalf("switch off: err=%v synchronous=%d deferred=%d, want the synchronous publication", err, deferB.calls-calls, ledger.deferredPublishes-publishes)
	}
	v41DeferredBitsEqual(t, "switch off", got, want)
	x := compute.NewF32(compute.Default(), []int{H}, append([]float32(nil), row...))
	if _, _, deferred, _ := sd.v41PublishResidentBF16Deferred(x, compute.V41BF16PublishRoutedDown); deferred {
		t.Fatal("switch off still defers the resident publication")
	}
	if ss.BackendSessionClosed() || sd.BackendSessionClosed() {
		t.Fatal("successful publications retired a session")
	}
}

// A nonfinite word at k (with a later overflow) fails closed on the deferred
// read exactly as on the synchronous one: the same typed boundary, layer, site,
// stage and lowest index, no row, a retired session, a consumed status record
// and no replay.
// fak-test:justify why=contract when=changed:internal/model/v41_bf16_publish*
// fak-test:runtime fast est=200ms lane=default
func TestV41BF16PublishDeferredFaultAttribution(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	H := m.Cfg.HiddenSize
	layer, site := m.Cfg.NumLayers-1, "ffn_norm.weight collapse"
	for _, k := range []int{0, 3, H - 1} {
		row := v41BF16PublishRow(H)
		row[k] = float32(math.NaN())
		if k+1 < H {
			row[k+1] = math.Float32frombits(0x7f7f8000)
		}
		outcome := func(b *v41BF16PublishRecorder) (error, *Session, v41BF16BoundaryFunc) {
			s := v41DenseTestSession(t, m, b)
			publish := s.v41BF16PublishFunc()
			var out []float32
			var err error
			rounds := v41BF16PublishHostRounds(t, func() { out, err = publish(layer, site, row) })
			if out != nil || rounds != 0 || !s.BackendSessionClosed() {
				t.Fatalf("k=%d: row=%v host rounds=%d closed=%t, want no row and a retired session", k, out != nil, rounds, s.BackendSessionClosed())
			}
			return err, s, publish
		}
		syncErr, _, _ := outcome(newV41BF16PublishRecorder())
		b := newV41DeferredPublishRecorder()
		deferErr, _, publish := outcome(b)
		for arm, err := range map[string]error{"synchronous": syncErr, "deferred": deferErr} {
			var boundary *V41BF16BoundaryOperationError
			var closed *BackendForwardOperationError
			var numeric *compute.V41BF16CheckedArithmeticError
			if !errors.Is(err, ErrV41BF16BoundaryOperation) || !errors.As(err, &boundary) || boundary.Layer != layer || boundary.Site != site ||
				!errors.As(err, &closed) || closed.Path != "v41-bf16-publish" || closed.Stage != "checked publication" || closed.Layer != layer ||
				!errors.As(err, &numeric) || numeric.Index != k || numeric.Site != compute.V41BF16PublishFFNNormInput {
				t.Fatalf("k=%d %s: %v, want the typed boundary at layer %d site %q index %d", k, arm, err, layer, site, k)
			}
		}
		ledger := b.v41DeferredLedger
		if ledger.deferredPublishes != 1 || ledger.deferredFences != 1 || len(ledger.deferredPending) != 0 || b.calls != 0 {
			t.Fatalf("k=%d: deferred=%d fences=%d pending=%d synchronous=%d", k, ledger.deferredPublishes, ledger.deferredFences, len(ledger.deferredPending), b.calls)
		}
		var retry error
		if panicked := recoverError(func() { _, retry = publish(layer, site, row) }); (retry == nil && panicked == nil) || ledger.deferredPublishes != 1 {
			t.Fatalf("k=%d: retired deferred publication replayed", k)
		}
	}
}

// The three resident callers fold the status into their own row read: the
// same rows as the synchronous status, one fence per publication, and an
// injected verdict fails with the caller's own typed identity and stage.
// fak-test:justify why=contract when=changed:internal/model/v41_bf16_publish*
// fak-test:runtime fast est=500ms lane=default
func TestV41ResidentBF16DeferredCallers(t *testing.T) {
	t.Run("grouped-output", func(t *testing.T) {
		m := v41GroupedBF16Fixture(t)
		c := m.Cfg
		input := make([]float32, c.NumHeads*c.HeadDim)
		input[0], input[1] = 1, 1
		run := func(deferred bool, faultAt int) ([]float32, error, *Session, *v41GroupedPublishGapBackend) {
			b := &v41GroupedPublishGapBackend{v41GroupedBackend: newV41GroupedBackend(t, m), rank: -1}
			numeric := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishGroupedOutputProjection, Index: 2}
			if deferred {
				b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
				b.deferredFaultAt, b.deferredFault = faultAt, numeric
			} else {
				b.faultAt, b.fault = faultAt, numeric
			}
			s := v41DenseTestSession(t, m, b)
			out, outcome, err := s.v41GroupedOutputFunc()(0, append([]float32(nil), input...), c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			if (err == nil) != (outcome == v41ProjectionHandled) {
				t.Fatalf("deferred=%t: outcome %v with err %v", deferred, outcome, err)
			}
			return out, err, s, b
		}
		want, err, _, _ := run(false, 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err, s, b := run(true, 0)
		if err != nil || s.BackendSessionClosed() {
			t.Fatal(err)
		}
		v41DeferredBitsEqual(t, "grouped output", got, want)
		if n := c.OGroups + 1; b.publishes != 0 || b.deferredPublishes != n || b.deferredFences != n || len(b.deferredPending) != 0 {
			t.Fatalf("synchronous=%d deferred=%d fences=%d, want %d deferred publications each behind its own fence", b.publishes, b.deferredPublishes, b.deferredFences, n)
		}
		for _, faultAt := range []int{1, c.OGroups + 1} {
			_, syncErr, syncSession, _ := run(false, faultAt)
			out, deferErr, deferSession, b := run(true, faultAt)
			for arm, err := range map[string]error{"synchronous": syncErr, "deferred": deferErr} {
				var closed *BackendForwardOperationError
				var numeric *compute.V41BF16CheckedArithmeticError
				if !errors.As(err, &closed) || closed.Path != "v41-grouped-output" || closed.Stage != "BF16 publication" || closed.Layer != 0 ||
					!errors.As(err, &numeric) || numeric.Index != 2 {
					t.Fatalf("fault at %d %s: %v, want the grouped-output BF16 publication identity", faultAt, arm, err)
				}
			}
			if out != nil || !syncSession.BackendSessionClosed() || !deferSession.BackendSessionClosed() || len(b.deferredPending) != 0 || b.deferredPublishes != faultAt {
				t.Fatalf("fault at %d: row=%t closed=%t/%t pending=%d publications=%d", faultAt, out != nil, syncSession.BackendSessionClosed(), deferSession.BackendSessionClosed(), len(b.deferredPending), b.deferredPublishes)
			}
		}
	})
	t.Run("compressed-rope", func(t *testing.T) {
		v41PublicationRoPERequireCPU(t)
		run := func(deferred bool, fault string) ([]float32, []float32, error, *Session, *v41PublicationRoPERecorder) {
			m := &Model{Cfg: Config{HeadDim: 8, IndexHeadDim: 6, QKRopeHeadDim: 2, RopeTheta: 10000}}
			b := newV41PublicationRoPERecorder()
			b.publishes = true
			if deferred {
				b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
			}
			b.fault, b.failCall = fault, 1
			s := &Session{M: m, Backend: b}
			t.Cleanup(s.Close)
			t.Cleanup(func() { b.armed = false })
			latent := []float32{math.Float32frombits(0x80000000), 1.001, -2.003, 3, 4, 5, math.Float32frombits(0x41820000), math.Float32frombits(0xc0ee0000)}
			key := []float32{math.Float32frombits(0x80000000), 1.001, -2.003, 3, math.Float32frombits(0xc0ee0000), math.Float32frombits(0x41820000)}
			err := m.v41CompressedPublicationRoPEWithDevice(0, 2, latent, key, s.v41CompressedPublicationRoPEFunc())
			return latent, key, err, s, b
		}
		wantL, wantK, err, _, _ := run(false, "")
		if err != nil {
			t.Fatal(err)
		}
		gotL, gotK, err, s, b := run(true, "")
		if err != nil || s.BackendSessionClosed() {
			t.Fatal(err)
		}
		v41DeferredBitsEqual(t, "latent", gotL, wantL)
		v41DeferredBitsEqual(t, "index key", gotK, wantK)
		if len(b.publishSites) != 0 || !slices.Equal(b.deferredSites, []uint32{compute.V41BF16PublishTailRoPEOutput}) || b.deferredFences != 1 || len(b.deferredPending) != 0 {
			t.Fatalf("synchronous=%v deferred=%v fences=%d, want one deferred tail-RoPE output publication", b.publishSites, b.deferredSites, b.deferredFences)
		}
		v41PublicationRoPECensus(t, b, false)
		for _, fault := range []string{"index-nonfinite", "latent-overflow"} {
			_, _, syncErr, _, _ := run(false, fault)
			_, _, deferErr, deferSession, b := run(true, fault)
			var syncClosed, deferClosed *BackendForwardOperationError
			var syncStage, deferStage *V41ForwardError
			var syncIndex, deferIndex *V41ProjectionOperationError
			if !errors.As(syncErr, &syncClosed) || !errors.As(deferErr, &deferClosed) || syncClosed.Path != deferClosed.Path || syncClosed.Stage != deferClosed.Stage ||
				syncClosed.Layer != deferClosed.Layer || !errors.As(syncErr, &syncStage) || !errors.As(deferErr, &deferStage) ||
				syncStage.Stage != deferStage.Stage || syncStage.Layer != deferStage.Layer ||
				errors.As(syncErr, &syncIndex) != errors.As(deferErr, &deferIndex) || !errors.Is(deferErr, ErrV41ForwardStage) {
				t.Fatalf("%s: deferred %v, want the synchronous identity %v", fault, deferErr, syncErr)
			}
			if !deferSession.BackendSessionClosed() || len(b.deferredPending) != 0 || len(b.publishSites) != 0 {
				t.Fatalf("%s: closed=%t pending=%d synchronous=%v", fault, deferSession.BackendSessionClosed(), len(b.deferredPending), b.publishSites)
			}
			v41PublicationRoPECensus(t, b, false)
		}
	})
	t.Run("index-weight-scale", func(t *testing.T) {
		v41IndexWeightScaleRequireCPU(t)
		m := v41IndexerTestFixture(t)
		m.Cfg.IndexNHeads = 4
		scale := v41IndexWeightScale(128, 4)
		run := func(deferred bool, faultAt int, row []float32) ([]float32, error, *Session, *v41IndexWeightScaleRecorder) {
			b := newV41IndexWeightScaleRecorder()
			numeric := &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishIndexerScaledWeight, Index: 1}
			if deferred {
				b.v41BF16PublishRecorder.v41DeferredLedger = newV41DeferredLedger(b.owner)
				b.deferredFaultAt, b.deferredFault = faultAt, numeric
			} else {
				b.v41BF16PublishRecorder.faultAt, b.v41BF16PublishRecorder.fault = faultAt, numeric
			}
			s := v41DenseTestSession(t, m, b)
			// The seam takes the projection after its own publication.
			widened := make([]float32, len(row))
			for i, v := range row {
				widened[i] = v41RoundBF16(v)
			}
			out, err := s.v41IndexWeightScaleFunc()(3, widened, scale)
			return out, err, s, b
		}
		for r, row := range v41IndexWeightScaleRows(4) {
			want, err, _, _ := run(false, 0, row)
			if err != nil {
				t.Fatal(err)
			}
			got, err, s, b := run(true, 0, row)
			if err != nil || s.BackendSessionClosed() || b.calls != 0 || b.deferredPublishes != 1 || b.deferredFences != 1 || len(b.deferredPending) != 0 {
				t.Fatalf("row %d: err=%v synchronous=%d deferred=%d fences=%d", r, err, b.calls, b.deferredPublishes, b.deferredFences)
			}
			v41DeferredBitsEqual(t, "index weight scale", got, want)
		}
		row := v41IndexWeightScaleRows(4)[0]
		_, syncErr, _, _ := run(false, 1, row)
		out, deferErr, s, b := run(true, 1, row)
		for arm, err := range map[string]error{"synchronous": syncErr, "deferred": deferErr} {
			var closed *BackendForwardOperationError
			var numeric *compute.V41BF16CheckedArithmeticError
			if !errors.Is(err, ErrV41IndexWeightScaleOperation) || !errors.As(err, &closed) || closed.Path != "v41-index-weight-scale" ||
				closed.Stage != "scaled weight publication" || closed.Layer != 3 || !errors.As(err, &numeric) || numeric.Index != 1 {
				t.Fatalf("%s: %v, want the index-weight-scale publication identity", arm, err)
			}
		}
		if out != nil || !s.BackendSessionClosed() || len(b.deferredPending) != 0 {
			t.Fatalf("deferred fault: row=%t closed=%t pending=%d", out != nil, s.BackendSessionClosed(), len(b.deferredPending))
		}
	})
}

// A full-profile session through the deferred status keeps the synchronous
// session's logits bit for bit, publishes the same ordinary sites, and reads
// none of their status records on its own; the switch restores the old route.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41BF16PublishDeferredSession(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	m := v41FullStepPatchedNorms(t)
	run := func(b *v41BF16PublishRecorder) [][]float32 {
		s := v41DenseTestSession(t, m, b)
		out := [][]float32{append([]float32(nil), s.Prefill([]int{1, 2})...)}
		for _, token := range []int{3, 4} {
			out = append(out, append([]float32(nil), s.Step(token)...))
		}
		if s.BackendSessionClosed() {
			t.Fatal("session retired")
		}
		return out
	}
	syncB := newV41BF16PublishRecorder()
	want := run(syncB)
	deferB := newV41DeferredPublishRecorder()
	got := run(deferB)
	for i := range want {
		v41DeferredBitsEqual(t, "session logits", got[i], want[i])
	}
	ordinary := func(sites []uint32) []uint32 {
		var out []uint32
		for _, site := range sites {
			if site != 3 && site != 4 {
				out = append(out, site)
			}
		}
		slices.Sort(out)
		return out
	}
	ledger := deferB.v41DeferredLedger
	if len(ordinary(deferB.sites)) != 0 || ledger.deferredPublishes == 0 || ledger.deferredFences != ledger.deferredPublishes || len(ledger.deferredPending) != 0 {
		t.Fatalf("synchronous ordinary=%v deferred=%d fences=%d pending=%d", ordinary(deferB.sites), ledger.deferredPublishes, ledger.deferredFences, len(ledger.deferredPending))
	}
	if !slices.Equal(ordinary(syncB.sites), ordinary(ledger.deferredSites)) {
		t.Fatalf("deferred sites %v differ from the synchronous session's %v", ordinary(ledger.deferredSites), ordinary(syncB.sites))
	}

	t.Setenv(v41DeferredBF16PublishEnv, "0")
	offB := newV41DeferredPublishRecorder()
	off := run(offB)
	for i := range want {
		v41DeferredBitsEqual(t, "switch-off logits", off[i], want[i])
	}
	if offB.deferredPublishes != 0 || !slices.Equal(ordinary(offB.sites), ordinary(syncB.sites)) {
		t.Fatalf("switch off: deferred=%d synchronous ordinary=%d", offB.deferredPublishes, len(ordinary(offB.sites)))
	}
}
