package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Frozen scalar ordering from public 64ee7b08. Only the function/method names
// change; the remaining layer methods are outside this extraction's scope.
func (m *Model) referenceV41PlainLayerStep(l int, x []float32, streams [][]float32, pos int, layerState *V41AttentionState, scratch *v41ProjScratch) (work *v41LayerStepWork, err error) {
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}
	st := &v41PlainLayerStep{m: m, l: l, pos: pos, x: x, streams: streams, layerState: layerState, scratch: scratch, full: full, eps: float32(cfg.RMSNormEps)}
	if full {
		if err := scratch.mhcCarry.validate(l, 1); err != nil {
			return nil, err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return nil, err
		}
		st.mhcDevice = scratch.mhcCarry.device
		st.resident = scratch.mhcCarry.resident
	}
	if st.resident != nil {
		defer func() {
			if work == nil {
				st.resident.abort()
			}
		}()
	}
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)
	if err := m.v41RequirePlainLayerStepWeights(l); err != nil {
		return nil, err
	}
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	if err := st.collapseMHC(attnNorm, mixScale, mixBase); err != nil {
		return nil, err
	}
	qLat, prefixHandled, err := st.queryInput()
	if err != nil {
		return nil, err
	}
	q, err := st.projectQKV(qLat, prefixHandled)
	if err != nil {
		return nil, err
	}
	visibleKeys, err := st.visibleKeys()
	if err != nil {
		return nil, err
	}
	if err := st.referenceV41PlainAttend(q, sink, visibleKeys, projectOutput); err != nil {
		return nil, err
	}
	if err := st.ffnInput(); err != nil {
		return nil, err
	}
	st.picks, err = m.v41RouterPicksForStep(l, st.ffnX, H, st.routeCfg, full, scratch, func() []float32 { return gateBias })
	if err != nil {
		return nil, err
	}
	m.emitExpertRouteAt(l, pos, st.picks)
	work = &v41LayerStepWork{layer: l, input: st.ffnX, picks: st.picks, full: full, scratch: scratch}
	work.scalar = st.routed
	work.finish = st.finish
	work.publishKV = st.commitKV
	if st.resident != nil {
		work.cleanup = st.resident.abort
	}
	return work, nil
}

func (st *v41PlainLayerStep) referenceV41PlainAttend(q, sink []float32, visibleKeys []int32, projectOutput func([]float32) ([]float32, error)) error {
	m, l, pos, cfg, scratch, layerState := st.m, st.l, st.pos, st.m.Cfg, st.scratch, st.layerState
	hd, nH := cfg.HeadDim, cfg.NumHeads
	// The ring is not advanced until this layer succeeds; no borrowed row
	// reaches a callback.
	flatKV, err := layerState.stepWindowKV(l, pos, visibleKeys, st.kv)
	if err != nil {
		return err
	}
	idx := v41PlainWindowIndexList(visibleKeys)
	opened := m.v41NowNanos()
	o, err := v41SparseAttentionSinkResident(l, q, flatKV, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: len(visibleKeys) + 1, N: len(visibleKeys), Softmax: cfg.attnScale(),
	}, &v41ResidentKVView{
		owner: layerState, pos: pos, ringRows: layerState.windowSize, firstKey: int(visibleKeys[0]),
		windowRows: len(visibleKeys), sinkKey: layerName(l, "attn.sink"),
	}, scratch.sharedAttentionFunc())
	m.v41NoteAttentionContraction(opened)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	o, err = v41RotatedOutputForProjection(cfg, l, pos, o, nH, hd, scratch.gatherOps)
	if err != nil {
		return err
	}
	st.attnOut, err = projectOutput(o)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	scratch.activationTrace.record(l, pos, v41TraceStageAttnOut, st.attnOut)
	return nil
}

// fak-test:runtime fast est=1s lane=default
func TestV41PlainAttentionContinuationParity(t *testing.T) {
	for _, full := range []bool{false, true} {
		t.Run(fmt.Sprintf("full=%t", full), func(t *testing.T) {
			m := v41DecodeStateModel(t, 1)
			if full {
				m = v41FullStepPatchedNorms(t)
			}
			m.Cfg.Window = []int{8}
			for _, prefix := range [][]int{nil, {1, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 1, 2}} {
				seed := func() *V41AttentionState {
					if full {
						return v41FullStepSeed(t, m, 0, prefix)
					}
					return v41LayerStepSeed(t, m, 0, prefix)
				}
				xWant, sWant := v41LayerStepInputs(t, m, 4, full)
				stateWant := seed()
				scratchWant := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
				want, err := m.referenceV41PlainLayerStep(0, xWant, sWant, len(prefix), stateWant, scratchWant)
				if err != nil {
					t.Fatal(err)
				}
				if err = want.complete(nil); err != nil {
					t.Fatal(err)
				}
				for _, explicit := range []bool{false, true} {
					x, streams := v41LayerStepInputs(t, m, 4, full)
					state := seed()
					scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
					var work *v41LayerStepWork
					if explicit {
						c, cause := m.prepareV41PlainAttention(0, x, streams, len(prefix), state, scratch)
						if cause != nil {
							t.Fatal(cause)
						}
						copied := c
						projected, cause := c.project(c.input)
						if cause != nil {
							c.close()
							t.Fatal(cause)
						}
						work, err = c.resume(projected)
						if err != nil {
							t.Fatal(err)
						}
						c.close() // Ownership has transferred; this must not close layer work.
						copied.close()
						if c.st.attentionState != v41PlainAttentionTransferred || work.closed {
							t.Fatal("resume ownership transfer lost")
						}
						if _, cause = c.resume(projected); cause == nil {
							t.Fatal("second resume accepted")
						}
						if _, cause = copied.resume(projected); cause == nil {
							t.Fatal("copied continuation resumed twice")
						}
					} else {
						work, err = m.prepareV41PlainLayerStep(0, x, streams, len(prefix), state, scratch)
						if err != nil {
							t.Fatal(err)
						}
					}
					if err = work.complete(nil); err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(x, xWant) || !reflect.DeepEqual(streams, sWant) || !reflect.DeepEqual(state, stateWant) || !reflect.DeepEqual(scratch.mhcCarry.pre, scratchWant.mhcCarry.pre) {
						t.Fatalf("prefix%d explicit%t differs from frozen scalar output/state", len(prefix), explicit)
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41PlainAttentionContinuationBoundaries(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	prefix := []int{1, 2, 3}
	sentinel := errors.New("projection stopped")
	makeInput := func() ([]float32, [][]float32, *V41AttentionState, *v41ProjScratch) {
		x, streams := v41LayerStepInputs(t, m, 4, true)
		return x, streams, v41FullStepSeed(t, m, 0, prefix), &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
	}
	// Compare the actual pre-projection boundary to the old path interrupted at
	// its projection callback. Do not assume preparation has no retained writes.
	xOld, sOld, stateOld, scratchOld := makeInput()
	scratchOld.groupedOutput = func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionError, sentinel
	}
	if _, err := m.referenceV41PlainLayerStep(0, xOld, sOld, len(prefix), stateOld, scratchOld); !errors.Is(err, sentinel) {
		t.Fatalf("old stop=%v", err)
	}
	for _, action := range []string{"close", "bad-width", "projection-panic", "resume-panic"} {
		t.Run(action, func(t *testing.T) {
			x, streams, state, scratch := makeInput()
			c, err := m.prepareV41PlainAttention(0, x, streams, len(prefix), state, scratch)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(x, xOld) || !reflect.DeepEqual(streams, sOld) || !reflect.DeepEqual(state, stateOld) {
				t.Fatal("pre-projection publication boundary differs")
			}
			switch action {
			case "close":
				c.close()
				c.close()
			case "bad-width":
				if _, err = c.resume(nil); err == nil {
					t.Fatal("bad projection accepted")
				}
			case "projection-panic":
				c.project = func([]float32) ([]float32, error) { panic(sentinel) }
				var got any
				func() {
					defer func() { got = recover() }()
					defer c.close()
					_, _ = c.project(c.input)
				}()
				if got != sentinel {
					t.Fatalf("panic identity=%v", got)
				}
			case "resume-panic":
				projected, cause := c.project(c.input)
				if cause != nil {
					t.Fatal(cause)
				}
				delete(m.manifest, layerName(0, "ffn_norm.weight"))
				var got any
				func() { defer func() { got = recover() }(); _, _ = c.resume(projected) }()
				// Restore fixture before other subtests; the panic itself must escape.
				m = v41FullStepPatchedNorms(t)
				if got == nil {
					t.Fatal("resume panic was hidden")
				}
			}
			if c.st.attentionState != v41PlainAttentionClosed {
				t.Fatalf("state=%d, want closed", c.st.attentionState)
			}
			if _, err = c.resume(make([]float32, m.Cfg.HiddenSize)); err == nil {
				t.Fatal("closed continuation resumed")
			}
		})
	}
}

type v41PlainContinuationFreeBackend struct {
	compute.Backend
	frees int
	cause any
}

// fak-test:runtime fast est=200ms lane=default
func TestV41PlainAttentionContinuationAllocations(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	prefix := []int{1, 2, 3}
	measure := func(prepare func(int, []float32, [][]float32, int, *V41AttentionState, *v41ProjScratch) (*v41LayerStepWork, error)) float64 {
		x, streams := v41LayerStepInputs(t, m, 4, true)
		state := v41FullStepSeed(t, m, 0, prefix)
		scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
		return testing.AllocsPerRun(30, func() {
			work, err := prepare(0, x, streams, len(prefix), state, scratch)
			if err != nil {
				panic(err)
			}
			work.close() // No final FFN/window publication; reuse identical inputs.
		})
	}
	old := measure(m.referenceV41PlainLayerStep)
	now := measure(m.prepareV41PlainLayerStep)
	t.Logf("plain scalar prepare allocations: frozen=%g extracted=%g", old, now)
	if now > old {
		t.Fatalf("scalar continuation added allocations: frozen=%g extracted=%g", old, now)
	}
}

func (b *v41PlainContinuationFreeBackend) Free(x compute.Tensor) {
	b.frees++
	if b.cause != nil {
		panic(b.cause)
	}
	b.Backend.Free(x)
}

// fak-test:runtime fast est=10ms lane=default
func TestV41PlainAttentionContinuationCleanupOwnership(t *testing.T) {
	makePending := func(cause any) (*v41PlainAttentionContinuation, *v41PlainContinuationFreeBackend) {
		b := &v41PlainContinuationFreeBackend{Backend: compute.Default(), cause: cause}
		r := &v41MHCResident{d: &v41MHCDevice{s: &Session{Backend: b}}}
		r.layer.attn = compute.NewF32(b, []int{1}, []float32{1})
		return &v41PlainAttentionContinuation{st: &v41PlainLayerStep{resident: r}}, b
	}
	c, b := makePending(nil)
	c.close()
	c.close()
	if b.frees != 1 {
		t.Fatalf("pending cleanup frees=%d", b.frees)
	}
	c, b = makePending(nil)
	c.st.attentionState = v41PlainAttentionTransferred
	c.close()
	if b.frees != 0 {
		t.Fatal("close stole transferred resources")
	}
	work := &v41LayerStepWork{cleanup: c.st.resident.abort}
	work.close()
	work.close()
	if b.frees != 1 {
		t.Fatalf("work cleanup frees=%d", b.frees)
	}
	primary, secondary := errors.New("primary"), errors.New("secondary cleanup")
	c, b = makePending(secondary)
	var got any
	func() {
		defer func() { got = recover() }()
		defer c.close()
		panic(primary)
	}()
	// The scalar contract historically lets abort's panic replace a primary.
	// Extraction must not silently introduce a different failure policy.
	if got != secondary || b.frees != 1 {
		t.Fatalf("cleanup panic=%v frees=%d", got, b.frees)
	}
	c.close()
	if b.frees != 1 {
		t.Fatal("cleanup retried after panic")
	}
}
