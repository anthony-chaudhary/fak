package model

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// This fixture builds the target at rank32 before declaring its tensors. It
// executes real three-layer Engram/capture/retained-round paths. The draft stays
// original-format MXFP4/FP8/BF16; these synthetic bytes are not stock provenance.
func v41DSparkCommitFixture(t *testing.T) (*Model, *v41DSparkRows, *V41DSparkPackedShard) {
	t.Helper()
	m, layout := v41FullEngramModelWith(t, func(cfg *Config) {
		cfg.OLoraRank = 32
		cfg.NumLayers = 3
		cfg.Window = []int{-1, -1, -1}
		meta, draft := *cfg.DeepSeekV41, *cfg.DeepSeekV41.DSpark
		draft.TargetLayerIDs = []int{0, 1, 2}
		draft.NumRoutedExperts, draft.NumExpertsPerTok = 128, 3
		draft.MarkovRank, draft.NoiseTokenID = 4, 7
		meta.DSpark = &draft
		meta.CompressRatios = make([]int, 6)
		// This synthetic three-layer target uses window-only attention.
		meta.KVSourceLayerIDs = nil
		meta.IndexSourceLayerIDs = nil
		meta.CandidateSourceLayerID = -1
		cfg.DeepSeekV41 = &meta
	})
	t.Cleanup(func() { _ = m.CloseWeights() })
	original := make(map[string]tensorMeta)
	for name, tm := range m.manifest {
		if strings.HasPrefix(name, "model.layers.0.") {
			original[name] = tm
		}
	}
	for layer := 1; layer < 3; layer++ {
		for name, tm := range original {
			m.manifest[strings.Replace(name, "model.layers.0.", "model.layers."+itoa(layer)+".", 1)] = tm
		}
	}
	m.Cfg.DeepSeekV41.EngramLayerIDs = []int{0, 1, 2}
	m.Cfg.DeepSeekV41.EngramNumEmbeddings = []int{48, 48, 48}
	multipliers, primes := layout.Multipliers[0], layout.Primes[0]
	layout.Rows = []uint32{48, 48, 48}
	layout.Multipliers, layout.Primes = make([][]uint64, 3), make([][]uint32, 3)
	sources := make([]V41EngramRowSource, 3)
	for layer := range sources {
		layout.Multipliers[layer] = append([]uint64(nil), multipliers...)
		layout.Primes[layer] = append([]uint32(nil), primes...)
		sources[layer] = &v41EngramMemorySource{packed: v41EngramTestPackedRows(48), rows: 48}
	}
	if err := m.wireV41Engram(layout, sources, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatal(err)
	}
	inv, err := DeepSeekV41DSparkInventory(m.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]tinySTTensor)
	for _, name := range inv.Names() {
		spec, _ := inv.Lookup(name)
		n, ok := checkedShapeProduct(spec.Shape...)
		if !ok {
			t.Fatal("draft fixture size")
		}
		dtype := spec.AllowedDtypes[0]
		stride := 1
		if dtype == "BF16" {
			stride = 2
		}
		if dtype == "F32" {
			stride = 4
		}
		data := make([]byte, n*stride)
		for i := 0; i < n; i++ {
			value := float32(i%7-3) / 4096
			if strings.HasSuffix(name, "norm.weight") || strings.HasSuffix(name, "_scale") {
				value = 1
			}
			if strings.HasSuffix(name, "e_score_correction_bias") {
				value = float32(i) / 128
			}
			switch dtype {
			case "BF16":
				binary.LittleEndian.PutUint16(data[i*2:], uint16(math.Float32bits(v41RoundBF16(value))>>16))
			case "F32":
				binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
			case "I8":
				data[i] = 0x21
			case "F8_E4M3":
				data[i] = 0x20
				if i%5 == 0 {
					data[i] = 0xa0
				}
			default:
				t.Fatalf("unexpected fixture dtype %s", dtype)
			}
		}
		tensors[name] = tinySTTensor{dtype: dtype, shape: append([]int(nil), spec.Shape...), data: data}
		if len(spec.ScaleShape) != 0 {
			n, _ := checkedShapeProduct(spec.ScaleShape...)
			scales := make([]byte, n)
			for i := range scales {
				scales[i] = 121
			}
			tensors[strings.TrimSuffix(name, ".weight")+".scale"] = tinySTTensor{dtype: "F8_E8M0", shape: append([]int(nil), spec.ScaleShape...), data: scales}
		}
	}
	shard, err := OpenV41DSparkPackedShard(writeTinySafetensors(t, tensors), m.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = shard.Close() })
	w, err := newV41DSparkRows(m.Cfg, []*V41DSparkPackedShard{shard})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("window-only target fixture admission: %v", err)
	}
	return m, w, shard
}

func v41DSparkCommitReference(t *testing.T, s *Session, w *v41DSparkRows) *v41DSparkReference {
	t.Helper()
	f, err := newV41DSparkReference(s, w)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	return f
}

func v41DSparkCommitCold(t *testing.T, m *Model, w *v41DSparkRows, ids []int) (*v41DSparkReference, *v41DSparkCapture) {
	t.Helper()
	s := v41DSparkCaptureSession(t, m)
	f := v41DSparkCommitReference(t, s, w)
	c, err := newV41DSparkCapture(context.Background(), s, ids, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.close)
	if _, err := m.forwardV41WithHeadSelection(ids, s.v41Forward, true); err != nil {
		t.Fatal(err)
	}
	return f, c
}

func v41DSparkCommitRound(t *testing.T, f *v41DSparkReference, ids []int) (*v41StepRound, *v41DSparkCapture) {
	t.Helper()
	r, err := f.model.beginV41StepRound(f.state, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.abort() })
	c, err := newV41DSparkCapture(context.Background(), f.target, ids, false, r)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.close)
	for _, id := range ids {
		tx, err := r.beginStep(id, nil)
		if err != nil {
			t.Fatal(err)
		}
		for tx.nextLayer < tx.layers {
			if err := tx.advanceLayer(); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := tx.finalize(); err != nil {
			t.Fatal(err)
		}
	}
	return r, c
}

func v41DSparkCommitBoot(t *testing.T, m *Model, w *v41DSparkRows) *v41DSparkReference {
	t.Helper()
	f, c := v41DSparkCommitCold(t, m, w, []int{1, 2})
	if err := f.BootstrapCapturedTail(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	return f
}

func v41DSparkCommitHolds(m *Model) int {
	s := m.ensureWeightCloser()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sessions
}

func v41DSparkCommitClosedOnce(t *testing.T, f *v41DSparkReference) {
	t.Helper()
	m, before := f.model, v41DSparkCommitHolds(f.model)
	if !f.invalid || f.closed {
		t.Fatal("invalidation skipped the live Close obligation")
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := v41DSparkCommitHolds(m); got != before-1 {
		t.Fatalf("hold after Close=%d, want%d", got, before-1)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := v41DSparkCommitHolds(m); got != before-1 {
		t.Fatal("Close released the Model hold twice")
	}
	if f.model != nil || f.w != nil || f.target != nil || f.state != nil {
		t.Fatal("Close retained owners")
	}
}

// fak-test:runtime medium est=8s lane=default
func TestV41DSparkCapturedTailBootstrap(t *testing.T) {
	for _, n := range []int{1, 128, 129} {
		t.Run(itoa(n), func(t *testing.T) {
			m, w, _ := v41DSparkCommitFixture(t)
			ids := make([]int, n)
			for i := range ids {
				ids[i] = (i + 1) % m.Cfg.VocabSize
			}
			f, c := v41DSparkCommitCold(t, m, w, ids)
			rows := append([]v41DSparkContextRow(nil), c.rows...)
			if n == 129 {
				// This token lies outside the retained tail. A tail-only hash
				// would incorrectly admit the changed full prefix.
				old := f.state.history[0]
				f.state.history[0] = (old + 1) % m.Cfg.VocabSize
				if err := f.BootstrapCapturedTail(context.Background(), c); err == nil || f.next != 0 || c.closed {
					t.Fatal("changed evicted prefix admitted or foreign lineage consumed")
				}
				f.state.history[0] = old
			}
			if err := f.BootstrapCapturedTail(context.Background(), c); err != nil {
				t.Fatal(err)
			}
			if !c.closed || f.state.dsparkCapture != nil || f.next != n || f.prefix != v41DSparkPrefix(ids) {
				t.Fatal("bootstrap lost ownership or full-prefix lineage")
			}
			g := v41DSparkCommitReference(t, f.target, w)
			if n > v41WindowSize {
				serial := v41DSparkCaptureSession(t, m)
				first, _ := v41DSparkCapturedCold(t, serial, ids[:1])
				if err := g.AppendCommitted(context.Background(), first); err != nil {
					t.Fatal(err)
				}
			}
			if err := g.AppendCommitted(context.Background(), rows); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(f.ring, g.ring) || f.next != g.next || f.prefix != g.prefix {
				t.Fatal("tail differs from contiguous projection")
			}
			for _, row := range rows {
				if row.Position < max(0, n-128) || row.Position >= n {
					t.Fatal("tail positions renumbered")
				}
				for stage := range f.ring {
					if len(f.ring[stage][row.Position%128]) != m.Cfg.HeadDim {
						t.Fatal("absolute ring slot missing")
					}
				}
			}
			a, err := f.Propose(context.Background(), 1)
			b, other := g.Propose(context.Background(), 1)
			if err != nil || other != nil || !reflect.DeepEqual(a, b) {
				t.Fatalf("post-bootstrap proposal differs: %v/%v", err, other)
			}
			before := v41DSparkRingCopy(f)
			if err := f.BootstrapCapturedTail(context.Background(), c); err == nil || !reflect.DeepEqual(before, v41DSparkRingCopy(f)) {
				t.Fatal("bootstrap replay admitted")
			}
		})
	}
}

// fak-test:runtime medium est=8s lane=default
func TestV41DSparkCapturedRoundPublish(t *testing.T) {
	for keep := 1; keep <= 6; keep++ {
		t.Run(itoa(keep), func(t *testing.T) {
			m, w, _ := v41DSparkCommitFixture(t)
			f, g := v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)
			ids := []int{3, 4, 5, 6, 7, 0}
			r, c := v41DSparkCommitRound(t, f, ids)
			rows := append([]v41DSparkContextRow(nil), c.rows[:keep]...)
			if err := f.FinishCapturedRound(context.Background(), c, r, keep); err != nil {
				t.Fatal(err)
			}
			for _, id := range ids[:keep] {
				if _, _, err := m.forwardV41Step(id, g.state, nil); err != nil {
					t.Fatal(err)
				}
			}
			if err := g.AppendCommitted(context.Background(), rows); err != nil {
				t.Fatal(err)
			}
			if f.invalid || !r.closed || r.poisoned || !c.closed || f.state.activeStepRound != nil ||
				!reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(g.state)) ||
				!reflect.DeepEqual(f.ring, g.ring) || f.next != g.next || f.prefix != g.prefix {
				t.Fatal("two-owner publication differs from serial accepted prefix")
			}
			if err := f.FinishCapturedRound(context.Background(), c, r, keep); err == nil {
				t.Fatal("closed round replay admitted")
			}
			// A second real round proves rejected suffix state was restored.
			r2, c2 := v41DSparkCommitRound(t, f, []int{1})
			r3, c3 := v41DSparkCommitRound(t, g, []int{1})
			if err := f.FinishCapturedRound(context.Background(), c2, r2, 1); err != nil {
				t.Fatal(err)
			}
			if err := g.FinishCapturedRound(context.Background(), c3, r3, 1); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(f.ring, g.ring) || !reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(g.state)) {
				t.Fatal("next cycle differs after rejected suffix")
			}
		})
	}
}

// fak-test:runtime medium est=8s lane=default
func TestV41DSparkCapturedCommitFailureAtomicity(t *testing.T) {
	for _, mode := range []string{"read-error", "cancel", "panic", "panic-nil", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			m, w, shard := v41DSparkCommitFixture(t)
			f := v41DSparkCommitBoot(t, m, w)
			targetBefore, ringBefore, nextBefore, prefixBefore := captureV41ForwardSnapshot(f.state), v41DSparkRingCopy(f), f.next, f.prefix
			r, c := v41DSparkCommitRound(t, f, []int{3, 4})
			ref, err := shard.Reference("mtp.2.attn.wkv.weight")
			if err != nil {
				t.Fatal(err)
			}
			original := shard.packed.file.r
			defer func() { shard.packed.file.r = original }()
			fault := errors.New("late stage2 draft projection")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			hits := 0
			shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
				if off == ref.Start {
					hits++
					switch mode {
					case "read-error":
						return 0, fault
					case "cancel":
						cancel()
					case "panic":
						panic(fault)
					case "panic-nil":
						panic(nil)
					case "goexit":
						runtime.Goexit()
					}
				}
				return original.ReadAt(p, off)
			})
			var returned error
			var caught any
			normal := false
			done := make(chan struct{})
			go func() {
				defer close(done)
				defer func() { caught = recover() }()
				returned = f.FinishCapturedRound(ctx, c, r, 1)
				normal = true
			}()
			<-done
			shard.packed.file.r = original
			if hits == 0 {
				t.Fatal("fault did not reach final-stage draft read")
			}
			if mode == "read-error" && (!normal || !errors.Is(returned, fault)) {
				t.Fatalf("read cause lost: %v/%v", returned, caught)
			}
			if mode == "cancel" && (!normal || !errors.Is(returned, context.Canceled)) {
				t.Fatalf("cancel cause lost: %v/%v", returned, caught)
			}
			if mode == "panic" && (normal || caught != fault) {
				t.Fatalf("primary panic changed: %v", caught)
			}
			if (mode == "panic-nil" || mode == "goexit") && normal {
				t.Fatal("non-normal operation returned")
			}
			if !r.closed || r.poisoned || !c.closed || f.state.activeStepRound != nil || f.state.dsparkCapture != nil ||
				!reflect.DeepEqual(targetBefore, captureV41ForwardSnapshot(f.state)) ||
				!reflect.DeepEqual(ringBefore, v41DSparkRingCopy(f)) || f.next != nextBefore || f.prefix != prefixBefore {
				t.Fatal("preparation failure changed either owner or failed to drain")
			}
			if normal {
				if f.invalid {
					t.Fatal("clean preparation rollback retired usable draft")
				}
				if _, err := f.Propose(context.Background(), 1); err != nil {
					t.Fatal(err)
				}
			} else {
				if _, err := f.Propose(context.Background(), 1); err == nil {
					t.Fatal("non-normal operation left draft usable")
				}
				v41DSparkCommitClosedOnce(t, f)
			}
		})
	}
	t.Run("poisoned-target-drain", func(t *testing.T) {
		m, w, _ := v41DSparkCommitFixture(t)
		f := v41DSparkCommitBoot(t, m, w)
		before, next, prefix := v41DSparkRingCopy(f), f.next, f.prefix
		r, c := v41DSparkCommitRound(t, f, []int{3, 4})
		fault := errors.New("target undo fault")
		undo := r.entries[1].undo
		r.entries[1].undo = func() { undo(); panic(fault) }
		var caught any
		func() {
			defer func() { caught = recover() }()
			_ = f.FinishCapturedRound(context.Background(), c, r, 1)
		}()
		if caught != fault || !r.closed || !r.poisoned || !c.closed || !f.invalid || f.closed ||
			!reflect.DeepEqual(before, v41DSparkRingCopy(f)) || f.next != next || f.prefix != prefix {
			t.Fatal("poisoned target drain published or lost failure")
		}
		v41DSparkCommitClosedOnce(t, f)
	})

	t.Run("clean-drain-late-cancellation-still-publishes", func(t *testing.T) {
		m, w, _ := v41DSparkCommitFixture(t)
		f, g := v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)
		r, c := v41DSparkCommitRound(t, f, []int{3, 4})
		rows := append([]v41DSparkContextRow(nil), c.rows[:1]...)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		undo := r.entries[1].undo
		if undo == nil {
			t.Fatal("rejected target entry has no real undo")
		}
		undos := 0
		r.entries[1].undo = func() { undo(); undos++; cancel() }
		if err := f.FinishCapturedRound(ctx, c, r, 1); err != nil {
			t.Fatalf("clean finish rejected late cancellation: %v", err)
		}
		if undos != 1 || !errors.Is(ctx.Err(), context.Canceled) {
			t.Fatal("late-cancellation witness did not execute real undo")
		}
		if _, _, err := m.forwardV41Step(3, g.state, nil); err != nil {
			t.Fatal(err)
		}
		if err := g.AppendCommitted(context.Background(), rows); err != nil {
			t.Fatal(err)
		}
		if f.invalid || !r.closed || r.poisoned || !c.closed || f.state.activeStepRound != nil ||
			f.next != g.next || f.prefix != g.prefix ||
			!reflect.DeepEqual(v41DSparkRingCopy(f), v41DSparkRingCopy(g)) ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(g.state)) {
			t.Fatal("late cancellation changed successful two-owner publication")
		}
		if _, err := f.Propose(context.Background(), 1); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("foreign-owner-host-gate-and-zero", func(t *testing.T) {
		m, w, _ := v41DSparkCommitFixture(t)
		f, g := v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)
		r, c := v41DSparkCommitRound(t, g, []int{3})
		before := captureV41ForwardSnapshot(g.state)
		if err := f.FinishCapturedRound(context.Background(), c, r, 1); err == nil ||
			r.closed || c.closed || !r.ownsState() || !reflect.DeepEqual(before, captureV41ForwardSnapshot(g.state)) {
			t.Fatal("foreign owner consumed")
		}
		oldPolicy := g.target.executionPolicy
		g.target.executionPolicy = ExecutionPolicyDeviceOnly
		if err := g.FinishCapturedRound(context.Background(), c, r, 1); err == nil || r.closed || c.closed {
			t.Fatal("at-use host gate bypassed")
		}
		g.target.executionPolicy = oldPolicy
		if err := g.FinishCapturedRound(context.Background(), c, r, 0); err == nil || r.closed || c.closed {
			t.Fatal("zero silently became accepted publication")
		}
		if err := g.FinishCapturedRound(context.Background(), c, r, 1); err != nil {
			t.Fatal(err)
		}
	})
}
