package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// statefulGDNBackend is a host-memory resident GDN owner double: it seeds and
// snapshots per-handle flat state exactly as the Metal backend's contract does,
// so the restored-prefix seeding and owner->host synchronization seams are
// exercised without any device.
type statefulGDNBackend struct {
	*fakeQwen35GDNSequenceBackend
	state     map[Qwen35GDNAuxState]qwen35GDNLayerSnapshot
	failSeed  bool
	snapshots int
}

func newStatefulGDNBackend() *statefulGDNBackend {
	return &statefulGDNBackend{fakeQwen35GDNSequenceBackend: newFakeQwen35GDNSequenceBackend(), state: make(map[Qwen35GDNAuxState]qwen35GDNLayerSnapshot)}
}

func (b *statefulGDNBackend) SeedQwen35GDNAuxState(state Qwen35GDNAuxState, conv, recurrent []float32) error {
	if b.failSeed {
		return errors.New("injected seed failure")
	}
	b.state[state] = qwen35GDNLayerSnapshot{conv: append([]float32(nil), conv...), recurrent: append([]float32(nil), recurrent...)}
	return nil
}

func (b *statefulGDNBackend) SnapshotQwen35GDNAuxState(state Qwen35GDNAuxState) ([]float32, []float32, error) {
	b.snapshots++
	got, ok := b.state[state]
	if !ok {
		return nil, nil, fmt.Errorf("owner %#v was never seeded", state)
	}
	return append([]float32(nil), got.conv...), append([]float32(nil), got.recurrent...), nil
}

// installStatefulGDNFactory makes Enable admit the host double on any build.
func installStatefulGDNFactory(t *testing.T, b *statefulGDNBackend) {
	t.Helper()
	prevFactory, prevAdmission := newQwen35MetalGDNSequenceBackend, qwen35MetalForwardProjectionAdmission
	newQwen35MetalGDNSequenceBackend = func() Qwen35GDNPreprojectedSequenceBackend { return b }
	qwen35MetalForwardProjectionAdmission = nil
	t.Cleanup(func() {
		newQwen35MetalGDNSequenceBackend, qwen35MetalForwardProjectionAdmission = prevFactory, prevAdmission
	})
}

// restoredHybridSession returns a resident-flagged session restored from a host
// prefix of n tokens, plus the model.
func restoredHybridSession(t *testing.T, n int) (*Model, *Session) {
	t.Helper()
	m := NewSynthetic(qwen35HybridTestCfg())
	donor := m.NewSession()
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*13 + 5) % m.Cfg.VocabSize
	}
	donor.Prefill(ids)
	snap, err := donor.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	s := m.NewSession()
	if err := snap.Restore(s); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	donor.Close()
	s.Q4K, s.MetalQ4K = true, true
	return m, s
}

func hostLinearFlat(t *testing.T, cfg Config, c *KVCache, layer int) qwen35GDNLayerSnapshot {
	t.Helper()
	conv, recurrent := oracleQwen35GDNSeedLayout(t, cfg, &c.linear.layers[layer])
	return qwen35GDNLayerSnapshot{layer: layer, conv: conv, recurrent: recurrent}
}

// advanceOwners overwrites every owner with distinct synthetic state, standing in
// for device-side recurrence the host cache never observed.
func advanceOwners(s *Session, b *statefulGDNBackend, bias float32) map[int]qwen35GDNLayerSnapshot {
	cfg := s.M.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	want := make(map[int]qwen35GDNLayerSnapshot)
	for layer, owner := range s.qwen35HAL.sequenceLayers {
		if !owner.valid() {
			continue
		}
		conv := make([]float32, (cfg.LinearConvKernelDim-1)*convDim)
		rec := make([]float32, nV*kHd*vHd)
		for i := range conv {
			conv[i] = bias + float32(layer) + float32(i)/1000
		}
		for i := range rec {
			rec[i] = -bias - float32(layer) - float32(i)/1000
		}
		b.state[owner] = qwen35GDNLayerSnapshot{conv: conv, recurrent: rec}
		want[layer] = qwen35GDNLayerSnapshot{layer: layer, conv: conv, recurrent: rec}
	}
	return want
}

func assertHostLinearEquals(t *testing.T, label string, cfg Config, c *KVCache, want map[int]qwen35GDNLayerSnapshot) {
	t.Helper()
	if len(want) == 0 {
		t.Fatalf("%s: no owner state to compare", label)
	}
	for layer, w := range want {
		got := hostLinearFlat(t, cfg, c, layer)
		if !reflect.DeepEqual(got.conv, w.conv) || !reflect.DeepEqual(got.recurrent, w.recurrent) {
			t.Fatalf("%s: layer %d host linear state does not match owner state", label, layer)
		}
	}
}

// assertHostCacheUnchanged checks KV rows and the host linear-attention state.
func assertHostCacheUnchanged(t *testing.T, label string, cfg Config, want, got *KVCache) {
	t.Helper()
	assertKVCacheQuantClose(t, label, want, got)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		if !cfg.isLinearAttnLayer(layer) {
			continue
		}
		w, g := hostLinearFlat(t, cfg, want, layer), hostLinearFlat(t, cfg, got, layer)
		if !reflect.DeepEqual(w.conv, g.conv) || !reflect.DeepEqual(w.recurrent, g.recurrent) {
			t.Fatalf("%s: layer %d host linear state changed", label, layer)
		}
	}
}

// oracleQwen35GDNSeedLayout is written independently of
// flattenQwen35HostGDNLayer: the owner seed layout is exactly K-1 convolution
// rows oldest first, with leading zero rows standing in for causal history the
// host never saw (history < K-1), followed by recurrent state head-major.
func oracleQwen35GDNSeedLayout(t *testing.T, cfg Config, st *linearAttnLayerState) (conv, recurrent []float32) {
	t.Helper()
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	keep := cfg.LinearConvKernelDim - 1
	missing := keep - len(st.conv)
	if missing < 0 {
		t.Fatalf("host carries %d conv rows, more than K-1=%d", len(st.conv), keep)
	}
	for row := 0; row < keep; row++ {
		for col := 0; col < convDim; col++ {
			if row < missing {
				conv = append(conv, 0)
			} else {
				conv = append(conv, st.conv[row-missing][col])
			}
		}
	}
	if len(st.recurrent) != nV {
		t.Fatalf("host recurrent heads=%d, want %d", len(st.recurrent), nV)
	}
	for head := 0; head < nV; head++ {
		for i := 0; i < kHd*vHd; i++ {
			recurrent = append(recurrent, st.recurrent[head][i])
		}
	}
	return conv, recurrent
}

func TestQwen35GDNEnableSeedsOwnersFromRestoredHostPrefix(t *testing.T) {
	keep := qwen35HybridTestCfg().LinearConvKernelDim - 1
	for _, n := range []int{5, keep - 1} {
		t.Run(fmt.Sprintf("prefix=%d", n), func(t *testing.T) {
			if n < 1 {
				t.Skip("kernel geometry leaves no history shorter than K-1")
			}
			b := newStatefulGDNBackend()
			installStatefulGDNFactory(t, b)
			_, s := restoredHybridSession(t, n)
			defer s.Close()
			if s.Cache.Len() != n {
				t.Fatalf("restored cache len=%d, want %d", s.Cache.Len(), n)
			}
			before := s.Cache.Clone()
			if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
				t.Fatalf("restored-prefix admission: %v", err)
			}
			if !s.qwen35HAL.sequenceAccepted || s.qwen35HAL.decodeAccepted {
				t.Fatalf("restored admission state sequence=%v decode=%v", s.qwen35HAL.sequenceAccepted, s.qwen35HAL.decodeAccepted)
			}
			seeded := 0
			for layer, owner := range s.qwen35HAL.sequenceLayers {
				if !owner.valid() {
					continue
				}
				st := &before.linear.layers[layer]
				if wantRows := min(n, keep); len(st.conv) != wantRows {
					t.Fatalf("layer %d host conv rows=%d, want %d", layer, len(st.conv), wantRows)
				}
				wantConv, wantRec := oracleQwen35GDNSeedLayout(t, s.M.Cfg, st)
				nonzero := false
				for _, v := range wantConv {
					nonzero = nonzero || v != 0
				}
				if !nonzero {
					t.Fatalf("layer %d oracle conv is all zero; the test would not distinguish row order", layer)
				}
				got := b.state[owner]
				if !reflect.DeepEqual(got.conv, wantConv) || !reflect.DeepEqual(got.recurrent, wantRec) {
					t.Fatalf("layer %d owner seed does not match the oracle layout", layer)
				}
				seeded++
			}
			if seeded != linearQwen35Layers(s.M.Cfg) {
				t.Fatalf("seeded %d owners, want %d", seeded, linearQwen35Layers(s.M.Cfg))
			}
			assertHostCacheUnchanged(t, "restored admission host state", s.M.Cfg, before, s.Cache)
		})
	}
}

// Enable must refuse a session whose resident GDN state is not reseedable: a
// recorded failure (and its audit record) or live promoted decode owners.
func TestQwen35GDNEnableRefusesFailedOrPromotedSession(t *testing.T) {
	t.Run("recorded failure", func(t *testing.T) {
		b := newStatefulGDNBackend()
		installStatefulGDNFactory(t, b)
		_, s := restoredHybridSession(t, 5)
		defer s.Close()
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		failure := s.failQwen35GDNSequence(0, "injected", errors.New("boom"))
		allocs := b.allocCalls
		err := s.EnableQwen35MetalGDNPreprojectedSequence()
		var unsupported *UnsupportedGDNPreprojectedSequenceError
		if !errors.As(err, &unsupported) {
			t.Fatalf("re-admission after failure err=%v, want UnsupportedGDNPreprojectedSequenceError", err)
		}
		if s.qwen35HAL.sequenceFailure != failure {
			t.Fatalf("refusal cleared the failure record: %v", s.qwen35HAL.sequenceFailure)
		}
		if b.allocCalls != allocs {
			t.Fatalf("refusal allocated owners: %d -> %d", allocs, b.allocCalls)
		}
		if snap, err := s.PrefixSnapshot(); err == nil {
			snap.Close()
			t.Fatal("snapshot after recorded failure did not fail closed")
		}
	})
	t.Run("promoted decode", func(t *testing.T) {
		b := newStatefulGDNBackend()
		installStatefulGDNFactory(t, b)
		_, s := restoredHybridSession(t, 5)
		defer s.Close()
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		if _, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		owners := append([]Qwen35GDNAuxState(nil), s.qwen35HAL.sequenceLayers...)
		allocs := b.allocCalls
		err := s.EnableQwen35MetalGDNPreprojectedSequence()
		var unsupported *UnsupportedGDNPreprojectedSequenceError
		if !errors.As(err, &unsupported) {
			t.Fatalf("re-admission over promoted decode err=%v, want UnsupportedGDNPreprojectedSequenceError", err)
		}
		if _, selected := s.Qwen35GDNDecodePath(); !selected || !reflect.DeepEqual(s.qwen35HAL.sequenceLayers, owners) || b.allocCalls != allocs {
			t.Fatal("refusal disturbed the promoted decode owners")
		}
	})
}

func TestQwen35GDNEnableRestoredPrefixDeclinesTyped(t *testing.T) {
	t.Run("incomplete host state", func(t *testing.T) {
		b := newStatefulGDNBackend()
		installStatefulGDNFactory(t, b)
		_, s := restoredHybridSession(t, 5)
		defer s.Close()
		s.Cache.linear.layers[0].recurrent = s.Cache.linear.layers[0].recurrent[:1]
		err := s.EnableQwen35MetalGDNPreprojectedSequence()
		var unsupported *UnsupportedGDNPreprojectedSequenceError
		if !errors.As(err, &unsupported) {
			t.Fatalf("incomplete restored state err=%v, want UnsupportedGDNPreprojectedSequenceError", err)
		}
		if s.qwen35HAL != nil || b.allocCalls != 0 {
			t.Fatalf("incomplete restored state allocated owners: hal=%#v allocs=%d", s.qwen35HAL, b.allocCalls)
		}
	})
	t.Run("seed failure", func(t *testing.T) {
		b := newStatefulGDNBackend()
		b.failSeed = true
		installStatefulGDNFactory(t, b)
		_, s := restoredHybridSession(t, 5)
		defer s.Close()
		before := s.Cache.Clone()
		err := s.EnableQwen35MetalGDNPreprojectedSequence()
		var unsupported *UnsupportedGDNPreprojectedSequenceError
		if !errors.As(err, &unsupported) {
			t.Fatalf("seed failure err=%v, want UnsupportedGDNPreprojectedSequenceError", err)
		}
		if s.qwen35HAL.sequenceAccepted || s.qwen35HAL.decodeAccepted || s.qwen35HAL.sequenceBackend != nil {
			t.Fatalf("seed failure left owners admitted: %#v", s.qwen35HAL)
		}
		assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, b.allocated)
		assertHostCacheUnchanged(t, "seed failure host state", s.M.Cfg, before, s.Cache)
	})
}
