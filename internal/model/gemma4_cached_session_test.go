package model

import (
	"context"
	"errors"
	"math"
	"runtime"
	"testing"
)

func gemma4CachedBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("logits length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
			t.Fatalf("non-finite logit %d", i)
		}
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("logit %d: %08x, want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func gemma4CachedPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("selected call did not refuse")
		}
	}()
	f()
}

func gemma4CachedSession(t *testing.T, m *Model, maxTokens int) *Session {
	t.Helper()
	s := m.NewSession()
	t.Cleanup(s.Close)
	if err := s.ConfigureGemma4CachedContinuation(Gemma4CachedContinuationOptions{MaxTokens: maxTokens, MaxStateBytes: 1 << 20}); err != nil {
		t.Fatal(err)
	}
	return s
}

// fak-test:runtime fast est=2s lane=default
func TestAttentionStateRows(t *testing.T) {
	specs := []attentionRowSpec{{KVHeads: 1, HeadDim: 2, Window: -1}, {KVHeads: 1, HeadDim: 3, Window: 3}}
	const wantBytes = 8*8 + 8*2*8 + 3*3*8
	if _, err := newAttentionStateRows(specs, 8, wantBytes-1); err == nil {
		t.Fatal("budget not enforced")
	}
	for _, tc := range []struct {
		spec   attentionRowSpec
		tokens int
	}{
		{attentionRowSpec{KVHeads: 1, HeadDim: 2, Window: 0}, 8},
		{attentionRowSpec{KVHeads: 1, HeadDim: 2, Window: -1}, 0},
		{attentionRowSpec{KVHeads: int(^uint(0) >> 1), HeadDim: 2, Window: -1}, 8},
		{attentionRowSpec{KVHeads: 1, HeadDim: 2, Window: -1}, int(^uint(0) >> 1)},
	} {
		if _, err := newAttentionStateRows([]attentionRowSpec{tc.spec}, tc.tokens, math.MaxInt64); err == nil {
			t.Fatal("invalid reservation admitted")
		}
	}
	s, err := newAttentionStateRows(specs, 8, wantBytes)
	if err != nil {
		t.Fatal(err)
	}
	if s.payloadBytes != wantBytes || cap(s.history) != 8 {
		t.Fatal("reservation charge/capacity mismatch")
	}
	for pos := 0; pos < 8; pos++ {
		pending := make([]attentionPendingRow, 2)
		for l, spec := range specs {
			pending[l] = attentionPendingRow{k: make([]float32, spec.HeadDim), v: make([]float32, spec.HeadDim)}
			for j := range pending[l].k {
				pending[l].k[j], pending[l].v[j] = float32(pos*10+j), float32(-pos*10-j)
			}
		}
		s.publish(pos, pending)
		// Publishing copies, never retains the current projection buffers.
		pending[0].k[0] = -999
		for l, spec := range specs {
			start := 0
			if spec.Window > 0 && pos+1 > spec.Window {
				start = pos + 1 - spec.Window
			}
			for old := start; old <= pos; old++ {
				k, v := s.row(l, old)
				for j := range k {
					if k[j] != float32(old*10+j) || v[j] != float32(-old*10-j) {
						t.Fatalf("layer%d absolute row%d after%d", l, old, pos)
					}
				}
			}
		}
	}
	gemma4CachedPanic(t, func() { s.row(1, 4) })
	gemma4CachedPanic(t, func() { s.publish(8, nil) })
	clone := s.clone()
	clone.layers[1].k[0]++
	clone.history[0] = -1
	if clone.layers[1].k[0] == s.layers[1].k[0] || s.history[0] != 0 {
		t.Fatal("clone aliases state")
	}
	clone.close()
	if clone.payloadBytes != 0 || s.payloadBytes != wantBytes {
		t.Fatal("close crossed owners")
	}
	partial, _ := newAttentionStateRows(specs, 8, wantBytes)
	gemma4CachedPanic(t, func() { partial.publish(1, []attentionPendingRow{{k: []float32{7, 8}, v: []float32{9, 10}}, {}}) })
	if len(partial.history) != 0 || partial.layers[0].k[0] != 0 {
		t.Fatal("invalid later layer partially published")
	}
}

// fak-test:runtime fast est=5s lane=default
func TestGemma4CachedContinuation(t *testing.T) {
	ids := []int{3, 9, 14, 27, 6, 31, 2, 18, 5}
	t.Run("same_math_and_bounded_wrap", func(t *testing.T) {
		m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
		base := m.NewSession()
		defer base.Close()
		s := gemma4CachedSession(t, m, 12)
		if base.Gemma4CachedContinuationObservation().Selected {
			t.Fatal("default route changed")
		}
		if m.has(layerName(3, "self_attn.v_proj.weight")) {
			t.Fatal("fixture lost raw-K-as-V global layer")
		}
		var got []float32
		for n := 1; n <= len(ids); n++ {
			if n == 1 {
				got = s.Prefill(ids[:1])
				gemma4CachedBits(t, got, base.Prefill(ids[:1]))
			} else {
				got = s.Step(ids[n-1])
				gemma4CachedBits(t, got, base.Step(ids[n-1]))
			}
			ref := m.Forward(ids[:n])
			gemma4CachedBits(t, got, ref.Logits[n-1])
			ob := s.Gemma4CachedContinuationObservation()
			if ob.Tokens != n || ob.ComputedTokens != uint64(n) || ob.Failed || ob.Closed {
				t.Fatalf("bad actual observation: %+v", ob)
			}
			if s.Cache.Len() != 0 || len(s.gemma4Hist) != 0 {
				t.Fatal("selected route fell through to uniform/recompute state")
			}
			for l, layer := range s.gemma4Cached.rows.layers {
				want := 2
				if l == 3 {
					want = 12
				}
				if layer.capacity != want || layer.stride != 16 {
					t.Fatal("heterogeneous reservation changed")
				}
			}
		}
		panel := gemma4CachedSession(t, m, 12)
		gemma4CachedBits(t, panel.Prefill(ids[:5]), m.Forward(ids[:5]).Logits[4])
		gemma4CachedBits(t, panel.Prefill(ids[5:]), m.Forward(ids).Logits[len(ids)-1])
		if panel.Gemma4CachedContinuationObservation().ComputedTokens != uint64(len(ids)) {
			t.Fatal("continuation recomputed old tokens")
		}
		wantBytes := int64(12*8 + (3*2*16+12*16)*8)
		if s.Gemma4CachedContinuationObservation().StateBytes != wantBytes {
			t.Fatal("payload does not charge all reservations")
		}
		if _, err := base.PrefixSnapshot(); !errors.Is(err, ErrGemma4CachedContinuation) {
			t.Fatalf("recompute snapshot silently omitted history: %v", err)
		}
		if m.Cfg.KVPrefixReuseSupported() || m.Cfg.HostCompletePrefixSnapshotSupported() {
			t.Fatal("unqualified bare/planner admission widened")
		}
	})
	t.Run("no_logits_clone_restore_and_retirement", func(t *testing.T) {
		m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
		s := gemma4CachedSession(t, m, 12)
		s.PrefillNoLogits(ids[:5])
		if s.Gemma4CachedContinuationObservation().ComputedTokens != 5 {
			t.Fatal("no-logits route did not actually compute state")
		}
		p, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		if p.Tokens != 5 || p.HostDiskSerializable() {
			t.Fatal("incomplete snapshot count/disk admission")
		}
		if _, err := p.MarshalCPU(); err == nil {
			t.Fatal("disk codec silently omitted ring state")
		}
		host, device := p.ResidencyBytes()
		if host != p.ResidentBytes() || device != 0 || host < s.Gemma4CachedContinuationObservation().StateBytes {
			t.Fatal("complete state not charged")
		}
		q, err := p.CloneForRestore()
		if err != nil {
			t.Fatal(err)
		}
		defer q.Close()
		if &p.gemma4.rows.layers[0].k[0] == &q.gemma4.rows.layers[0].k[0] ||
			&p.gemma4.rows.history[0] == &q.gemma4.rows.history[0] {
			t.Fatal("snapshot clone aliases mutable state")
		}
		wrong := gemma4CachedSession(t, NewSyntheticGemma4(tinyGemma4ReuseCfg()), 12)
		if err := q.Restore(wrong); !errors.Is(err, ErrGemma4CachedContinuation) || q.Cache == nil || q.gemma4 == nil {
			t.Fatal("wrong-model restore consumed snapshot")
		}
		unselected := m.NewSession()
		defer unselected.Close()
		if err := q.Restore(unselected); !errors.Is(err, ErrGemma4CachedContinuation) {
			t.Fatal("restore silently selected route")
		}
		badProfile := gemma4CachedSession(t, m, 11)
		if err := q.Restore(badProfile); !errors.Is(err, ErrGemma4CachedContinuation) {
			t.Fatal("restore changed reservation profile")
		}
		s.Close() // snapshots must outlive their original session's row buffers
		if s.Gemma4CachedContinuationObservation().StateBytes != 0 {
			t.Fatal("session close retained payload")
		}
		target := gemma4CachedSession(t, m, 12)
		if err := q.Restore(target); err != nil {
			t.Fatal(err)
		}
		if q.Cache != nil || q.gemma4 != nil || target.Gemma4CachedContinuationObservation().ComputedTokens != 0 {
			t.Fatal("transfer/counter provenance mismatch")
		}
		want := m.Forward(ids[:6]).Logits[5]
		gemma4CachedBits(t, target.Step(ids[5]), want)
		// The other independent snapshot can roll back a healthy diverged branch.
		if err := p.Restore(target); err != nil {
			t.Fatal(err)
		}
		gemma4CachedBits(t, target.Step(ids[5]), want)
		target.Close()
		gemma4CachedPanic(t, func() { target.Step(ids[6]) })
		if _, err := target.PrefixSnapshot(); err == nil {
			t.Fatal("closed owner snapshotted")
		}
	})
	t.Run("refusals_before_mutation_and_fault_poison", func(t *testing.T) {
		m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
		s := gemma4CachedSession(t, m, 2)
		s.Prefill(ids[:1])
		for _, bad := range [][]int{{ids[1], ids[2]}, {-1}, {m.Cfg.VocabSize}} {
			gemma4CachedPanic(t, func() { s.Prefill(bad) })
			ob := s.Gemma4CachedContinuationObservation()
			if ob.Tokens != 1 || ob.Failed {
				t.Fatal("input refusal changed state")
			}
		}
		gemma4CachedBits(t, s.Step(ids[1]), m.Forward(ids[:2]).Logits[1])
		s2 := m.NewSession()
		defer s2.Close()
		if err := s2.ConfigureGemma4CachedContinuation(Gemma4CachedContinuationOptions{MaxTokens: 12, MaxStateBytes: 1}); err == nil || s2.gemma4Cached != nil {
			t.Fatal("budget refusal published owner")
		}
		s2.Metal = true
		if err := s2.ConfigureGemma4CachedContinuation(Gemma4CachedContinuationOptions{MaxTokens: 12, MaxStateBytes: 1 << 20}); err == nil {
			t.Fatal("unwired mode selected")
		}
		s2.Metal = false
		s2.Close()
		if err := s2.ConfigureGemma4CachedContinuation(Gemma4CachedContinuationOptions{MaxTokens: 12, MaxStateBytes: 1 << 20}); err == nil {
			t.Fatal("closed session reopened")
		}
		broken := NewSyntheticGemma4(tinyGemma4ReuseCfg())
		b := gemma4CachedSession(t, broken, 12)
		// A real missing projection fails after embedding, inside the selected
		// body; no test-only production hook or recovery fallback is involved.
		name := layerName(1, "self_attn.q_proj.weight")
		old := broken.manifest[name]
		delete(broken.manifest, name)
		gemma4CachedPanic(t, func() { b.Prefill(ids[:1]) })
		broken.manifest[name] = old
		if ob := b.Gemma4CachedContinuationObservation(); !ob.Failed || ob.Tokens != 0 || ob.ComputedTokens != 0 {
			t.Fatal("partial layer state escaped failed token")
		}
		for _, layer := range b.gemma4Cached.rows.layers {
			for _, value := range layer.k {
				if value != 0 {
					t.Fatal("failed token overwrote cache")
				}
			}
		}
		gemma4CachedPanic(t, func() { b.Step(ids[0]) })
		if _, err := b.PrefixSnapshot(); err == nil {
			t.Fatal("poisoned snapshot admitted")
		}
	})
	t.Run("cancel_and_goexit_close_partial_owner", func(t *testing.T) {
		for _, goexit := range []bool{false, true} {
			m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
			s := gemma4CachedSession(t, m, 12)
			ctx, cancel := context.WithCancel(context.Background())
			probe := &gemma4CachedCancelContext{Context: ctx, cancel: cancel, session: s, goexit: goexit}
			if goexit {
				done := make(chan struct{})
				go func() { defer close(done); s.PrefillNoLogitsContext(probe, ids[:3]) }()
				<-done
			} else if _, err := s.PrefillContext(probe, ids[:3]); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel: %v", err)
			}
			cancel()
			if ob := s.Gemma4CachedContinuationObservation(); ob.Tokens != 1 || !ob.Failed || ob.ComputedTokens != 1 {
				t.Fatalf("partial-token boundary: %+v", ob)
			}
			gemma4CachedPanic(t, func() { s.Step(ids[1]) })
			if _, err := s.PrefixSnapshot(); err == nil {
				t.Fatal("cancelled state reused")
			}
			s.Close()
			if s.Gemma4CachedContinuationObservation().StateBytes != 0 {
				t.Fatal("close retained cancelled rows")
			}
		}
	})
	t.Run("geometry_epoch_and_mode_identity", func(t *testing.T) {
		m := NewSyntheticGemma4(tinyGemma4ReuseCfg())
		s := gemma4CachedSession(t, m, 12)
		s.PrefillNoLogits(ids[:2])
		p, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		s.cacheGeometryEpoch++
		gemma4CachedPanic(t, func() { s.Step(ids[2]) })
		target := gemma4CachedSession(t, m, 12)
		if err := p.Restore(target); err == nil || p.Cache == nil {
			t.Fatal("stale source epoch restored")
		}
		old := m.Cfg.HeadDimPerLayer[0]
		m.Cfg.HeadDimPerLayer[0]++
		gemma4CachedPanic(t, func() { target.Step(ids[0]) })
		m.Cfg.HeadDimPerLayer[0] = old
		target.Metal = true
		gemma4CachedPanic(t, func() { target.Step(ids[0]) })
	})
}

type gemma4CachedCancelContext struct {
	context.Context
	cancel  context.CancelFunc
	session *Session
	goexit  bool
}

func (c *gemma4CachedCancelContext) Err() error {
	if c.session.Gemma4CachedContinuationObservation().Tokens == 1 {
		if c.goexit {
			runtime.Goexit()
		}
		c.cancel()
	}
	return c.Context.Err()
}
