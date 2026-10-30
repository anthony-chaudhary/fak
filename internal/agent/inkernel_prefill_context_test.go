package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type partialPrefillKV struct {
	compute.KVStore
	cfg       compute.KVConfig
	completed int
	frees     int
}

func (k *partialPrefillKV) KVConfig() compute.KVConfig { return k.cfg }
func (k *partialPrefillKV) Free() {
	k.frees++
	k.KVStore.Free()
}

// The arithmetic is the real CPU HAL. Only token completion, KV retirement and
// the existing execution-owner capability are observed; no device path is faked.
type partialPrefillBackend struct {
	compute.Backend
	owner          sync.Mutex
	stores         []*partialPrefillKV
	cancel         context.CancelFunc
	cancelled      bool
	postCancelWork int
	unowned        bool
}

func (b *partialPrefillBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *partialPrefillBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	k := &partialPrefillKV{KVStore: b.Backend.NewKV(cfg), cfg: cfg}
	b.stores = append(b.stores, k)
	return k
}
func (b *partialPrefillBackend) Recycle() {
	if len(b.stores) == 0 {
		return
	}
	k := b.stores[len(b.stores)-1]
	if k.frees != 0 || k.Len() <= k.completed {
		return // Constructor/cleanup recycling is not completed token work.
	}
	if b.owner.TryLock() {
		b.unowned = true
		b.owner.Unlock()
	}
	if b.cancelled {
		b.postCancelWork += k.Len() - k.completed
	}
	k.completed = k.Len()
	if b.cancel != nil && !b.cancelled {
		b.cancelled = true
		b.cancel()
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelPartialOffloadPrefillCancellation(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	ids := synthIDs(cfg.VocabSize, 8, 311)
	newFixture := func() (*model.Model, *partialPrefillBackend) {
		m := model.NewSynthetic(cfg)
		t.Cleanup(func() { _ = m.CloseWeights() })
		return m, &partialPrefillBackend{Backend: compute.Default()}
	}
	planner := func(m *model.Model, b *partialPrefillBackend) *InKernelPlanner {
		return &InKernelPlanner{m: m, modelID: "partial-prefill-cpu-witness", backend: b, denseGPULayers: 1, tree: radixkv.New(0)}
	}
	session := func(t *testing.T, m *model.Model, b *partialPrefillBackend) *model.Session {
		t.Helper()
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		s.DenseGPULayers = 1
		if s.Backend != b || s.DenseGPULayers >= m.Cfg.NumLayers {
			t.Fatal("fixture must execute the partial-placement HAL token loop")
		}
		t.Cleanup(s.Close)
		return s
	}
	assertReleased := func(t *testing.T, b *partialPrefillBackend) {
		t.Helper()
		if !b.owner.TryLock() {
			t.Fatal("request retained the shared backend execution owner")
		}
		b.owner.Unlock()
		if b.unowned {
			t.Error("token completed outside its backend execution owner")
		}
		for _, k := range b.stores {
			if k.frees != 1 {
				t.Errorf("session KV retired %d times, want exactly once", k.frees)
			}
		}
	}
	generate := func(p *InKernelPlanner, ctx context.Context) ([]int, error) {
		var out []int
		_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctx, ids, 2, 0, 0, 0, nil, 0, 0, nil, func(id int) bool {
			out = append(out, id)
			return false
		})
		return out, err
	}

	t.Run("planner cancellation cleanup and shared backend reuse", func(t *testing.T) {
		m, b := newFixture()
		want, err := generate(planner(m, b), context.Background())
		if err != nil || len(want) != 2 {
			t.Fatalf("uncancelled control output=%v err=%v", want, err)
		}
		assertReleased(t, b)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		b.cancel = cancel
		p := planner(m, b)
		got, err := generate(p, ctx)
		if !errors.Is(err, context.Canceled) || len(got) != 0 || !b.cancelled {
			t.Fatalf("cancelled result=%v err=%v cancellation_observed=%t", got, err, b.cancelled)
		}
		t.Logf("prompt_tokens=%d cancellation_after_completed_token=1 post_cancel_tokens=%d", len(ids), b.postCancelWork)
		if b.postCancelWork != 0 {
			t.Errorf("completed %d additional tokens after cancellation, want zero", b.postCancelWork)
		}
		if got := p.cachedPrefixLen(ids); got != 0 {
			t.Errorf("cancelled incomplete prefix admitted %d tokens", got)
		}
		assertReleased(t, b)
		b.cancel, b.cancelled = nil, false
		got, err = generate(p, context.Background())
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("same-backend next request=%v err=%v, want %v", got, err, want)
		}
		assertReleased(t, b)
	})

	t.Run("uncancelled logits and cache match legacy", func(t *testing.T) {
		m, b := newFixture()
		legacy := session(t, m, b)
		want := append([]float32(nil), legacy.Prefill(ids)...)
		wantLen := legacy.Cache.Len()
		legacy.Close()
		s := session(t, m, b)
		got, err := planner(m, b).prefillDivergentSuffix(context.Background(), s, ids)
		if err != nil || !reflect.DeepEqual(got, want) || s.Cache.Len() != wantLen || wantLen != len(ids) {
			t.Errorf("uncancelled parity: err=%v logits_equal=%t cache=%d legacy=%d", err, reflect.DeepEqual(got, want), s.Cache.Len(), wantLen)
		}
		s.Close()
		assertReleased(t, b)
	})

	t.Run("no logits uses the same token boundary", func(t *testing.T) {
		m, b := newFixture()
		s := session(t, m, b)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		b.cancel = cancel
		var err error
		// The legacy fallback keeps this witness compilable on the parent. Both
		// arms execute a real Session; missing cancellation fails on token work.
		if contextual, ok := any(s).(interface {
			PrefillNoLogitsContext(context.Context, []int) error
		}); ok {
			err = contextual.PrefillNoLogitsContext(ctx, ids)
		} else {
			s.PrefillNoLogits(ids)
			err = ctx.Err()
		}
		t.Logf("no_logits_post_cancel_tokens=%d cache_tokens=%d", b.postCancelWork, s.Cache.Len())
		if !errors.Is(err, context.Canceled) || !b.cancelled || b.postCancelWork != 0 || s.Cache.Len() != 1 {
			t.Errorf("no-logits cancellation: err=%v observed=%t extra=%d cache=%d", err, b.cancelled, b.postCancelWork, s.Cache.Len())
		}
		s.Close()
		assertReleased(t, b)
	})
}
