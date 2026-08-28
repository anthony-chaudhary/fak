package engine_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type restoreMemStore struct{ data map[string][]byte }

func (s *restoreMemStore) Put(_ context.Context, key string, payload []byte) error {
	if s.data == nil {
		s.data = map[string][]byte{}
	}
	s.data[key] = append([]byte(nil), payload...)
	return nil
}
func (s *restoreMemStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	b, ok := s.data[key]
	return append([]byte(nil), b...), ok, nil
}

type countingRestoreBackend struct {
	abi.KVBackend
	prefills int
}

func (b *countingRestoreBackend) Prefill(ids []int) []float32 {
	b.prefills++
	return b.KVBackend.Prefill(ids)
}
func (b *countingRestoreBackend) StageSpanBytes(from, n int) ([]byte, error) {
	return b.KVBackend.(interface {
		StageSpanBytes(int, int) ([]byte, error)
	}).StageSpanBytes(from, n)
}
func (b *countingRestoreBackend) RestoreSpanBytes(payload []byte) (int, error) {
	return b.KVBackend.(interface{ RestoreSpanBytes([]byte) (int, error) }).RestoreSpanBytes(payload)
}

func restoreTestConfig() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
}

func TestRestoreAdapterStageEvictDemandRestoresWithoutRecompute(t *testing.T) {
	ctx := context.Background()
	m := model.NewSynthetic(restoreTestConfig())
	control, victim := m.NewSession(), m.NewSession()
	ids := []int{3, 17, 5, 23, 41, 2, 19}
	control.Prefill(ids)
	victim.Prefill(ids)
	inner, ok := model.KVBackend(victim)
	if !ok {
		t.Fatal("model KV backend unavailable")
	}
	counted := &countingRestoreBackend{KVBackend: inner}
	store := &restoreMemStore{}
	tier := l3kv.New(counted, store)
	const digest = "span-middle"
	staged, err := tier.StageSpan(ctx, digest, 2, 3)
	if err != nil || staged.Outcome != abi.KVResidencyOK || staged.BytesMoved <= 0 {
		t.Fatalf("stage=%+v err=%v", staged, err)
	}
	if removed := tier.Evict(2, 3); removed != 3 {
		t.Fatalf("removed=%d", removed)
	}
	adapter := &engine.RestoreAdapter{KV: tier, Recorder: engine.NewCacheEventRecorder()}
	result, err := adapter.Restore(ctx, engine.RestoreMove{
		SpanDigest: digest, ModelID: "synthetic", FromTier: cachemeta.TierRemote, ToTier: cachemeta.TierHBM,
	})
	if err != nil || !result.Restored || !result.Verdict.CanServe() || result.Residency.Positions != 3 || result.Residency.BytesMoved != staged.BytesMoved {
		t.Fatalf("restore=%+v err=%v", result, err)
	}
	if counted.prefills != 0 {
		t.Fatalf("restore called Prefill %d times, want zero", counted.prefills)
	}
	if victim.Cache.Len() != control.Cache.Len() || !reflect.DeepEqual(victim.Cache.K, control.Cache.K) ||
		!reflect.DeepEqual(victim.Cache.Kraw, control.Cache.Kraw) || !reflect.DeepEqual(victim.Cache.V, control.Cache.V) {
		t.Fatal("restored live cache differs from uninterrupted cache")
	}
	for _, token := range []int{29, 31, 37} {
		if got, want := victim.Step(token), control.Step(token); !reflect.DeepEqual(got, want) {
			t.Fatalf("continuation logits differ for token %d", token)
		}
	}

	miss, err := adapter.Restore(ctx, engine.RestoreMove{SpanDigest: "never-staged"})
	if err != nil || miss.Restored || miss.Verdict.Reason != cachemeta.ReasonRestoreMiss || counted.prefills != 0 {
		t.Fatalf("never-staged restore=%+v err=%v prefills=%d", miss, err, counted.prefills)
	}
}

func TestRestoreAdapterCorruptPayloadIsTypedFaultAndNoPublication(t *testing.T) {
	ctx := context.Background()
	m := model.NewSynthetic(restoreTestConfig())
	victim := m.NewSession()
	victim.Prefill([]int{3, 17, 5, 23, 41, 2, 19})
	inner, _ := model.KVBackend(victim)
	counted := &countingRestoreBackend{KVBackend: inner}
	store := &restoreMemStore{}
	tier := l3kv.New(counted, store)
	const digest = "corrupt-middle"
	if staged, _ := tier.StageSpan(ctx, digest, 2, 3); staged.Outcome != abi.KVResidencyOK {
		t.Fatalf("stage=%+v", staged)
	}
	tier.Evict(2, 3)
	beforeKraw := cloneRows(storeTestRows(victim.Cache.Kraw))
	store.data[digest][len(store.data[digest])/2] ^= 1
	result, err := (&engine.RestoreAdapter{KV: tier}).Restore(ctx, engine.RestoreMove{SpanDigest: digest})
	if err != nil || result.Restored || result.Verdict.Reason != cachemeta.ReasonResidencyFault || result.Residency.BytesMoved != 0 {
		t.Fatalf("corrupt restore=%+v err=%v", result, err)
	}
	if counted.prefills != 0 || !reflect.DeepEqual(victim.Cache.Kraw, beforeKraw) {
		t.Fatalf("corrupt restore prefills=%d or mutated cache", counted.prefills)
	}
}

func storeTestRows(rows [][]float32) [][]float32 { return rows }
func cloneRows(rows [][]float32) [][]float32 {
	out := make([][]float32, len(rows))
	for i := range rows {
		out[i] = append([]float32(nil), rows[i]...)
	}
	return out
}
