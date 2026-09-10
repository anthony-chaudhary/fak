package radixkv

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestAuditCoalesceLeaderOwnsIndependentKV(t *testing.T) {
	tree := New(0)
	group := NewPrefixFlightGroup(tree)
	prefix := []int{1, 2, 3, 4}
	m := model.NewSynthetic(model.Config{
		ModelType: "llama", VocabSize: 16, HiddenSize: 16, IntermediateSize: 32,
		NumLayers: 1, NumHeads: 2, NumKVHeads: 2, HeadDim: 8,
		RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1,
	})

	got, _, leader, err := group.Coalesce(context.Background(), prefix, func(context.Context) (*model.KVCache, []float32, error) {
		sess := m.NewSession()
		logits := sess.Prefill(prefix)
		return sess.Cache, logits, nil
	})
	if err != nil || !leader {
		t.Fatalf("first Coalesce: leader=%v err=%v", leader, err)
	}
	if got == nil || got.Len() != len(prefix) {
		t.Fatalf("returned KV len=%v, want %d", got, len(prefix))
	}
	got.Truncate(0)
	n, matched := tree.Lookup(prefix)
	defer tree.Done(n)
	if matched != len(prefix) || n == nil {
		t.Fatalf("resident prefix lookup matched=%d node=%v", matched, n)
	}
	if n.KV().Len() != len(prefix) {
		t.Fatalf("request-local mutation changed resident KV len to %d; want %d", n.KV().Len(), len(prefix))
	}
}
