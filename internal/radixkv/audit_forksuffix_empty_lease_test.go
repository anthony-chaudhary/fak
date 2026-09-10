package radixkv

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestAuditForkSuffixEmptyReturnsLeasedNode(t *testing.T) {
	tree := New(0)
	group := NewPrefixFlightGroup(tree)
	prefix := []int{1, 2, 3, 4}
	m := model.NewSynthetic(model.Config{
		ModelType: "llama", VocabSize: 16, HiddenSize: 16, IntermediateSize: 32,
		NumLayers: 1, NumHeads: 2, NumKVHeads: 2, HeadDim: 8,
		RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1,
	})

	n, leader, err := group.ForkSuffix(
		context.Background(), "", prefix, nil,
		func(context.Context) (*model.KVCache, []float32, error) {
			sess := m.NewSession()
			logits := sess.Prefill(prefix)
			return sess.Cache, logits, nil
		},
		func(context.Context, *node) (*model.KVCache, []float32, error) {
			t.Fatal("suffix callback called for empty suffix")
			return nil, nil, nil
		},
	)
	if err != nil || !leader || n == nil {
		t.Fatalf("ForkSuffix: node=%v leader=%v err=%v", n, leader, err)
	}
	if n.refs != 1 {
		t.Fatalf("returned node refs=%d, want 1 active caller lease", n.refs)
	}
	group.Done(n)
}
