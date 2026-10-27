package radixkv

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=5s lane=default
func TestSharedPrefixReadersCopyOnlyDensePrefix(t *testing.T) {
	for _, precision := range []model.KVPrecision{model.KVPrecisionFP32, model.KVPrecisionQ8_0} {
		t.Run(string(precision), func(t *testing.T) {
			cfg := model.Config{HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 4,
				HeadDim: 8, IntermediateSize: 64, VocabSize: 64, RMSNormEps: 1e-5,
				RopeTheta: 10000, EOSTokenID: 63}
			m := model.NewSynthetic(cfg)
			leader := m.NewSession()
			defer leader.Close()
			leader.Cache = model.NewKVCacheWithPrecision(cfg, precision)
			tokens := sharedPrefixTestIDs(32, 17)
			leader.Prefill(tokens)
			published := leader.Cache.Clone()
			before := published.Clone()
			f := &flightPromise{kv: published, tokens: tokens, waiters: map[int]context.Context{
				1: context.Background(), 2: context.Background(), 3: context.Background(),
			}}
			g := NewPrefixFlightGroup(nil)
			var copied int64
			for i, keep := range []int{8, 16} {
				want := published.Clone()
				if n, err := want.TryEvict(keep, len(tokens)-keep); err != nil || n != len(tokens)-keep {
					t.Fatalf("oracle truncate: removed=%d err=%v", n, err)
				}
				got, logits, matched, retry, err := g.sharedPrefixFollowerResult(context.Background(), f, i+1, keep, keep+1, false)
				if err != nil || retry || matched != keep || len(logits) != 0 || !reflect.DeepEqual(got, want) {
					t.Fatalf("keep=%d: matched=%d retry=%v err=%v; prefix differs from original clone/evict", keep, matched, retry, err)
				}
				copied += got.ClonePayloadBytes()
				if g.ClonePayloadBytes() != copied || got.ClonePayloadBytes() >= published.ClonePayloadBytes() {
					t.Fatal("clone accounting retained discarded leader suffix")
				}
				if f.kv != published || len(f.waiters) != 2-i || !reflect.DeepEqual(published, before) {
					t.Fatal("concurrent follower detached or changed the publication")
				}
				got.Kraw[0][0]++
				if !reflect.DeepEqual(published, before) || leader.Cache.Kraw[0][0] != before.Kraw[0][0] {
					t.Fatal("prefix clone aliases the publication or live leader")
				}
			}
			last, _, matched, retry, err := g.sharedPrefixFollowerResult(context.Background(), f, 3, 24, 25, false)
			if err != nil || retry || matched != 24 || last != published || f.kv != nil || len(f.waiters) != 0 || g.ClonePayloadBytes() != copied {
				t.Fatal("sole-reader transfer or no-copy accounting changed")
			}
		})
	}
}
