package benchcli

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestDecodeLCGMatchesInlineLoop(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 101, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)
	gotSession := m.NewSession()
	wantSession := m.NewSession()
	prefix := []int{3, 14, 15}
	gotSession.Prefill(prefix)
	wantSession.Prefill(prefix)

	const (
		start = 92
		steps = 7
	)
	gotToken := DecodeLCG(gotSession, start, steps, cfg.VocabSize)
	wantToken := start
	for i := 0; i < steps; i++ {
		wantSession.Step(wantToken)
		wantToken = (wantToken*48271 + 1) % cfg.VocabSize
	}

	if gotToken != wantToken {
		t.Fatalf("next token = %d, want %d", gotToken, wantToken)
	}
	if got, want := gotSession.Cache.Len(), wantSession.Cache.Len(); got != want {
		t.Fatalf("cache len = %d, want %d", got, want)
	}
	if got, want := gotSession.Step(gotToken), wantSession.Step(wantToken); !reflect.DeepEqual(got, want) {
		t.Fatal("helper left the session in a different decode state")
	}
}
