package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

func fixture(t *testing.T) Evidence {
	t.Helper()
	e, err := Load("testdata/complete.json")
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestAll16ExactlyOnceAndDeterministic(t *testing.T) {
	e := fixture(t)
	if len(e.Dimensions) != 16 {
		t.Fatalf("got %d", len(e.Dimensions))
	}
	a := Score(e)
	b := Score(e)
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if !bytes.Equal(aj, bj) {
		t.Fatal("nondeterministic output")
	}
}
func TestUnknownDebtAndDominantBottleneck(t *testing.T) {
	e := fixture(t)
	e.Dimensions[0].Current = nil
	e.Dimensions[1].Current = nil
	r := Score(e)
	if r.UnknownDebt != 2 || r.DominantBottleneck != "evaluation_latency" {
		t.Fatalf("debt=%d bottleneck=%s", r.UnknownDebt, r.DominantBottleneck)
	}
}
func TestBottleneckSelectionUsesLowestRatio(t *testing.T) {
	r := Score(fixture(t))
	if r.DominantBottleneck != "evaluation_latency" {
		t.Fatal(r.DominantBottleneck)
	}
}
func TestUnsaturatedRatioAnd100x(t *testing.T) {
	e := fixture(t)
	v := 250.0
	e.Dimensions[1].Current = &v
	r := Score(e)
	if *r.Dimensions[1].NormalizedRatio != 2.5 || r.TargetMultiplier != 100 {
		t.Fatalf("%+v", r)
	}
}
func TestInvalidValues(t *testing.T) {
	for _, v := range []float64{-1, math.NaN(), math.Inf(1)} {
		e := fixture(t)
		e.Dimensions[0].Current = &v
		b, _ := json.Marshal(e)
		if _, err := Decode(bytes.NewReader(b)); err == nil {
			t.Fatalf("accepted %v", v)
		}
	}
}
func TestRenderersAndComparison(t *testing.T) {
	e := fixture(t)
	r := Score(e)
	prior := r
	prior.Snapshot = "prior"
	old := .1
	prior.Dimensions[0].NormalizedRatio = &old
	if err := Compare(&r, prior); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(RenderHuman(r), "dominant bottleneck") || !strings.Contains(RenderMarkdown(r), "Normalized ratio") {
		t.Fatal("renderer missing fields")
	}
	b, err := MarshalJSON(r)
	if err != nil || !bytes.Contains(b, []byte(`"comparison"`)) {
		t.Fatalf("json: %v", err)
	}
}
func TestNativeProvenanceAndNoLlamaFallback(t *testing.T) {
	e := fixture(t)
	e.Dimensions[2].EvidenceKind = "native_benchmark"
	e.Dimensions[2].Engine = ""
	b, _ := json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted unnamed native engine")
	}
	e.Dimensions[2].Engine = "llama.cpp"
	b, _ = json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted llama fallback")
	}
	e.Dimensions[2].Engine = "fak-native/qwen3.8"
	b, _ = json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
}
func TestFixtureIsVersioned(t *testing.T) {
	b, err := os.ReadFile("testdata/complete.json")
	if err != nil || !bytes.Contains(b, []byte(EvidenceSchema)) {
		t.Fatal(err)
	}
}
