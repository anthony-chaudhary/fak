package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"fmt"
)

func TestVerbalizableDirectionProbe(t *testing.T) {
	positive := [][]float32{{3, 1, 0}, {4, 1, 0}, {5, 2, 0}}
	negative := [][]float32{{0, 2, 0}, {1, 3, 0}, {0, 4, 0}}
	d := VerbalizableDirection(positive, negative)
	want := []float32{11.0 / 3, -5.0 / 3, 0}
	if len(d) != 3 || DirectionCosine(d, want) < .999 {
		t.Fatalf("direction=%v cosine=%f", d, DirectionCosine(d, want))
	}
}

func TestSteerHookByteIdentical(t *testing.T) {
	base := []float32{1, 2, 3}
	got := append([]float32(nil), base...)
	if (DirectionSteer{}).Apply(2, 7, got) || !reflect.DeepEqual(got, base) {
		t.Fatalf("unarmed changed hidden: base=%v got=%v", base, got)
	}
	armed := DirectionSteer{Layer: 2, Position: 7, Alpha: 2, Direction: []float32{0.5, -0.5, 0}}
	if !armed.Apply(2, 7, got) || !reflect.DeepEqual(got, []float32{2, 1, 3}) {
		t.Fatalf("armed got=%v", got)
	}
}

func TestDirectionBroadcast(t *testing.T) {
	d := VerbalizableDirection([][]float32{{2, 0}}, [][]float32{{0, 0}})
	coefficients := []float32{-2, 0, 2}
	for _, layer := range []int{8, 12} {
		var prior float32 = -100
		for _, a := range coefficients {
			h := []float32{float32(layer) / 20, 1}
			steer := DirectionSteer{Layer: 4, Position: 9, Alpha: a, Direction: d}
			// Hermetic downstream propagation: the injected residual is carried
			// through an identity residual path to each observed later layer.
			steer.Layer = layer
			steer.Apply(layer, 9, h)
			projection := DirectionProjection(h, d)
			if projection <= prior {
				t.Fatalf("layer=%d alpha=%f projection=%f prior=%f", layer, a, projection, prior)
			}
			prior = projection
		}
	}
}

func TestDirectionSteerFromEnv(t *testing.T) {
	path := filepath.Join(t.TempDir(), "direction.f32")
	values := []float32{0.5, -0.5}
	b := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_HIDDEN_STEER", path)
	t.Setenv("FAK_HIDDEN_STEER_LAYER", "3")
	t.Setenv("FAK_HIDDEN_STEER_POS", "7")
	t.Setenv("FAK_HIDDEN_STEER_ALPHA", "2")
	s := directionSteerFromEnv()
	got := []float32{1, 1}
	if !s.Apply(3, 7, got) || !reflect.DeepEqual(got, []float32{2, 0}) {
		t.Fatalf("steer=%+v got=%v", s, got)
	}
}

func TestVerbalizableDirectionLiveArtifact(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "verbalizable-direction-qwen25-0.5b")
	read := func(path string) []float32 {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(b)%4 != 0 {
			t.Fatalf("%s has %d trailing bytes", path, len(b)%4)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return out
	}
	d := read(filepath.Join(root, "direction.f32"))
	positive := read(filepath.Join(root, "positive", "layer_08.f32"))
	negative := read(filepath.Join(root, "negative", "layer_08.f32"))
	diff := make([]float32, len(positive))
	for i := range diff {
		diff[i] = positive[i] - negative[i]
	}
	if cosine := DirectionCosine(d, diff); cosine < .99999 {
		t.Fatalf("captured direction cosine with diff-of-means=%f", cosine)
	}
	for _, layer := range []int{12, 23} {
		prior := float32(-math.MaxFloat32)
		for _, label := range []string{"am2", "am1", "a0", "a1", "a2"} {
			h := read(filepath.Join(root, "sweep", label, fmt.Sprintf("layer_%02d.f32", layer)))
			projection := DirectionProjection(h, d)
			if projection <= prior {
				t.Fatalf("captured layer=%d label=%s projection=%f prior=%f", layer, label, projection, prior)
			}
			prior = projection
		}
	}
}
