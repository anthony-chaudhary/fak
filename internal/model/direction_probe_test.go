package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
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
