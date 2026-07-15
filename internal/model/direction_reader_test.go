package model

import (
	"errors"
	"reflect"
	"testing"

	"math"
)

func TestDirectionReaderFitsReadsSteersAndReloads(t *testing.T) {
	pairs := []DirectionContrast{
		{Positive: []float32{4, 1, 0}, Negative: []float32{0, 3, 0}},
		{Positive: []float32{6, 2, 0}, Negative: []float32{2, 4, 0}},
	}
	reader, err := FitDirectionReader("positive framing", 8, pairs)
	if err != nil {
		t.Fatal(err)
	}
	if reader.Name() != "positive framing" || reader.Layer() != 8 || reader.Scale() <= 0 {
		t.Fatalf("metadata name=%q layer=%d scale=%f", reader.Name(), reader.Layer(), reader.Scale())
	}
	positive, _ := reader.Read([]float32{5, 1.5, 0})
	negative, _ := reader.Read([]float32{1, 3.5, 0})
	if positive <= negative {
		t.Fatalf("fitted direction does not separate fixtures: positive=%f negative=%f", positive, negative)
	}

	prior := float32(-100)
	for _, alpha := range []float32{-2, -1, 0, 1, 2} {
		hidden := []float32{1, 1, 0}
		if err := reader.Steer(8, alpha, hidden); err != nil {
			t.Fatal(err)
		}
		projection, err := reader.Read(hidden)
		if err != nil {
			t.Fatal(err)
		}
		if projection <= prior {
			t.Fatalf("alpha=%f projection=%f prior=%f", alpha, projection, prior)
		}
		prior = projection
	}

	encoded, err := MarshalDirectionReader(reader)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadDirectionReader(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Name() != reader.Name() || reloaded.Layer() != reader.Layer() || reloaded.Scale() != reader.Scale() {
		t.Fatalf("reload metadata got=(%q,%d,%f) want=(%q,%d,%f)", reloaded.Name(), reloaded.Layer(), reloaded.Scale(), reader.Name(), reader.Layer(), reader.Scale())
	}
	before := []float32{3, -2, 1}
	want, _ := reader.Read(before)
	got, _ := reloaded.Read(before)
	if got != want {
		t.Fatalf("reload projection=%f want=%f", got, want)
	}
	encodedAgain, err := MarshalDirectionReader(reloaded)
	if err != nil || !reflect.DeepEqual(encodedAgain, encoded) {
		t.Fatalf("serialization is not deterministic: err=%v\nfirst=%s\nagain=%s", err, encoded, encodedAgain)
	}
}

func TestDirectionReaderRejectsInvalidFitAndUse(t *testing.T) {
	if _, err := FitDirectionReader("", 1, []DirectionContrast{{Positive: []float32{1}, Negative: []float32{0}}}); !errors.Is(err, ErrDirectionName) {
		t.Fatalf("empty name err=%v", err)
	}
	if _, err := FitDirectionReader("x", -1, []DirectionContrast{{Positive: []float32{1}, Negative: []float32{0}}}); !errors.Is(err, ErrDirectionLayer) {
		t.Fatalf("negative layer err=%v", err)
	}
	if _, err := FitDirectionReader("x", 1, []DirectionContrast{{Positive: []float32{1}, Negative: []float32{1}}}); !errors.Is(err, ErrDirectionFit) {
		t.Fatalf("zero direction err=%v", err)
	}
	if _, err := FitDirectionReader("x", 1, []DirectionContrast{{Positive: []float32{float32(math.NaN())}, Negative: []float32{0}}}); !errors.Is(err, ErrDirectionFit) {
		t.Fatalf("non-finite direction err=%v", err)
	}
	reader, err := FitDirectionReader("x", 1, []DirectionContrast{{Positive: []float32{2, 0}, Negative: []float32{0, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := reader.Steer(2, 1, []float32{0, 0}); !errors.Is(err, ErrDirectionLayer) {
		t.Fatalf("wrong layer err=%v", err)
	}
	if _, err := reader.Read([]float32{0}); !errors.Is(err, ErrDirectionWidth) {
		t.Fatalf("wrong width err=%v", err)
	}
	if _, err := LoadDirectionReader([]byte(`{"schema":"fak.direction-reader.v1","name":"x","layer":1,"scale":1,"direction":[2]}`)); !errors.Is(err, ErrDirectionFit) {
		t.Fatalf("invalid norm err=%v", err)
	}
}
