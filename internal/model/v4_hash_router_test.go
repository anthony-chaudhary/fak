package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writeV4HashFixture(t *testing.T, mutate func(layer int, tensor *tinySTTensor)) string {
	t.Helper()
	dir := t.TempDir()
	weightMap := map[string]string{}
	for layer := 0; layer < v4HashLayers; layer++ {
		name := v4HashTensorName(layer)
		data := make([]byte, v4HashVocab*v4HashTopK*8)
		for token := 0; token < v4HashVocab; token++ {
			for slot := 0; slot < v4HashTopK; slot++ {
				binary.LittleEndian.PutUint64(data[(token*v4HashTopK+slot)*8:], uint64((token+slot+layer*11)%v4HashExperts))
			}
		}
		tensor := tinySTTensor{dtype: "I64", shape: []int{v4HashVocab, v4HashTopK}, data: data}
		if mutate != nil {
			mutate(layer, &tensor)
		}
		shard := "hash" + string(rune('0'+layer)) + ".safetensors"
		writeV4RawShard(t, filepath.Join(dir, shard), map[string]tinySTTensor{name: tensor})
		weightMap[name] = shard
	}
	ib, err := json.Marshal(map[string]any{"weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestV4HashRouterReadsOneExactRowAndMatchesOracle(t *testing.T) {
	dir := writeV4HashFixture(t, nil)
	s, err := newV4HashRouterSource(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.openCount != 0 || s.readCount != 0 || s.readBytes != 0 {
		t.Fatalf("constructor touched payload tier: opens=%d reads=%d bytes=%d", s.openCount, s.readCount, s.readBytes)
	}
	ids, err := s.lookup(1, 42)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []int{53, 54, 55, 56, 57, 58}
	for i := range wantIDs {
		if ids[i] != wantIDs[i] {
			t.Fatalf("ids=%v want=%v", ids, wantIDs)
		}
	}
	if s.openCount != 1 || s.readCount != 1 || s.readBytes != 48 {
		t.Fatalf("lookup accounting: opens=%d reads=%d bytes=%d", s.openCount, s.readCount, s.readBytes)
	}

	logits := make([]float32, v4HashExperts)
	for i := range logits {
		logits[i] = float32(i%17)/4 - 2
	}
	got, err := v4HashRoute(logits, ids, 2.5)
	if err != nil {
		t.Fatal(err)
	}
	var denom float64
	for _, id := range wantIDs {
		denom += math.Sqrt(math.Log1p(math.Exp(float64(logits[id]))))
	}
	for i, id := range wantIDs {
		if got[i].expert != id {
			t.Fatalf("pick %d=%+v want expert %d", i, got[i], id)
		}
		want := float32(math.Sqrt(math.Log1p(math.Exp(float64(logits[id])))) / denom * 2.5)
		if math.Abs(float64(got[i].weight-want)) > 2e-6 {
			t.Fatalf("expert %d weight=%g want=%g", id, got[i].weight, want)
		}
	}
}

func TestV4HashRouterMetadataAndValueFailuresAreTyped(t *testing.T) {
	t.Run("dtype", func(t *testing.T) {
		dir := writeV4HashFixture(t, func(layer int, x *tinySTTensor) {
			if layer == 0 {
				x.dtype = "I32"
			}
		})
		s, err := newV4HashRouterSource(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		_, err = s.lookup(0, 0)
		assertV4RouteError(t, err)
		if s.readCount != 0 || s.readBytes != 0 {
			t.Fatalf("metadata refusal read payload: reads=%d bytes=%d", s.readCount, s.readBytes)
		}
	})
	t.Run("shape", func(t *testing.T) {
		dir := writeV4HashFixture(t, func(layer int, x *tinySTTensor) {
			if layer == 0 {
				x.shape = []int{v4HashVocab, 5}
			}
		})
		s, _ := newV4HashRouterSource(dir)
		defer s.Close()
		_, err := s.lookup(0, 0)
		assertV4RouteError(t, err)
		if s.readCount != 0 {
			t.Fatal("shape refusal read payload")
		}
	})
	t.Run("data range", func(t *testing.T) {
		dir := writeV4HashFixture(t, func(layer int, x *tinySTTensor) {
			if layer == 0 {
				x.data = x.data[:len(x.data)-1]
			}
		})
		s, _ := newV4HashRouterSource(dir)
		defer s.Close()
		_, err := s.lookup(0, 0)
		assertV4RouteError(t, err)
		if s.readCount != 0 {
			t.Fatal("range refusal read payload")
		}
	})
	t.Run("expert range", func(t *testing.T) {
		dir := writeV4HashFixture(t, func(layer int, x *tinySTTensor) {
			if layer == 0 {
				binary.LittleEndian.PutUint64(x.data, v4HashExperts)
			}
		})
		s, _ := newV4HashRouterSource(dir)
		defer s.Close()
		_, err := s.lookup(0, 0)
		assertV4RouteError(t, err)
	})
	t.Run("duplicate", func(t *testing.T) {
		dir := writeV4HashFixture(t, func(layer int, x *tinySTTensor) {
			if layer == 0 {
				binary.LittleEndian.PutUint64(x.data[8:], binary.LittleEndian.Uint64(x.data))
			}
		})
		s, _ := newV4HashRouterSource(dir)
		defer s.Close()
		_, err := s.lookup(0, 0)
		assertV4RouteError(t, err)
	})
}

func TestV4HashRouterRejectsInvalidRequestsAndWeights(t *testing.T) {
	dir := writeV4HashFixture(t, nil)
	s, err := newV4HashRouterSource(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, tc := range []struct{ layer, token int }{{-1, 0}, {3, 0}, {0, -1}, {0, v4HashVocab}} {
		_, err := s.lookup(tc.layer, tc.token)
		assertV4RouteError(t, err)
	}
	if s.openCount != 0 || s.readCount != 0 {
		t.Fatalf("invalid request touched shards: opens=%d reads=%d", s.openCount, s.readCount)
	}

	baseIDs := []int{0, 1, 2, 3, 4, 5}
	cases := []struct {
		logits []float32
		ids    []int
		scale  float32
	}{
		{make([]float32, 383), baseIDs, 2.5},
		{make([]float32, 384), []int{0, 1}, 2.5},
		{make([]float32, 384), []int{0, 1, 2, 3, 4, 384}, 2.5},
		{make([]float32, 384), []int{0, 1, 2, 3, 4, 4}, 2.5},
		{make([]float32, 384), baseIDs, 0},
	}
	for _, tc := range cases {
		_, err := v4HashRoute(tc.logits, tc.ids, tc.scale)
		assertV4RouteError(t, err)
	}
	logits := make([]float32, 384)
	logits[2] = float32(math.NaN())
	_, err = v4HashRoute(logits, baseIDs, 2.5)
	assertV4RouteError(t, err)
}

func assertV4RouteError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected fail-closed error")
	}
	var typed *v4RouteError
	if !errors.As(err, &typed) {
		t.Fatalf("error %T is not *v4RouteError: %v", err, err)
	}
}
