// Default suites: go test ./internal/model and go test ./... (also under -race).
// These small in-memory contracts require no artifact, subprocess, or opt-in.
package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func quantArenaBytes(values []float32) []byte {
	b := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(value))
	}
	return b
}

func quantArenaAdd(t *testing.T, b *QuantBuilder, name string, shape []int, data []float32) {
	t.Helper()
	if err := b.AddF32Tensor(name, shape, data); err != nil {
		t.Fatalf("add %s: %v", name, err)
	}
}

func quantArenaHead(t *testing.T, b *QuantBuilder) {
	t.Helper()
	data := make([]float32, 32)
	for i := range data {
		data[i] = 1
	}
	quantArenaAdd(t, b, "lm_head.weight", []int{1, 32}, data)
}

func quantArenaBuild(t *testing.T, b *QuantBuilder) *Model {
	t.Helper()
	m, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	return m
}

func quantArenaTensor(t *testing.T, m *Model, name string, offset int, shape []int, want []byte) {
	t.Helper()
	meta, ok := m.manifest[name]
	if !ok || meta.Dtype != "f32" || meta.Offset != offset || meta.Nbytes != len(want) || !reflect.DeepEqual(meta.Shape, shape) {
		t.Fatalf("tensor %s metadata=%+v present=%v, want offset=%d shape=%v bytes=%d", name, meta, ok, offset, shape, len(want))
	}
	if got := quantArenaBytes(m.tensor(name)); !bytes.Equal(got, want) {
		t.Fatalf("tensor %s did not preserve its F32 snapshot", name)
	}
}

func quantArenaExactStorage(t *testing.T, b *QuantBuilder, m *Model) {
	t.Helper()
	if cap(m.raw) != len(m.raw) {
		t.Fatalf("final F32 raw length=%d capacity=%d; capacity must equal length", len(m.raw), cap(m.raw))
	}
	// Examine ownership, not field names or a particular chunk representation.
	// The returned Model owns its buffers. A builder reference wholly within the
	// final raw range is only an alias; independent byte/F32 backing is retained work.
	modelValue := reflect.ValueOf(m)
	rawPointer := reflect.ValueOf(m.raw).Pointer()
	rawBytes := uintptr(len(m.raw))
	seen := make(map[uintptr]bool)
	var walk func(reflect.Value, int) error
	walk = func(v reflect.Value, depth int) error {
		if !v.IsValid() {
			return nil
		}
		if depth > 128 {
			return fmt.Errorf("builder ownership graph is not bounded")
		}
		switch v.Kind() {
		case reflect.Interface:
			if !v.IsNil() {
				return walk(v.Elem(), depth+1)
			}
		case reflect.Pointer:
			if v.IsNil() || (v.Type() == modelValue.Type() && v.Pointer() == modelValue.Pointer()) {
				return nil
			}
			if seen[v.Pointer()] {
				return nil
			}
			seen[v.Pointer()] = true
			return walk(v.Elem(), depth+1)
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if err := walk(v.Field(i), depth+1); err != nil {
					return err
				}
			}
		case reflect.Slice:
			kind := v.Type().Elem().Kind()
			if (kind == reflect.Uint8 || kind == reflect.Float32) && v.Cap() != 0 {
				n := uintptr(v.Cap()) * v.Type().Elem().Size()
				p := v.Pointer()
				if p < rawPointer || n > rawBytes || p-rawPointer > rawBytes-n {
					return fmt.Errorf("builder retains independent %s backing with capacity %d", v.Type(), v.Cap())
				}
				return nil
			}
			// Pointer-bearing backing storage remains live beyond the slice length.
			full := v.Slice(0, v.Cap())
			for i := 0; i < full.Len(); i++ {
				if err := walk(full.Index(i), depth+1); err != nil {
					return err
				}
			}
		case reflect.Array:
			for i := 0; i < v.Len(); i++ {
				if err := walk(v.Index(i), depth+1); err != nil {
					return err
				}
			}
		case reflect.Map:
			iter := v.MapRange()
			for iter.Next() {
				if err := walk(iter.Value(), depth+1); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := walk(reflect.ValueOf(b), 0); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestQuantBuilderF32ArenaSnapshotsOffsetsAndCapacity(t *testing.T) {
	t.Parallel()
	b := NewQuantBuilder(Config{HiddenSize: 32}, false)
	quantArenaHead(t, b)
	input := make([]float32, 31)
	shape := []int{17}
	for i := range input {
		input[i] = math.Float32frombits(uint32(0x3f000000 + i))
	}
	input[0] = math.Float32frombits(0x80000000)
	input[1] = math.Float32frombits(0x7fc12345)
	input[2] = math.Float32frombits(0xffc54321)
	first := quantArenaBytes(input[:17])
	quantArenaAdd(t, b, "contract.first", shape, input[:17])
	shape[0] = 99
	for i := range input {
		input[i] = float32(i + 20)
	}
	second := quantArenaBytes(input)
	shape[0] = 31
	quantArenaAdd(t, b, "contract.second", shape, input)
	for i := range input {
		input[i] = -float32(i + 1)
	}
	last := quantArenaBytes(input[:5])
	shape[0] = 5
	quantArenaAdd(t, b, "contract.first", shape, input[:5])
	shape[0] = 99
	for i := range input {
		input[i] = 900
	}
	m := quantArenaBuild(t, b)
	wantRaw := append(append(append([]byte(nil), first...), second...), last...)
	if !bytes.Equal(m.raw, wantRaw) {
		t.Fatal("global F32 storage changed its input snapshots or append history")
	}
	quantArenaTensor(t, m, "contract.second", len(first), []int{31}, second)
	quantArenaTensor(t, m, "contract.first", len(first)+len(second), []int{5}, last)
	quantArenaExactStorage(t, b, m)
}

// fak-test:runtime fast est=2ms lane=default
func TestQuantBuilderF32ArenaTiedQ8Numerics(t *testing.T) {
	t.Parallel()
	b := NewQuantBuilder(Config{HiddenSize: 32, VocabSize: 2, TieWordEmbeddings: true}, true)
	prefix := []float32{3, -2, 1}
	quantArenaAdd(t, b, "contract.prefix", []int{3}, prefix)
	data := make([]float32, 64)
	x := make([]float32, 32)
	for i := range x {
		sign := float32(-1)
		if i%2 != 0 {
			sign = 1
		}
		x[i] = sign
		data[i] = sign
		data[32+i] = sign / 2
	}
	embedding := quantArenaBytes(data)
	quantArenaAdd(t, b, "model.embed_tokens.weight", []int{2, 32}, data)
	for i := range data {
		data[i] = 0
	}
	m := quantArenaBuild(t, b)
	quantArenaTensor(t, m, "model.embed_tokens.weight", 12, []int{2, 32}, embedding)
	if m.headName() != "model.embed_tokens.weight" {
		t.Fatalf("tied head resolved to %s", m.headName())
	}
	qt := m.q8w[m.headName()]
	if qt == nil || qt.out != 2 || qt.in != 32 || len(qt.q) != 64 || len(qt.d) != 2 {
		t.Fatal("tied Q8 head geometry changed")
	}
	for i, code := range qt.q {
		want := int8(-127)
		if i%2 != 0 {
			want = 127
		}
		if code != want {
			t.Fatalf("tied Q8 code[%d]=%d, want %d", i, code, want)
		}
	}
	for i, want := range []float32{float32(1.0 / 127), float32(0.5 / 127)} {
		if math.Float32bits(qt.d[i]) != math.Float32bits(want) {
			t.Fatalf("tied Q8 scale[%d]=%g, want %g", i, qt.d[i], want)
		}
	}
	y := qMatRows(qt, quantizeVecQ8(x))
	for i, want := range []float32{32, 16} {
		if math.Abs(float64(y[i]-want)) > 1e-4 {
			t.Fatalf("tied Q8 CPU row[%d]=%g, want %g", i, y[i], want)
		}
	}
	quantArenaExactStorage(t, b, m)
}

// fak-test:runtime fast est=2ms lane=default
func TestQuantBuilderF32ArenaAliasesAndPartialError(t *testing.T) {
	t.Parallel()
	b := NewQuantBuilder(Config{HiddenSize: 32, NumLayers: 1, NumExperts: 2, IntermediateSize: 3}, false)
	quantArenaHead(t, b)
	router := []float32{0.5, -0.5}
	quantArenaAdd(t, b, "model.layers.0.mlp.gate.bias", []int{2}, router)
	quantArenaAdd(t, b, "model.layers.0.mlp.router.bias", []int{2}, []float32{8, 9})
	preexisting := []float32{91, 92, 93}
	quantArenaAdd(t, b, "model.layers.0.mlp.experts.1.gate_proj.bias", []int{3}, preexisting)
	input := []float32{1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60}
	err := b.AddF32Tensor("model.layers.0.mlp.experts.gate_up_proj_bias", []int{2, 6}, input)
	if err == nil || !strings.Contains(err.Error(), "expert 1") {
		t.Fatalf("expected the later expert collision, got %v", err)
	}
	for i := range input {
		input[i] = 700
	}
	m := quantArenaBuild(t, b)
	quantArenaTensor(t, m, "model.layers.0.mlp.gate.bias", 0, []int{2}, quantArenaBytes(router))
	quantArenaTensor(t, m, "model.layers.0.mlp.experts.1.gate_proj.bias", 8, []int{3}, quantArenaBytes(preexisting))
	quantArenaTensor(t, m, "model.layers.0.mlp.experts.0.gate_proj.bias", 20, []int{3}, quantArenaBytes([]float32{1, 2, 3}))
	quantArenaTensor(t, m, "model.layers.0.mlp.experts.0.up_proj.bias", 32, []int{3}, quantArenaBytes([]float32{10, 20, 30}))
	if len(m.raw) != 44 || m.has("model.layers.0.mlp.router.bias") || m.has("model.layers.0.mlp.experts.1.up_proj.bias") {
		t.Fatal("alias handling or partial-error append boundary changed")
	}
	quantArenaExactStorage(t, b, m)
}

// fak-test:runtime fast est=2ms lane=default
func TestQuantBuilderF32ArenaPolicyAndLifetime(t *testing.T) {
	t.Parallel()
	for _, retain := range []bool{false, true} {
		b := NewQuantBuilder(Config{ModelType: "glm_moe_dsa", HiddenSize: 32}, false)
		if err := b.SetMTPRetention(retain); err != nil {
			t.Fatal(err)
		}
		quantArenaHead(t, b)
		data := []float32{2, 3, 4}
		quantArenaAdd(t, b, "mtp.contract.norm", []int{3}, data)
		if err := b.SetMTPRetention(!retain); err == nil {
			t.Fatal("MTP retention changed after tensor input")
		}
		data[0] = 800
		m := quantArenaBuild(t, b)
		if m.has("mtp.contract.norm") != retain {
			t.Fatalf("MTP presence does not match retain=%v", retain)
		}
		if retain {
			quantArenaTensor(t, m, "mtp.contract.norm", 0, []int{3}, quantArenaBytes([]float32{2, 3, 4}))
		} else if len(m.raw) != 0 {
			t.Fatal("dropped MTP retained F32 storage")
		}
		before := append([]byte(nil), m.raw...)
		if err := b.AddF32Tensor("contract.after", []int{1}, []float32{9}); err == nil {
			t.Fatal("Add succeeded after Build")
		}
		if again, err := b.Build(); err == nil || again != nil {
			t.Fatal("Build succeeded twice")
		}
		if !bytes.Equal(m.raw, before) {
			t.Fatal("rejected post-Build operation changed model storage")
		}
		quantArenaExactStorage(t, b, m)
	}
	failed := NewQuantBuilder(Config{}, false)
	quantArenaAdd(t, failed, "contract.only", []int{1}, []float32{1})
	if m, err := failed.Build(); err == nil || m != nil {
		t.Fatal("Build without quantizable weights succeeded")
	}
	if err := failed.AddF32Tensor("contract.retry", []int{1}, []float32{2}); err == nil {
		t.Fatal("failed Build did not consume the builder")
	}
	if m, err := failed.Build(); err == nil || m != nil {
		t.Fatal("failed builder was rebuilt")
	}
	refused := NewQuantBuilder(Config{ModelType: "deepseek_v41"}, false)
	for _, err := range []error{
		refused.SetMTPRetention(true),
		refused.AddF32Tensor("contract.refused", []int{1}, []float32{1}),
		refused.SetQ2KEmbedding(nil),
	} {
		if !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("V4.1 mutation lost typed refusal: %v", err)
		}
	}
	if m, err := refused.Build(); m != nil || !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("V4.1 Build returned model=%v error=%v", m != nil, err)
	}
}
