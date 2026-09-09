package model

import (
	"encoding/binary"
	"strings"
	"testing"
)

func TestAddCanonicalMTPQ4KIsNarrowAndFailClosed(t *testing.T) {
	cfg := Config{
		ModelType:             "qwen35",
		HiddenSize:            256,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               64,
		AttnOutputGate:        true,
		NumNextNPredictLayers: 1,
		LayerTypes:            []string{"linear_attention"},
	}
	qName := "mtp.layers.0.self_attn.q_proj.weight"
	qShape := []int{512, 256}
	qRaw := make([]byte, 512*q4kBlockBytes)
	b := NewQuantBuilder(cfg, false)
	if err := b.AddCanonicalMTPQ4K(qName, qShape, qRaw); err != nil {
		t.Fatalf("AddCanonicalMTPQ4K(q): %v", err)
	}
	kName := "mtp.layers.0.self_attn.k_proj.weight"
	kShape := []int{128, 256}
	kRaw := make([]byte, 128*q4kBlockBytes)
	if err := b.AddCanonicalMTPQ4K(kName, kShape, kRaw); err != nil {
		t.Fatalf("AddCanonicalMTPQ4K(k): %v", err)
	}
	fcName := "mtp.fc.weight"
	fcShape := []int{256, 512}
	fcRaw := make([]byte, 256*(512/32)*(2+32))
	binary.LittleEndian.PutUint16(fcRaw, 0x3c00) // f16(1)
	fcRaw[2] = 0x7f
	if err := b.AddCanonicalMTPFCQ8(fcName, fcShape, fcRaw); err != nil {
		t.Fatalf("AddCanonicalMTPFCQ8: %v", err)
	}
	m, err := b.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	for _, name := range []string{qName, kName} {
		if !m.HasQ4K(name) {
			t.Fatalf("canonical MTP projection %s was not retained", name)
		}
	}
	if !m.HasQ8(fcName) {
		t.Fatalf("canonical MTP fusion projection %s was not retained", fcName)
	}
	if got := m.q8w[fcName]; got.d[0] != 1 || got.q[0] != 127 {
		t.Fatalf("canonical MTP fusion Q8_0 first block=(d=%v q=%v), want source (1,127)", got.d[0], got.q[0])
	}

	tests := []struct {
		name  string
		canon string
		shape []int
		raw   []byte
		want  string
	}{
		{name: "target projection", canon: "model.layers.0.self_attn.q_proj.weight", shape: qShape, raw: qRaw, want: "outside mtp.layers.N"},
		{name: "other MTP weight", canon: "mtp.layers.0.self_attn.o_proj.weight", shape: []int{256, 256}, raw: make([]byte, 256*q4kBlockBytes), want: "not a q/k projection"},
		{name: "undeclared layer", canon: "mtp.layers.1.self_attn.q_proj.weight", shape: qShape, raw: qRaw, want: "invalid layer"},
		{name: "wrong shape", canon: qName, shape: []int{256, 256}, raw: make([]byte, 256*q4kBlockBytes), want: "has shape"},
		{name: "short payload", canon: qName, shape: qShape, raw: qRaw[:len(qRaw)-1], want: "payload bytes"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			builder := NewQuantBuilder(cfg, false)
			err := builder.AddCanonicalMTPQ4K(tc.canon, tc.shape, tc.raw)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want containing %q", err, tc.want)
			}
		})
	}
	for _, tc := range []struct {
		name  string
		canon string
		shape []int
		raw   []byte
		want  string
	}{
		{name: "fc wrong name", canon: "mtp.other.weight", shape: fcShape, raw: fcRaw, want: "not mtp.fc.weight"},
		{name: "fc wrong shape", canon: fcName, shape: []int{256, 256}, raw: fcRaw[:len(fcRaw)/2], want: "has shape"},
		{name: "fc short payload", canon: fcName, shape: fcShape, raw: fcRaw[:len(fcRaw)-1], want: "payload bytes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			builder := NewQuantBuilder(cfg, false)
			err := builder.AddCanonicalMTPFCQ8(tc.canon, tc.shape, tc.raw)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want containing %q", err, tc.want)
			}
		})
	}
}
