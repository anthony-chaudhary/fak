package model

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

// The resolved builder hoists the per-name fill decision out of the element loop;
// it must emit the same LCG sequence and bytes as the per-element builder.
// fak-test:runtime fast est=10ms
func TestSynthBuildRawResolvedMatchesPerElement(t *testing.T) {
	tensors := []synthTensor{
		{"model.embed_tokens.weight", []int{7, 5}},
		{"model.layers.0.attn_norm.weight", []int{5}},
		{"model.layers.0.attn.sink", []int{3}},
		{"model.layers.0.attn.wq.weight", []int{4, 5}},
		{"model.layers.0.compressor.norm.weight", []int{6}},
		{"model.layers.0.mhc.base", []int{2}},
		{"model.layers.0.ffn.w1.weight", []int{5, 3}},
	}
	wantMan, wantRaw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case strings.HasSuffix(name, "attn_norm.weight"):
			return 1
		case strings.HasSuffix(name, "attn.sink"):
			return 0.25 * next()
		case strings.HasSuffix(name, "compressor.norm.weight"):
			return 0.8 + 0.2*next()
		case strings.HasSuffix(name, "mhc.base"):
			return 0
		default:
			return synthMatmulFill(name, next)
		}
	})
	gotMan, gotRaw := synthBuildRawResolved(tensors, func(name string) synthElemFill {
		switch {
		case strings.HasSuffix(name, "attn_norm.weight"):
			return synthConstFill(1)
		case strings.HasSuffix(name, "attn.sink"):
			return synthScaledFill(0.25)
		case strings.HasSuffix(name, "compressor.norm.weight"):
			return synthElemFill{custom: func(next func() float32) float32 { return 0.8 + 0.2*next() }}
		case strings.HasSuffix(name, "mhc.base"):
			return synthConstFill(0)
		default:
			return synthMatmulFillFor(name)
		}
	})
	if !reflect.DeepEqual(gotMan, wantMan) {
		t.Fatal("resolved manifest differs from per-element manifest")
	}
	if !bytes.Equal(gotRaw, wantRaw) {
		t.Fatal("resolved raw bytes differ from per-element raw bytes")
	}
}
