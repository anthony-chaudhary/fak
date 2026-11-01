package model

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestQwen35StateLayoutShapesAndBytes(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.LinearValueHeadDim, cfg.LinearConvKernelDim = 6, 4
	for _, dtype := range []string{"", "f32", "f16", "bf16"} {
		p, err := ComputeQwen35StateLayout(cfg, dtype)
		if err != nil {
			t.Fatal(err)
		}
		width := int64(4)
		if dtype == "f16" || dtype == "bf16" {
			width = 2
		}
		if p.ConvRows != 3 || p.ConvDim != 56 || p.ConvElements != 168 ||
			p.RecurrentHeads != 4 || p.RecurrentRowElements != 48 || p.RecurrentElements != 192 ||
			p.ConvBytes != 168*width || p.RecurrentBytes != 192*width || p.BytesPerLayer != 360*width {
			t.Fatalf("%q layout=%+v", dtype, p)
		}
	}
	if _, err := ComputeQwen35StateLayout(cfg, "q8"); err == nil {
		t.Fatal("unsupported state dtype accepted")
	}
	cfg.LinearConvKernelDim = 1
	p, err := ComputeQwen35StateLayout(cfg, "f32")
	if err != nil || p.ConvRows != 0 || p.ConvBytes != 0 {
		t.Fatalf("width-one convolution: %+v, %v", p, err)
	}
	for _, edit := range []func(*Config){
		func(c *Config) { c.LinearConvKernelDim = 0 },
		func(c *Config) { c.LinearNumKeyHeads = int(^uint(0) >> 1) },
		func(c *Config) { c.LinearConvKernelDim = int(^uint(0) >> 1) },
		func(c *Config) { c.LinearNumValueHeads = int(^uint(0)>>1) / 64 },
	} {
		bad := cfg
		edit(&bad)
		if _, err := ComputeQwen35StateLayout(bad, "f32"); err == nil {
			t.Fatalf("invalid/overflow geometry accepted: %+v", bad)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35StateLayoutHostHALAllocationParity(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	p, err := ComputeQwen35StateLayout(cfg, "f32")
	if err != nil {
		t.Fatal(err)
	}
	host := newLinearAttnLayerState(cfg)
	if len(host.conv) != 0 || len(host.recurrent) != p.RecurrentHeads {
		t.Fatal("host lazy convolution or recurrent ownership changed")
	}
	for _, row := range host.recurrent {
		if len(row) != p.RecurrentRowElements {
			t.Fatal("host recurrent row differs from plan")
		}
	}
	host.recurrent[0][0] = 7
	if host.recurrent[1][0] != 0 {
		t.Fatal("host head buffers alias")
	}
	m := NewSynthetic(cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })
	be := &qwen35InitFailureBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)}
	t.Cleanup(be.reference.Close)
	s := &Session{M: m, Backend: be}
	s.initQwen35HALState(be)
	t.Cleanup(func() { s.qwen35HAL.free(be) })
	for _, l := range be.linearLayers {
		state := s.qwen35HAL.layers[l]
		if !reflect.DeepEqual(state.conv.Shape, []int{p.ConvRows, p.ConvDim}) ||
			!reflect.DeepEqual(state.recurrent.Shape, []int{p.RecurrentHeads, p.KeyHeadDim, p.ValueHeadDim}) ||
			state.conv.Numel() != p.ConvElements || state.recurrent.Numel() != p.RecurrentElements {
			t.Fatalf("HAL layer %d shapes differ from plan", l)
		}
	}
	before := be.attempts
	m.Cfg.LinearNumValueHeads = 0
	bad := &Session{M: m, Backend: be}
	var caught any
	func() {
		defer func() { caught = recover() }()
		bad.initQwen35HALState(be)
	}()
	if caught == nil || !strings.Contains(fmt.Sprint(caught), "qwen35 state layout") || be.attempts != before || bad.qwen35HAL != nil {
		t.Fatalf("HAL invalid shape not refused before upload: panic=%v attempts=%d/%d state=%p", caught, be.attempts, before, bad.qwen35HAL)
	}
}
