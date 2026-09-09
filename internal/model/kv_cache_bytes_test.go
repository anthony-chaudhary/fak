package model

import "testing"

type cpuPayloadSizer interface {
	OwnedPayloadBytes() int64
	ClonePayloadBytes() int64
}

func requireCPUPayloadSizer(t *testing.T, c *KVCache) cpuPayloadSizer {
	t.Helper()
	s, ok := any(c).(cpuPayloadSizer)
	if !ok {
		t.Fatal("CPU cache backing payload sizing is unavailable")
	}
	return s
}

func TestCPUCacheClonePayloadIncludesNestedState(t *testing.T) {
	c := NewKVCache(Config{NumLayers: 1})
	c.K[0] = make([]float32, 3, 9)
	c.K[0][0] = 7
	c.linear = &linearAttnCache{layers: []linearAttnLayerState{{conv: [][]float32{{1, 2, 3}}, recurrent: [][]float32{{4, 5, 6, 7, 8}}}}}
	// Nine reserved K floats plus eight nested state floats; clone keeps three K floats.
	sizer := requireCPUPayloadSizer(t, c)
	if got := sizer.OwnedPayloadBytes(); got != 68 {
		t.Fatalf("owned bytes=%d want68", got)
	}
	if got := sizer.ClonePayloadBytes(); got != 44 {
		t.Fatalf("clone bytes=%d want44", got)
	}
	cp := c.Clone()
	if requireCPUPayloadSizer(t, cp).OwnedPayloadBytes() != 44 || cp.K[0][0] != 7 || cp.linear.layers[0].recurrent[0][4] != 8 {
		t.Fatal("clone allocation/value mismatch")
	}
	cp.K[0][0] = 99
	cp.linear.layers[0].conv[0][0] = 99
	if c.K[0][0] != 7 || c.linear.layers[0].conv[0][0] != 1 {
		t.Fatal("clone aliases source")
	}
}
