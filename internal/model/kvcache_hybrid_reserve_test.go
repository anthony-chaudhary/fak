package model

import "testing"

func TestKVCacheHybridReserveSkipsRecurrentKVPlanes(t *testing.T) {
	hybrid := Config{
		ModelType:  "qwen3_5",
		NumLayers:  4,
		NumKVHeads: 1,
		HeadDim:    2,
		LayerTypes: []string{"linear_attention", "full_attention", "linear_attention", "full_attention"},
	}
	cache := NewKVCache(hybrid)
	cache.K[0], cache.Kraw[0], cache.V[0] = []float32{1, 2}, []float32{3, 4}, []float32{5, 6}
	cache.K[1], cache.Kraw[1], cache.V[1] = []float32{7, 8}, []float32{9, 10}, []float32{11, 12}
	cache.pos = []int{0}

	cache.Reserve(3)
	for _, layer := range []int{1, 3} {
		want := len(cache.K[layer]) + 6
		if got := cap(cache.K[layer]); got < want || cap(cache.Kraw[layer]) < want || cap(cache.V[layer]) < want {
			t.Fatalf("full-attention layer %d capacities = %d/%d/%d, want >= %d", layer, got, cap(cache.Kraw[layer]), cap(cache.V[layer]), want)
		}
	}
	if cap(cache.K[2]) != 0 || cap(cache.Kraw[2]) != 0 || cap(cache.V[2]) != 0 {
		t.Fatalf("empty recurrent layer reserved KV capacity: %d/%d/%d", cap(cache.K[2]), cap(cache.Kraw[2]), cap(cache.V[2]))
	}
	if len(cache.K[0]) != 2 || cache.K[0][0] != 1 || cap(cache.K[0]) < 8 {
		t.Fatalf("pre-existing recurrent KV was not preserved safely: len=%d cap=%d values=%v", len(cache.K[0]), cap(cache.K[0]), cache.K[0])
	}

	clone := cache.CloneWithReserve(2)
	if cap(clone.K[2]) != 0 || cap(clone.Kraw[2]) != 0 || cap(clone.V[2]) != 0 {
		t.Fatalf("clone reserved empty recurrent KV capacity: %d/%d/%d", cap(clone.K[2]), cap(clone.Kraw[2]), cap(clone.V[2]))
	}
	if len(clone.K[0]) != 2 || clone.K[0][0] != 1 || cap(clone.K[0]) < len(clone.K[0])+4 {
		t.Fatalf("clone dropped or failed to reserve pre-existing recurrent KV: len=%d cap=%d values=%v", len(clone.K[0]), cap(clone.K[0]), clone.K[0])
	}

	plain := NewKVCache(Config{NumLayers: 2, NumKVHeads: 1, HeadDim: 2})
	plain.Reserve(3)
	for layer := range plain.K {
		if cap(plain.K[layer]) < 6 || cap(plain.Kraw[layer]) < 6 || cap(plain.V[layer]) < 6 {
			t.Fatalf("nonhybrid layer %d lost reserve behavior: %d/%d/%d", layer, cap(plain.K[layer]), cap(plain.Kraw[layer]), cap(plain.V[layer]))
		}
	}
}
