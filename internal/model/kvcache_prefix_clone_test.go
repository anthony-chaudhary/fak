package model

import (
	"reflect"
	"testing"
)

func prefixCloneOracle(c *KVCache, keep int) *KVCache {
	n := c.Clone()
	if keep < n.Len() {
		if _, err := n.TryEvict(keep, n.Len()-keep); err != nil {
			return nil
		}
	}
	return n
}

func TestKVCacheClonePrefixExactDense(t *testing.T) {
	for _, prec := range []KVPrecision{KVPrecisionFP32, KVPrecisionQ8_0} {
		for _, keep := range []int{-1, 0, 1, 7, 20, 21} {
			c := NewKVCacheWithPrecision(q8TestConfig(), prec)
			appendSynthetic(t, c, 20, 811)
			c.Reserve(12)
			c.lineage.fault = "preserve diagnostic fault"
			before := c.Clone()
			want, got := prefixCloneOracle(c, keep), c.ClonePrefixExact(keep)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("precision=%s keep=%d differs from Clone+TryEvict", prec, keep)
			}
			if !reflect.DeepEqual(c.Clone(), before) {
				t.Fatalf("precision=%s keep=%d mutated source", prec, keep)
			}
			if keep < 0 || keep >= c.Len() {
				continue
			}
			if !c.canCloneDensePrefix(keep) || cap(got.pos) != keep || cap(got.lineage.ids) != keep {
				t.Fatalf("precision=%s keep=%d did not produce tight prefix metadata", prec, keep)
			}
			for l := range got.Kraw {
				if cap(got.Kraw[l]) != keep*c.kvStride() {
					t.Fatal("retained discarded Kraw capacity")
				}
				if got.quantized() {
					if cap(got.kQ8[l].codes) != keep*c.kvStride() || cap(got.vQ8[l].codes) != keep*c.kvStride() {
						t.Fatal("retained discarded packed capacity")
					}
				} else if cap(got.K[l]) != keep*c.kvStride() || cap(got.V[l]) != keep*c.kvStride() {
					t.Fatal("retained discarded K/V capacity")
				}
			}
			if keep > 0 {
				got.Kraw[0][0]++
				got.lineage.ids[0]++
				if got.quantized() {
					got.kQ8[0].codes[0]++
					got.vQ8[0].scales[0]++
				} else {
					got.K[0][0]++
					got.V[0][0]++
				}
				if !reflect.DeepEqual(c.Clone(), before) {
					t.Fatal("result aliases source payload")
				}
			}
		}
	}
}

func TestKVCacheClonePrefixExactFallbacks(t *testing.T) {
	for _, corrupt := range []func(*KVCache){
		func(c *KVCache) { c.lineage.ids = c.lineage.ids[:2] },
		func(c *KVCache) { c.pos[0] = 2 },
	} {
		c := NewKVCache(q8TestConfig())
		appendSynthetic(t, c, 8, 99)
		corrupt(c)
		if c.canCloneDensePrefix(3) {
			t.Fatal("irregular cache admitted to prefix-copy fast path")
		}
		if !reflect.DeepEqual(c.ClonePrefixExact(3), prefixCloneOracle(c, 3)) {
			t.Fatal("fallback differs from original lifecycle")
		}
	}
	c := NewKVCache(q8TestConfig())
	appendSynthetic(t, c, 8, 99)
	c.linear = &linearAttnCache{}
	if c.ClonePrefixExact(3) != nil || c.canCloneDensePrefix(3) {
		t.Fatal("recurrent truncation was admitted")
	}
	if !reflect.DeepEqual(c.ClonePrefixExact(8), c.Clone()) {
		t.Fatal("complete recurrent clone changed")
	}
}
