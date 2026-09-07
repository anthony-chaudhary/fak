package agent

import (
	"context"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/model"
	"reflect"
	"strconv"
	"testing"
)

// This regression uses only the parent production API, so the same test produces
// an assertion failure before the cache-bound implementation exists.
func TestNativeCPUCacheByteBudgetPreservesGeneration(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyCfg()
	ids := synthIDs(cfg.VocabSize, 12, 17)
	run := func(cap string, scoped bool) ([]int, int64) {
		t.Setenv("FAK_INKERNEL_RADIX_CPU_BYTES", cap)
		m := model.NewSynthetic(cfg)
		p := NewInKernelPlanner(m, nil, "synthetic-cache-budget", false, nil, false)
		p.quant = false
		ctx := context.Background()
		if scoped {
			ctx = WithPrefixCacheIdentity(ctx, "tenant-budget", "")
		}
		var tokens []int
		_, err := p.generateReusedRecovering(ctx, ids, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool { tokens = append(tokens, id); return true })
		if err != nil {
			t.Fatal(err)
		}
		if cap=="1" && !scoped {
			b,_:=p.tree.Lookup(ids)
			retained:=b.KV()!=nil
			p.tree.Done(b)
			if retained { t.Fatal("native cache retained the full prompt KV despite a one-byte cap") }
		}
		data, _ := json.Marshal(p.tree.Stats())
		var stats map[string]any
		if err := json.Unmarshal(data, &stats); err != nil {
			t.Fatal(err)
		}
		if cap == "1" {
			value, ok := stats["max_cpu_cache_bytes"].(float64)
			if !ok || value != 1 {
				t.Fatalf("native cache cap was not installed: %s", data)
			}
			if stats["cpu_cache_bytes"].(float64) > 1 {
				t.Fatalf("native retained cache exceeded cap: %s", data)
			}
			if stats["cpu_cache_bypasses"].(float64) < 1 {
				t.Fatalf("oversized fill did not report bypass: %s", data)
			}
		}
		return tokens, 0
	}
	baseline, _ := run("0", false)
	for _, scoped := range []bool{false, true} {
		got, _ := run("1", scoped)
		if !reflect.DeepEqual(got, baseline) {
			t.Fatalf("cache bypass changed tokens: %v != %v", got, baseline)
		}
	}
}

func TestNativeCPUCacheScopedRefillAfterEviction(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyCfg()
	a := synthIDs(cfg.VocabSize, 12, 71)
	b := synthIDs(cfg.VocabSize, 12, 91)
	b[0] = (a[0] + 1) % cfg.VocabSize
	newPlanner := func(cap string) *InKernelPlanner {
		t.Setenv("FAK_INKERNEL_RADIX_CPU_BYTES", cap)
		p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic-cache-refill", false, nil, false)
		p.quant = false
		return p
	}
	ctx := WithPrefixCacheIdentity(context.Background(), "tenant-refill", "")
	turn := func(p *InKernelPlanner, ids []int) []int {
		var tokens []int
		_, err := p.generateReusedRecovering(ctx, ids, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool { tokens = append(tokens, id); return true })
		if err != nil {
			t.Fatal(err)
		}
		return tokens
	}
	reference := newPlanner("0")
	want := turn(reference, a)
	// Use the actual owned first-prefix payload as a one-prefix test capacity.
	raw, _ := json.Marshal(reference.tree.Stats())
	var stat map[string]any
	_ = json.Unmarshal(raw, &stat)
	bytes, ok := stat["cpu_cache_bytes"].(float64)
	if !ok || bytes <= 0 {
		t.Fatalf("retained CPU payload is not reported: %s", raw)
	}
	p := newPlanner(strconv.FormatInt(int64(bytes), 10))
	if got := turn(p, a); !reflect.DeepEqual(got, want) {
		t.Fatal("first turn changed output")
	}
	turn(p, b)
	for i := 0; i < 2; i++ {
		if got := turn(p, a); !reflect.DeepEqual(got, want) {
			t.Fatalf("scoped refill/exact hit changed output: %v != %v", got, want)
		}
	}
}
