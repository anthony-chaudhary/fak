package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak-test:runtime medium est=5s lane=default
func TestInKernelSharedDensePrefixOrdinaryRoute(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	// The unbounded budget the retired FAK_INKERNEL_RADIX_BUDGET=0 spelling meant
	// is now the explicit negative RadixBudgetTokens below (#13605 leaf A-S).
	t.Setenv("FAK_INKERNEL_RADIX_CPU_BYTES", "")
	t.Setenv("FAK_INKERNEL_BATCH", "off")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	makePlanner := func() *InKernelPlanner {
		p := NewInKernelPlannerWithConfig(m, nil, "dense-prefix-c03", false, nil, false, InKernelPlannerConfig{RadixBudgetTokens: -1, CPUCacheBytes: 16 << 20})
		p.quant = false
		p.maxNew = 4
		p.temp = 0
		return p
	}
	selected := makePlanner()
	if selected.tree == nil || selected.tree.SharedDensePrefixesEnabled() != sharedDensePrefixQualification {
		t.Fatal("real planner constructor did not honor the build selection")
	}
	if !sharedDensePrefixQualification {
		t.Skip("positive ordinary-route witness requires fak_radix_dense_sharing_qualification; default remains legacy")
	}
	baseline := makePlanner()
	// The comparator deliberately uses the pre-port clone-owning tree API.
	// Both arms execute the same ordinary generateReusedContext consumer.
	baseline.tree = radixkv.New(0)
	baseline.tree.SetCPUCacheByteBudget(16 << 20)
	baseline.scopedTree = radixkv.WrapScopedWithLocker(baseline.tree, &baseline.mu)
	prefix := make([]int, 32)
	for i := range prefix {
		prefix[i] = i + 1
	}
	requests := [][]int{
		append(append([]int(nil), prefix...), 40, 41, 42),
		append(append([]int(nil), prefix...), 50, 51, 52),
		append(append([]int(nil), prefix...), 40, 41, 42),
	}
	run := func(p *InKernelPlanner, ids []int) ([]int, int) {
		var tokens []int
		_, _, matched, _, _, _, err := p.generateReusedContext(context.Background(), ids, 4, 0, 0, 0, map[int]bool{}, func(token int) bool { tokens = append(tokens, token); return false })
		if err != nil {
			t.Fatal(err)
		}
		return tokens, matched
	}
	for i, ids := range requests {
		want, wm := run(baseline, ids)
		got, gm := run(selected, ids)
		if !reflect.DeepEqual(want, got) || wm != gm {
			t.Fatalf("ordinary request%d output/match differs: baseline%v/%d shared%v/%d", i, want, wm, got, gm)
		}
		if i > 0 && gm == 0 {
			t.Fatal("fixture did not reach ordinary prefix reuse")
		}
	}
	// Positively bind the ordinary producer to the immutable store without
	// assuming that sharing wins every tiny workload. Legacy fallback returns
	// the same live KV pointer; an admitted dense owner materializes privately.
	selected.mu.Lock()
	boundary, matched := selected.tree.Lookup(requests[0])
	first, second := boundary.KV(), boundary.KV()
	selected.tree.Done(boundary)
	selected.mu.Unlock()
	if matched != len(requests[0]) || first == nil || second == nil || first == second || !reflect.DeepEqual(first, second) {
		t.Fatal("ordinary producer did not retain a privately materialized dense owner")
	}
	before := selected.tree.Stats().CPUCacheBytes
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, _, _, _, _, err := selected.generateReusedContext(ctx, prefix, 4, 0, 0, 0, map[int]bool{}, nil)
	if !errors.Is(err, context.Canceled) || selected.tree.Stats().CPUCacheBytes != before {
		t.Fatal("cancelled admission changed cache ownership")
	}
	if selected.tree.Stats().CPUCacheBytes <= 0 {
		t.Fatal("ordinary route retained no real payload")
	}
}
