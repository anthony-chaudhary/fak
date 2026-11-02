package model

import (
	"math"
	"reflect"
	"strconv"
	"testing"
)

func densePrefixFixture(n int) *KVCache {
	cfg := Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 4}
	c := NewKVCache(cfg)
	c.pos = make([]int, n)
	c.lineage.ids = make([]uint32, n)
	for i := 0; i < n; i++ {
		c.pos[i] = 100 + i
		c.lineage.ids[i] = uint32(i + 7)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for plane, rows := range [][][]float32{c.K, c.V, c.Kraw} {
			rows[l] = make([]float32, n*8)
			for i := range rows[l] {
				rows[l][i] = float32((l+1)*1000 + plane*100 + i)
			}
			if n > 0 {
				rows[l][0] = math.Float32frombits(0x7fc01234)
				rows[l][1] = math.Float32frombits(0x80000000)
			}
		}
	}
	c.lineage.fault = "retained diagnostic"
	return c
}

func assertDensePrefixEqual(t *testing.T, a, b *KVCache) {
	t.Helper()
	if a == nil || b == nil || !reflect.DeepEqual(a.cfg, b.cfg) || a.prec != b.prec ||
		!reflect.DeepEqual(a.pos, b.pos) || !reflect.DeepEqual(a.lineage, b.lineage) {
		t.Fatal("configuration/position/precision/lineage mismatch")
	}
	for plane, rows := range [][][]float32{a.K, a.V, a.Kraw} {
		other := [][][]float32{b.K, b.V, b.Kraw}[plane]
		if len(rows) != len(other) {
			t.Fatal("layer count mismatch")
		}
		for l := range rows {
			if len(rows[l]) != len(other[l]) {
				t.Fatal("row length mismatch")
			}
			for i, x := range rows[l] {
				if math.Float32bits(x) != math.Float32bits(other[l][i]) {
					t.Fatalf("plane%d layer%d element%d bits differ", plane, l, i)
				}
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestDensePrefixSnapshotOwnership(t *testing.T) {
	src := densePrefixFixture(33)
	plan, ok := PlanDensePrefixCapture(src, nil)
	if !ok {
		t.Fatal("dense source refused")
	}
	root := plan.Capture()
	defer root.Free()
	if plan.Capture() != nil {
		t.Fatal("capture plan consumed twice")
	}
	var census DensePrefixAccounting
	if census.Add(root) > plan.AdditionalBytes() {
		t.Fatal("retained allocation exceeded preflight")
	}
	assertDensePrefixEqual(t, src, root.Materialize())
	prefix := root.ForkPrefix(17)
	defer prefix.Free()
	branchSource := densePrefixFixture(35)
	branchPlan, ok := PlanDensePrefixCapture(branchSource, prefix)
	if !ok || branchPlan.prefix != prefix {
		t.Fatal("exact prefix was not shared")
	}
	branch := branchPlan.Capture()
	defer branch.Free()
	pool := root.seq.pool
	if branch.seq.pool != pool || pool.PhysicalBlocks() != 5 {
		t.Fatalf("full page sharing/partial-tail COW: physical=%d, want5", pool.PhysicalBlocks())
	}
	assertDensePrefixEqual(t, src, root.Materialize())
	assertDensePrefixEqual(t, branchSource, branch.Materialize())
	private := prefix.Materialize()
	private.K[0][0] = 42
	if math.Float32bits(prefix.Materialize().K[0][0]) != 0x7fc01234 {
		t.Fatal("materialization aliases retained page")
	}
	src.K[0][2] = 99
	if root.Materialize().K[0][2] == 99 {
		t.Fatal("capture aliases original session")
	}
	root.Free()
	assertDensePrefixEqual(t, branchSource, branch.Materialize())
	prefix.Free()
	branch.Free()
	branch.Free()
	if pool.PhysicalBlocks() != 0 || pool.MisuseReleases() != 0 {
		t.Fatal("snapshot release leaked or double-released references")
	}
	for _, block := range pool.blocks {
		if block != nil {
			t.Fatal("zero-reference backing remained retained")
		}
	}
	// The opt-in reclaimed slot can be reused without exposing the old bytes.
	fresh := pool.NewSequence()
	rows := [][]float32{make([]float32, 8), make([]float32, 8)}
	fresh.AppendRaw(rows, rows, rows)
	if len(pool.blocks[fresh.table[0]]) == 0 {
		t.Fatal("reclaimed slot was not refilled")
	}
	fresh.Free()
	legacy := NewPagedKVPoolWithRaw(Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}, 16)
	old := legacy.NewSequence()
	old.AppendRaw([][]float32{{1}}, [][]float32{{2}}, [][]float32{{3}})
	old.Free()
	if legacy.blocks[0] == nil {
		t.Fatal("legacy warm allocator changed")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestDensePrefixSnapshotAdmissionAndEmpty(t *testing.T) {
	c := densePrefixFixture(0)
	plan, ok := PlanDensePrefixCapture(c, nil)
	if !ok {
		t.Fatal("empty dense cache refused")
	}
	empty := plan.Capture()
	assertDensePrefixEqual(t, c.Clone(), empty.Materialize())
	empty.Free()
	bad := densePrefixFixture(3)
	bad.Kraw[1] = bad.Kraw[1][:2]
	if _, ok := PlanDensePrefixCapture(bad, nil); ok {
		t.Fatal("short raw plane admitted")
	}
	bad = densePrefixFixture(3)
	bad.lineage.ids = bad.lineage.ids[:1]
	if _, ok := PlanDensePrefixCapture(bad, nil); ok {
		t.Fatal("malformed lineage admitted")
	}
	bad = densePrefixFixture(3)
	bad.prec = KVPrecisionQ8_0
	if _, ok := PlanDensePrefixCapture(bad, nil); ok {
		t.Fatal("packed precision admitted")
	}
	bad = densePrefixFixture(3)
	bad.linear = &linearAttnCache{}
	if _, ok := PlanDensePrefixCapture(bad, nil); ok {
		t.Fatal("hybrid state admitted")
	}
	bad = densePrefixFixture(0)
	bad.cfg.NumKVHeads = int(^uint(0) >> 1)
	bad.cfg.HeadDim = 2
	if _, ok := PlanDensePrefixCapture(bad, nil); ok {
		t.Fatal("overflow geometry admitted")
	}
	base := densePrefixFixture(17)
	p, _ := PlanDensePrefixCapture(base, nil)
	owned := p.Capture()
	defer owned.Free()
	different := densePrefixFixture(18)
	different.K[0][3]++
	q, ok := PlanDensePrefixCapture(different, owned)
	if !ok || q.prefix != nil {
		t.Fatal("different actual prefix bytes shared")
	}
	copy := q.Capture()
	defer copy.Free()
	assertDensePrefixEqual(t, different, copy.Materialize())
}

// rawDensePrefixCapacityOracle reads backing capacities independently of all
// production accounting methods. Inputs are distinct live owner handles, not
// references to count repeatedly. This prices numeric storage reachable through
// those owners, not process RSS, allocation peaks, Go headers/maps/strings or logits.
func rawDensePrefixCapacityOracle(owners ...*DensePrefixSnapshot) (metadata, pools int64) {
	seen := make(map[*PagedKVPool]bool)
	intBytes := int64(strconv.IntSize / 8)
	for _, owner := range owners {
		if owner == nil || owner.seq == nil {
			continue
		}
		metadata += int64(cap(owner.seq.table)+cap(owner.pos))*intBytes + int64(cap(owner.lineage.ids))*4
		pool := owner.seq.pool
		if seen[pool] {
			continue
		}
		seen[pool] = true
		pools += int64(cap(pool.ref)+cap(pool.free)) * intBytes
		for _, block := range pool.blocks {
			pools += int64(cap(block)) * 4
		}
	}
	return metadata, pools
}

// Estimate is unmeasured; this fixed-size fixture performs no model inference.
// fak-test:runtime fast est=500ms lane=default
func TestDensePrefixSnapshotIndependentCapacityOracle(t *testing.T) {
	check := func(label string, owners ...*DensePrefixSnapshot) (int64, int64) {
		t.Helper()
		metadata, pools := rawDensePrefixCapacityOracle(owners...)
		var production DensePrefixAccounting
		var got int64
		for _, owner := range owners {
			got += production.Add(owner)
		}
		if got != metadata+pools {
			t.Fatalf("%s: reported=%d raw metadata=%d unique pools=%d", label, got, metadata, pools)
		}
		return metadata, pools
	}
	plan, ok := PlanDensePrefixCapture(densePrefixFixture(33), nil)
	if !ok {
		t.Fatal("root fixture refused")
	}
	root := plan.Capture()
	if root == nil {
		t.Fatal("root capture failed")
	}
	defer root.Free()
	rootMetadata, rootPools := check("root", root)
	prefix := root.ForkPrefix(17)
	if prefix == nil || prefix.seq.pool != root.seq.pool {
		t.Fatal("prefix did not share root pool")
	}
	defer prefix.Free()
	bothMetadata, bothPools := check("fork", root, prefix)
	if bothMetadata <= rootMetadata || bothPools != rootPools {
		t.Fatal("fork must add owner metadata without double-counting shared pool backing")
	}
	branchPlan, ok := PlanDensePrefixCapture(densePrefixFixture(35), prefix)
	if !ok || branchPlan.prefix != prefix {
		t.Fatal("branch did not select shared prefix")
	}
	branch := branchPlan.Capture()
	if branch == nil || branch.seq.pool != root.seq.pool {
		t.Fatal("branch lost shared pool")
	}
	defer branch.Free()
	check("partial-tail COW", root, prefix, branch)
	// A second independent pool must be charged separately, not deduplicated
	// merely because its geometry and token contents happen to match.
	otherPlan, ok := PlanDensePrefixCapture(densePrefixFixture(33), nil)
	if !ok {
		t.Fatal("independent fixture refused")
	}
	other := otherPlan.Capture()
	if other == nil || other.seq.pool == root.seq.pool {
		t.Fatal("independent capture unexpectedly shared a pool")
	}
	defer other.Free()
	_, originalPools := check("shared family", root, prefix, branch)
	_, separatePools := check("second pool", other)
	_, combinedPools := check("both pools", root, prefix, branch, other)
	if combinedPools != originalPools+separatePools {
		t.Fatal("independent pools were omitted or counted repeatedly")
	}
	other.Free()
	check("independent owner released", root, prefix, branch, other)
	pool := root.seq.pool // Deliberately keep a test-only handle after final release.
	root.Free()
	check("parent released", root, prefix, branch)
	assertDensePrefixEqual(t, densePrefixFixture(35), branch.Materialize())
	prefix.Free()
	check("prefix released", root, prefix, branch)
	assertDensePrefixEqual(t, densePrefixFixture(35), branch.Materialize())
	branch.Free()
	metadata, pools := check("last owner released", root, prefix, branch, other)
	if metadata != 0 || pools != 0 {
		t.Fatal("released handles still expose numeric ownership")
	}
	for i, block := range pool.blocks {
		if cap(block) != 0 {
			t.Fatalf("last owner release retained page %d backing", i)
		}
	}
	// The local pool pointer may still retain ref/free metadata. Zero owner
	// census is not a claim that every test/process-held allocation disappeared.
	t.Logf("test-only detached pool metadata capacity: ref=%d free=%d; excluded from live-owner census", cap(pool.ref), cap(pool.free))
}
