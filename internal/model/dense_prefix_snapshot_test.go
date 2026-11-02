package model

import (
	"math"
	"reflect"
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
