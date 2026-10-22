package model

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestPagedKVPoolCountsMisuseReleases(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		setup  func(p *PagedKVPool) int
		misuse func(p *PagedKVPool, id int)
		want   int
	}{
		{
			name:   "empty_pool",
			setup:  func(*PagedKVPool) int { return 0 },
			misuse: func(p *PagedKVPool, id int) { p.Release(id) },
			want:   1,
		},
		{
			name:   "negative_id",
			setup:  func(p *PagedKVPool) int { p.Alloc(); return -1 },
			misuse: func(p *PagedKVPool, id int) { p.Release(id) },
			want:   1,
		},
		{
			name:   "out_of_range_id",
			setup:  func(p *PagedKVPool) int { p.Alloc(); return p.TotalBlocks() },
			misuse: func(p *PagedKVPool, id int) { p.Release(id) },
			want:   1,
		},
		{
			name: "both_bounds",
			setup: func(p *PagedKVPool) int {
				p.Alloc()
				return 0
			},
			misuse: func(p *PagedKVPool, _ int) {
				p.Release(-1)
				p.Release(p.TotalBlocks())
			},
			want: 2,
		},
		{
			name: "double_release",
			setup: func(p *PagedKVPool) int {
				id := p.Alloc()
				p.Release(id)
				return id
			},
			misuse: func(p *PagedKVPool, id int) { p.Release(id) },
			want:   1,
		},
		{
			name: "shared_over_release",
			setup: func(p *PagedKVPool) int {
				s := pagedFillSeq(p, 1)
				s.Fork()
				id := s.table[0]
				p.Release(id)
				p.Release(id)
				return id
			},
			misuse: func(p *PagedKVPool, id int) { p.Release(id) },
			want:   1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := NewPagedKVPool(pagedTestCfg(), 4)
			id := tc.setup(p)
			if got := p.MisuseReleases(); got != 0 {
				t.Fatalf("MisuseReleases() after legitimate setup = %d, want 0", got)
			}
			before := snapshotPool(p)
			tc.misuse(p, id)
			if got := p.MisuseReleases(); got != tc.want {
				t.Fatalf("MisuseReleases() = %d, want %d", got, tc.want)
			}
			assertPoolUnchanged(t, tc.name, p, before)

			seen := make(map[int]bool)
			for range before.nFree + 1 {
				got := p.Alloc()
				if seen[got] {
					t.Fatalf("Alloc() returned block %d twice; free list holds a duplicate", got)
				}
				seen[got] = true
			}
			if got := p.TotalBlocks(); got != before.nBlocks+1 {
				t.Fatalf("TotalBlocks() = %d, want %d", got, before.nBlocks+1)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPagedKVPoolLegitimateLifecycleCountsNoMisuse(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		pool func() *PagedKVPool
		raw  bool
	}{
		{name: "two_plane", pool: func() *PagedKVPool { return NewPagedKVPool(pagedTestCfg(), 4) }},
		{name: "raw_three_plane", pool: func() *PagedKVPool { return NewPagedKVPoolWithRaw(pagedTestCfg(), 4) }, raw: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.pool()
			s := pagedFillSeq(p, 10)
			f := s.Fork()
			k, v := pagedTokenKV(p, 10)
			f.Append(k, v)
			if got := p.PhysicalBlocks(); got != 4 {
				t.Fatalf("PhysicalBlocks() after copy-on-write append = %d, want 4", got)
			}
			s.Free()
			f.Free()
			s.Free()
			f.Free()
			p.Release(p.Alloc())
			if tc.raw {
				c := pagedFillSeq(p, 12)
				if err := PruneAndCompactPagedTreeKV(c, 3, []int{0, 3, 7}, 9); err != nil {
					t.Fatalf("PruneAndCompactPagedTreeKV: %v", err)
				}
				if got := c.Blocks(); got != 2 {
					t.Fatalf("compacted Blocks() = %d, want 2", got)
				}
				c.Free()
			}
			if got := p.MisuseReleases(); got != 0 {
				t.Fatalf("MisuseReleases() = %d, want 0", got)
			}
			if got := p.PhysicalBlocks(); got != 0 {
				t.Fatalf("PhysicalBlocks() = %d, want 0", got)
			}
			if free, total := p.FreeBlocks(), p.TotalBlocks(); free != total {
				t.Fatalf("FreeBlocks() = %d, TotalBlocks() = %d, want equal", free, total)
			}
		})
	}
}
