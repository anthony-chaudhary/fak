package model

// v41_engram_placement_test.go pins the V4.1 Engram host-table placement
// contract (v41_engram_placement.go): closed layout select, fail-closed
// admission accounting for a resident table distinct from the file-backed row
// cache, and the huge-page-backing check parsed from smaps-format evidence.
// It asserts values and sentinel errors (errors.Is), never error prose.

import (
	"errors"
	"math"
	"testing"
)

func TestEngramPlacementLayoutSelect(t *testing.T) {
	cases := []struct {
		spelling string
		want     V41EngramHostTableLayout
	}{
		{"", V41EngramHostTableShared}, // default
		{"shared", V41EngramHostTableShared},
		{"per_rank", V41EngramHostTableSharded},
	}
	for _, tc := range cases {
		got, err := ParseV41EngramHostTableLayout(tc.spelling)
		if err != nil {
			t.Fatalf("Parse(%q): unexpected error %v", tc.spelling, err)
		}
		if got != tc.want {
			t.Fatalf("Parse(%q) = %s, want %s", tc.spelling, got.String(), tc.want.String())
		}
	}

	for _, bad := range []string{"auto", "SHARED", "Shared", "per-rank", "host_sharded", " shared", "shared ", "" + "\t"} {
		_, err := ParseV41EngramHostTableLayout(bad)
		if !errors.Is(err, ErrV41EngramHostTableUnknownLayout) {
			t.Fatalf("Parse(%q) = %v, want ErrV41EngramHostTableUnknownLayout", bad, err)
		}
	}
}

func TestEngramPlacementAdmissionAccounting(t *testing.T) {
	const table = int64(189 << 30) // ~189 GiB published Engram table extent

	// Shared: one copy, no lookup reduction.
	p, err := SelectV41EngramHostTablePlacement(V41EngramHostTableRequest{
		Layout: "shared", TableBytes: table, Ranks: 8, HostBudget: table,
	})
	if err != nil {
		t.Fatalf("shared admission: %v", err)
	}
	if p.Layout != V41EngramHostTableShared || p.Copies != 1 || p.Resident != table || p.LookupReduce {
		t.Fatalf("shared placement = %+v, want copies=1 resident=%d lookup-reduce=false", p, table)
	}

	// Sharded: one private shard per rank; lookups must reduce across ranks.
	p, err = SelectV41EngramHostTablePlacement(V41EngramHostTableRequest{
		Layout: "per_rank", TableBytes: table, Ranks: 8, HostBudget: 8 * table,
	})
	if err != nil {
		t.Fatalf("sharded admission: %v", err)
	}
	if p.Layout != V41EngramHostTableSharded || p.Copies != 8 || p.Resident != 8*table || !p.LookupReduce {
		t.Fatalf("sharded placement = %+v, want copies=8 resident=%d lookup-reduce=true", p, 8*table)
	}

	// Fail-closed: the full sharded charge does not fit the budget.
	_, err = SelectV41EngramHostTablePlacement(V41EngramHostTableRequest{
		Layout: "per_rank", TableBytes: table, Ranks: 8, HostBudget: 7*table + 1,
	})
	if !errors.Is(err, ErrV41EngramHostTableNotAdmitted) {
		t.Fatalf("over-budget sharded = %v, want ErrV41EngramHostTableNotAdmitted", err)
	}

	// Geometry refusals.
	geometry := []V41EngramHostTableRequest{
		{Layout: "shared", TableBytes: 0, Ranks: 1, HostBudget: 1},
		{Layout: "shared", TableBytes: -1, Ranks: 1, HostBudget: 1},
		{Layout: "shared", TableBytes: 1, Ranks: 0, HostBudget: 1},
		{Layout: "shared", TableBytes: 1, Ranks: 1, HostBudget: -1},
		{Layout: "per_rank", TableBytes: math.MaxInt64/2 + 1, Ranks: 2, HostBudget: math.MaxInt64},
	}
	for i, req := range geometry {
		_, err := SelectV41EngramHostTablePlacement(req)
		if !errors.Is(err, ErrV41EngramHostTableGeometry) {
			t.Fatalf("geometry case %d (%+v) = %v, want ErrV41EngramHostTableGeometry", i, req, err)
		}
	}

	// Unknown layout is refused before any accounting.
	_, err = SelectV41EngramHostTablePlacement(V41EngramHostTableRequest{
		Layout: "auto", TableBytes: table, Ranks: 1, HostBudget: table,
	})
	if !errors.Is(err, ErrV41EngramHostTableUnknownLayout) {
		t.Fatalf("unknown layout = %v, want ErrV41EngramHostTableUnknownLayout", err)
	}
}

// engramSmapsFixture renders a minimal /proc/self/smaps document: two VMAs,
// the first containing the probed address.
const engramSmapsFixture = `
7f0000000000-7f0000010000 rw-p 00000000 00:00 0
Size:               64 kB
Rss:                64 kB
AnonHugePages:         0 kB
VmFlags: rd wr mr mw me sd
7f1000000000-7f1000400000 rw-p 00000000 00:00 0
Size:            4194304 kB
Rss:             1048576 kB
AnonHugePages:    1000000 kB
ShmemPmdMapped:      48576 kB
FilePmdMapped:           0 kB
Private_Dirty:       1024 kB
VmFlags: rd wr mr mw me sd
7f2000000000-7f2000001000 r-xp 00000000 00:00 0
Size:                 4 kB
Rss:                  4 kB
AnonHugePages:          4 kB
VmFlags: rd mr me
`

func TestEngramPlacementHugePageProbe(t *testing.T) {
	const addr = uint64(0x7f1000000000) + 0x1234

	probe, err := ProbeV41EngramHugePages(engramSmapsFixture, addr)
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	// Rss of the containing VMA; huge fields summed; later-VMA fields excluded.
	if probe.MappedKB != 1048576 {
		t.Fatalf("MappedKB = %d, want 1048576", probe.MappedKB)
	}
	if probe.HugeKB != 1000000+48576 {
		t.Fatalf("HugeKB = %d, want %d", probe.HugeKB, 1000000+48576)
	}
	// The selected VMA is fully huge-backed: Rss equals the sum of its
	// AnonHugePages + ShmemPmdMapped + FilePmdMapped fields, so the ratio is
	// exactly 1.
	if got := probe.Ratio(); got != 1 {
		t.Fatalf("Ratio = %v, want 1", got)
	}

	// Boundary addresses of the containing VMA are inside it.
	if _, err := ProbeV41EngramHugePages(engramSmapsFixture, 0x7f1000000000); err != nil {
		t.Fatalf("probe at VMA start: %v", err)
	}
	// One past the end is outside; so is an address between the VMAs.
	for _, out := range []uint64{0x7f1000400000, 0x7f1ffffffffff} {
		if _, err := ProbeV41EngramHugePages(engramSmapsFixture, out); !errors.Is(err, ErrV41EngramHugePageProbe) {
			t.Fatalf("probe(0x%x) = %v, want ErrV41EngramHugePageProbe", out, err)
		}
	}

	// Malformed recognized field fails closed.
	bad := "7f1000000000-7f1000400000 rw-p 00000000 00:00 0\nRss: notanumber kB\n"
	if _, err := ProbeV41EngramHugePages(bad, 0x7f1000000000); !errors.Is(err, ErrV41EngramHugePageProbe) {
		t.Fatalf("malformed Rss = %v, want ErrV41EngramHugePageProbe", err)
	}
}

func TestEngramPlacementHugePageBacking(t *testing.T) {
	// Sharded threshold boundary: 98% passes, one page below fails.
	atThreshold := V41EngramHugePageProbe{MappedKB: 1000, HugeKB: 980}
	ok, err := atThreshold.HugePageBacked(V41EngramHostTableSharded)
	if err != nil || !ok {
		t.Fatalf("98%% backing = (%v, %v), want (true, nil)", ok, err)
	}
	below := V41EngramHugePageProbe{MappedKB: 1000, HugeKB: 979}
	ok, err = below.HugePageBacked(V41EngramHostTableSharded)
	if err != nil || ok {
		t.Fatalf("97.9%% backing = (%v, %v), want (false, nil)", ok, err)
	}

	// Shared is advisory: zero huge pages still reports without refusal.
	sharedZero := V41EngramHugePageProbe{MappedKB: 1000, HugeKB: 0}
	ok, err = sharedZero.HugePageBacked(V41EngramHostTableShared)
	if err != nil || !ok {
		t.Fatalf("shared zero-huge = (%v, %v), want (true, nil)", ok, err)
	}

	// No resident pages is never backed, under either layout.
	empty := V41EngramHugePageProbe{}
	for _, layout := range []V41EngramHostTableLayout{V41EngramHostTableShared, V41EngramHostTableSharded} {
		if _, err := empty.HugePageBacked(layout); !errors.Is(err, ErrV41EngramHugePageBacking) {
			t.Fatalf("empty probe (%s) = %v, want ErrV41EngramHugePageBacking", layout.String(), err)
		}
	}
}

func TestEngramPlacementBindingExtent(t *testing.T) {
	rows := 100
	binding := V41EngramArtifactBinding{
		Shard: "engram-shard-0", Offset: 0, Size: int64(rows) * V41EngramPackedRowBytes,
		Rows: rows, Digest: "sha256:" + "00", RowBytes: V41EngramPackedRowBytes,
	}
	ext, err := binding.HostTableExtent()
	if err != nil {
		t.Fatalf("valid binding extent: %v", err)
	}
	if ext != binding.Size {
		t.Fatalf("extent = %d, want %d", ext, binding.Size)
	}

	// A size/rows disagreement is refused before placement accounting.
	bad := binding
	bad.Size++
	if _, err := bad.HostTableExtent(); err == nil {
		t.Fatal("mismatched binding extent: expected refusal")
	}

	req, err := V41EngramHostTableRequestFromBinding(binding, "per_rank", 4, 4*binding.Size)
	if err != nil {
		t.Fatalf("request from binding: %v", err)
	}
	if req.TableBytes != binding.Size || req.Ranks != 4 || req.HostBudget != 4*binding.Size {
		t.Fatalf("request = %+v, want table=%d ranks=4 budget=%d", req, binding.Size, 4*binding.Size)
	}
	p, err := SelectV41EngramHostTablePlacement(req)
	if err != nil {
		t.Fatalf("binding-derived admission: %v", err)
	}
	if p.Copies != 4 || p.Resident != 4*binding.Size || !p.LookupReduce {
		t.Fatalf("binding-derived placement = %+v", p)
	}
}
