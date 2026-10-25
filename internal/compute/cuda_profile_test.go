package compute

import (
	"testing"
)

func cudaTSNameFixture(kid uint32) string {
	return []string{"k_q4k_gemm", "k_rmsnorm", "d2d_copy"}[kid]
}

func cudaTSFixture() (cudaTSTotals, []cudaTSRow) {
	tot := cudaTSTotals{
		Enabled: true, Usable: true, InitTried: true,
		Batches: 4, Ops: 30, UntimedTransfer: 4, GPUNS: 900, WallNS: 1200,
	}
	rows := []cudaTSRow{
		{Kid: 1, GX: 1, GY: 1, Calls: 10, NS: 100},
		{Kid: 0, GX: 2560, GY: 1, Calls: 10, NS: 700},
		{Kid: 2, Calls: 10, NS: 100},
	}
	return tot, rows
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileFromShimRows(t *testing.T) {
	tot, rows := cudaTSFixture()
	p, ok := cudaKernelProfileFromShim(tot, len(rows), rows, cudaTSNameFixture)
	if !ok {
		t.Fatal("a usable ring must read")
	}
	if p.Batches != 4 || p.Ops != 30 || p.UntimedTransfer != 4 || p.GPUNS != 900 || p.WallNS != 1200 || p.BarriersElided != 0 {
		t.Fatalf("totals = %+v", p)
	}
	want := []KernelRow{
		{Kernel: "k_q4k_gemm", GX: 2560, GY: 1, Calls: 10, NS: 700},
		{Kernel: "d2d_copy", Calls: 10, NS: 100},
		{Kernel: "k_rmsnorm", GX: 1, GY: 1, Calls: 10, NS: 100},
	}
	if len(p.Rows) != len(want) {
		t.Fatalf("rows = %+v", p.Rows)
	}
	var sum float64
	for i, r := range p.Rows {
		if r != want[i] {
			t.Fatalf("row %d = %+v, want %+v", i, r, want[i])
		}
		sum += r.NS
	}
	if sum != p.GPUNS {
		t.Fatalf("rows sum %v, GPUNS %v: event-pair rows must telescope to GPUNS", sum, p.GPUNS)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileFromShimRefusals(t *testing.T) {
	tot, rows := cudaTSFixture()
	for name, mut := range map[string]func(*cudaTSTotals) int{
		"off":           func(t *cudaTSTotals) int { t.Enabled = false; return len(rows) },
		"ring_failed":   func(t *cudaTSTotals) int { t.Usable = false; return len(rows) },
		"capture_open":  func(*cudaTSTotals) int { return -1 },
		"short_copyout": func(*cudaTSTotals) int { return len(rows) + 1 },
	} {
		tt := tot
		n := mut(&tt)
		if _, ok := cudaKernelProfileFromShim(tt, n, rows, cudaTSNameFixture); ok {
			t.Errorf("%s: read as available", name)
		}
	}
	// A ring not yet created (no launch since load) is a valid zero baseline, never unavailable.
	p, ok := cudaKernelProfileFromShim(cudaTSTotals{Enabled: true}, 0, nil, cudaTSNameFixture)
	if !ok || p.Ops != 0 || len(p.Rows) != 0 {
		t.Fatalf("zero baseline = %+v ok=%v", p, ok)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAKernelProfileRegistersThroughSeam(t *testing.T) {
	tot, rows := cudaTSFixture()
	usable := true
	read := func() (KernelProfile, bool) {
		tt := tot
		tt.Usable = usable
		return cudaKernelProfileFromShim(tt, len(rows), rows, cudaTSNameFixture)
	}
	unregister := RegisterKernelProfileReader("cuda", read)
	defer unregister()
	before := KernelProfileSnapshot()
	tot.Batches, tot.Ops, tot.GPUNS, tot.WallNS = 5, 33, 1000, 1500
	rows = append([]cudaTSRow(nil), rows...)
	rows[1].Calls, rows[1].NS = 13, 800
	after := KernelProfileSnapshot()
	if before == nil || after == nil || after.Backend != "cuda" {
		t.Fatalf("snapshots before=%+v after=%+v", before, after)
	}
	d := KernelProfileDelta(before, after)
	if d == nil || d.Backend != "cuda" || d.Ops != 3 || d.GPUNS != 100 || len(d.Rows) != 1 ||
		d.Rows[0] != (KernelRow{Kernel: "k_q4k_gemm", GX: 2560, GY: 1, Calls: 3, NS: 100}) {
		t.Fatalf("delta = %+v", d)
	}
	if got := KernelOpClass(d.Rows[0].Kernel); got != KernelOpMatmul {
		t.Fatalf("k_q4k_gemm class = %s, want matmul", got)
	}
	usable = false
	if KernelProfileSnapshot() != nil || !KernelProfileReaderRegistered() {
		t.Fatal("a failed ring must stay registered and read nil")
	}
}
