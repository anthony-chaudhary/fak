//go:build cuda && nccl

// Tests for pgsmoke's CLI seams: flag parsing, the bootstrap-id handshake, and the
// expected-sum check. They carry the package's cuda&&nccl tag because they exercise the
// same closed contracts main.go enforces; the pure-logic tests need no GPU, while
// BenchmarkAllReduceSumPG skips cleanly on hosts without a CUDA+NCCL toolchain.
package main

import (
	"bytes"
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestParseConfigDefaults(t *testing.T) {
	t.Parallel()
	c, err := parseConfig(nil)
	if err != nil {
		t.Fatalf("parseConfig(nil): %v", err)
	}
	want := config{Rank: 0, World: 2, Device: 0, N: 4096, Val: 1, Expect: 0, IDPath: "/tmp/pgsmoke.id", Wait: 60 * time.Second}
	if *c != want {
		t.Fatalf("defaults = %+v, want %+v", *c, want)
	}
}

func TestParseConfigExplicit(t *testing.T) {
	t.Parallel()
	c, err := parseConfig([]string{"-rank", "1", "-world", "4", "-device", "3", "-n", "128", "-val", "2.5", "-expect", "10", "-id", "/x/y.id", "-wait", "5s"})
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if c.Rank != 1 || c.World != 4 || c.Device != 3 || c.N != 128 || c.Val != 2.5 || c.Expect != 10 || c.IDPath != "/x/y.id" || c.Wait != 5*time.Second {
		t.Fatalf("parsed = %+v", *c)
	}
}

func TestParseConfigRejectsClosedContractViolations(t *testing.T) {
	t.Parallel()
	cases := map[string][]string{
		"world_below_two":   {"-world", "1"},
		"world_zero":        {"-world", "0"},
		"negative_world":    {"-world", "-2"},
		"rank_negative":     {"-rank", "-1"},
		"rank_equals_world": {"-rank", "2", "-world", "2"},
		"rank_above_world":  {"-rank", "3", "-world", "2"},
		"n_zero":            {"-n", "0"},
		"n_negative":        {"-n", "-4"},
		"wait_zero":         {"-wait", "0s"},
		"unknown_flag":      {"-bogus", "1"},
		"bad_value_type":    {"-world", "abc"},
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, err := parseConfig(args)
			if !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("parseConfig(%v) error = %v, want ErrInvalidConfig", args, err)
			}
		})
	}
}

// A help request must stay identifiable as flag.ErrHelp (so main exits 0 with usage)
// while still carrying the closed contract, so callers that branch on either sentinel
// keep working.
func TestParseConfigHelpRequest(t *testing.T) {
	t.Parallel()
	_, err := parseConfig([]string{"-h"})
	if !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("error = %v, want flag.ErrHelp in chain", err)
	}
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("error = %v, want ErrInvalidConfig in chain", err)
	}
}

func TestPublishBootstrapID(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "pgsmoke.id")
	id := bytes.Repeat([]byte{0xAB}, bootstrapIDLen)

	if err := publishBootstrapID(path, id); err != nil {
		t.Fatalf("publishBootstrapID: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if !bytes.Equal(got, id) {
		t.Fatalf("round trip = %x..., want %x...", got[:8], id[:8])
	}

	if err := publishBootstrapID(path, id[:64]); !errors.Is(err, ErrBootstrapMalformed) {
		t.Fatalf("short id error = %v, want ErrBootstrapMalformed", err)
	}
	if err := publishBootstrapID(filepath.Join(dir, "missing", "dir", "id"), id); err == nil {
		t.Fatal("publish into absent directory: want error, got nil")
	}
}

func TestWaitForBootstrapIDReceivesPublishedID(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "pgsmoke.id")
	id := bytes.Repeat([]byte{0x5A}, bootstrapIDLen)
	if err := os.WriteFile(path, id, 0o600); err != nil {
		t.Fatalf("seed id file: %v", err)
	}
	got, err := waitForBootstrapID(path, time.Second)
	if err != nil {
		t.Fatalf("waitForBootstrapID: %v", err)
	}
	if !bytes.Equal(got, id) {
		t.Fatalf("id = %x..., want %x...", got[:8], id[:8])
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestWaitForBootstrapIDTimesOutWhenAbsent(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	_, err := waitForBootstrapID(filepath.Join(dir, "never.id"), 300*time.Millisecond)
	if !errors.Is(err, ErrBootstrapTimeout) {
		t.Fatalf("error = %v, want ErrBootstrapTimeout", err)
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestWaitForBootstrapIDReportsMalformedAtDeadline(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "short.id")
	if err := os.WriteFile(path, make([]byte, 64), 0o600); err != nil {
		t.Fatalf("seed short id file: %v", err)
	}
	_, err := waitForBootstrapID(path, 300*time.Millisecond)
	if !errors.Is(err, ErrBootstrapMalformed) {
		t.Fatalf("error = %v, want ErrBootstrapMalformed", err)
	}
}

func TestCheckExpectedSum(t *testing.T) {
	t.Parallel()
	var clean bytes.Buffer
	if err := checkExpectedSum(&clean, 0, []float32{3, 3, 3}, 3); err != nil {
		t.Fatalf("clean sum: %v", err)
	}
	if clean.Len() != 0 {
		t.Fatalf("clean sum wrote diagnostics: %q", clean.String())
	}

	var diag bytes.Buffer
	err := checkExpectedSum(&diag, 1, []float32{3, 9, 3, 9, 9}, 3)
	if !errors.Is(err, ErrSumMismatch) {
		t.Fatalf("mismatched sum error = %v, want ErrSumMismatch", err)
	}
	if diag.Len() == 0 {
		t.Fatal("mismatched sum wrote no diagnostics")
	}

	// Empty result is vacuously correct: nothing reduced, nothing to disagree with.
	if err := checkExpectedSum(io.Discard, 0, nil, 3); err != nil {
		t.Fatalf("empty result: %v", err)
	}
}

// BenchmarkAllReduceSumPG measures the multi-process all-reduce path end to end on a
// single-rank group (world=1 keeps the benchmark runnable on one GPU; the collective is
// in-place, so repeated iterations reuse the same device buffer). It skips on hosts
// without a CUDA+NCCL toolchain rather than failing the suite.
func BenchmarkAllReduceSumPG(b *testing.B) {
	be := compute.Pick("cuda")
	if be == nil {
		b.Skip("no cuda backend registered (GPU-free host)")
	}
	pg, ok := be.(compute.ProcessGroupBackend)
	if !ok {
		b.Skip("cuda backend does not implement ProcessGroupBackend")
	}
	id, err := pg.ProcessGroupUniqueID()
	if err != nil {
		b.Skipf("cannot mint bootstrap id on this host: %v", err)
	}
	const world, rank, device, n = 1, 0, 0, 4096
	if err := pg.InitProcessGroup(id, world, rank, device); err != nil {
		b.Skipf("cannot form single-rank process group on this host: %v", err)
	}
	defer pg.DestroyProcessGroup()

	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	dev := be.Upload(compute.NewF32(be, []int{n}, data), compute.F32)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := pg.AllReduceSumPG(dev); err != nil {
			b.Fatalf("AllReduceSumPG iteration %d: %v", i, err)
		}
	}
}
