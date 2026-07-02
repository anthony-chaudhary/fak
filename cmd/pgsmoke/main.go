//go:build cuda && nccl

// Command pgsmoke is a minimal multi-PROCESS witness for compute.ProcessGroupBackend
// (internal/compute/cuda_collective_pg.go, #971 follow-on) — the piece cuda_collective_pg.go
// itself documents as "unverified on a GPU-free host ... 2+ processes on distinct GPUs".
//
// Rank 0 mints a bootstrap ID via ProcessGroupUniqueID and drops it in a shared file; every
// other rank polls for that file. Each rank then InitProcessGroup's on its own device,
// AllReduceSumPG's a tensor filled with its own constant, and checks the device result
// against a caller-supplied -expect sum bit-exactly (integer-valued f32 sums are exact).
//
// Usage (2 ranks, devices 1 and 2, values 1 and 2, expected sum 3):
//
//	pgsmoke -rank 0 -world 2 -device 1 -val 1 -expect 3 -id /tmp/pgsmoke.id &
//	pgsmoke -rank 1 -world 2 -device 2 -val 2 -expect 3 -id /tmp/pgsmoke.id
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func main() {
	rank := flag.Int("rank", 0, "this process's rank")
	world := flag.Int("world", 2, "world size")
	device := flag.Int("device", 0, "CUDA device index for this process")
	idPath := flag.String("id", "/tmp/pgsmoke.id", "shared file rank 0 publishes its NCCL bootstrap id to")
	n := flag.Int("n", 4096, "tensor element count")
	val := flag.Float64("val", 1, "constant value this rank's tensor is filled with")
	expect := flag.Float64("expect", 0, "expected all-reduce sum across every rank (bit-exact check)")
	waitTimeout := flag.Duration("wait", 60*time.Second, "how long a non-zero rank waits for the bootstrap id file")
	flag.Parse()

	runtime.LockOSThread()

	be := compute.Pick("cuda")
	if be == nil {
		fmt.Fprintln(os.Stderr, "pgsmoke: no cuda backend registered (no reachable CUDA device)")
		os.Exit(2)
	}
	pg, ok := be.(compute.ProcessGroupBackend)
	if !ok {
		fmt.Fprintln(os.Stderr, "pgsmoke: cuda backend does not implement ProcessGroupBackend")
		os.Exit(2)
	}

	var id []byte
	if *rank == 0 {
		var err error
		id, err = pg.ProcessGroupUniqueID()
		if err != nil {
			fmt.Fprintf(os.Stderr, "pgsmoke: ProcessGroupUniqueID: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(*idPath, id, 0o600); err != nil {
			fmt.Fprintf(os.Stderr, "pgsmoke: write id file: %v\n", err)
			os.Exit(1)
		}
	} else {
		deadline := time.Now().Add(*waitTimeout)
		for {
			b, err := os.ReadFile(*idPath)
			if err == nil && len(b) == 128 {
				id = b
				break
			}
			if time.Now().After(deadline) {
				fmt.Fprintf(os.Stderr, "pgsmoke: rank %d timed out waiting for id file %s\n", *rank, *idPath)
				os.Exit(1)
			}
			time.Sleep(200 * time.Millisecond)
		}
	}

	if err := pg.InitProcessGroup(id, *world, *rank, *device); err != nil {
		fmt.Fprintf(os.Stderr, "pgsmoke: InitProcessGroup: %v\n", err)
		os.Exit(1)
	}
	defer pg.DestroyProcessGroup()

	data := make([]float32, *n)
	for i := range data {
		data[i] = float32(*val)
	}
	host := compute.NewF32(be, []int{*n}, data)
	dev := be.Upload(host, compute.F32)

	out, err := pg.AllReduceSumPG(dev)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pgsmoke: AllReduceSumPG: %v\n", err)
		os.Exit(1)
	}

	result := be.Read(out)
	want := float32(*expect)
	bad := 0
	for i, v := range result {
		if v != want {
			bad++
			if bad <= 3 {
				fmt.Fprintf(os.Stderr, "pgsmoke: rank %d result[%d]=%v != want=%v\n", *rank, i, v, want)
			}
		}
	}
	pass := bad == 0
	fmt.Printf("PGSMOKE rank=%d world=%d device=%d n=%d got=%v want=%v pass=%v\n", *rank, *world, *device, *n, result[0], want, pass)
	if !pass {
		os.Exit(1)
	}
}
