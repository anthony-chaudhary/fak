package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// cluster.go — `fak cluster`, the first RUNNABLE multi-node compute witness.
//
// internal/model ships DistComm (dist_collective.go): a real cross-process
// coordinator-rooted process group that performs AllReduceSum / AllGather over a
// TCP wire, each rank holding ONLY its own part, proven byte-identical to the
// in-process LocalCollective. But that proof lives entirely inside a test that
// spins the ranks as goroutines over a loopback socket on ONE box — there is no
// command an operator can launch on two SEPARATE machines (two laptops, two cloud
// nodes, two GPU boxes) to make fak compute actually cross a machine boundary.
//
// `fak cluster` is that command. It is a thin shell over DistComm: the kernel
// primitive is unchanged and still carries its own bit-exactness gates; this verb
// only binds it to a real listener/dialer and a CLI so the collective runs between
// processes that the operator places wherever they like.
//
//	# self-check on one box (loopback, asserts bit-exact vs LocalCollective):
//	fak cluster selftest
//
//	# a real two-node all-reduce — run on node A (the coordinator):
//	fak cluster coordinator --listen 0.0.0.0:7777 --size 2 --vec 1,2,3
//	# ...and on node B (worker rank 1), pointing at node A's address:
//	fak cluster worker --coord A.B.C.D:7777 --rank 1 --size 2 --vec 4,5,6
//	# both print 5,7,9 — the element-wise sum, reduced across the wire.
//
// HONESTY. This is a cross-PROCESS / cross-NODE collective over HOST float32. It is
// NOT multi-GPU and NOT NCCL: the device-tensor collective (a non-cpu-ref
// compute.CollectiveBackend over NCCL/RCCL) is the separate, GPU-gated rung. What
// `fak cluster` proves is the distributed plumbing ABOVE the device line — rank
// coordination, the wire protocol, rank-order reduction across real machines, the
// fail-closed contract — runnable today on any two CPU hosts. The path from here to
// high performance is the rung ladder in docs/serving/multi-node-compute.md.
func cmdCluster(args []string) {
	if len(args) == 0 {
		clusterUsage(os.Stderr)
		os.Exit(2)
	}
	switch args[0] {
	case "selftest":
		cmdClusterSelftest(args[1:])
	case "coordinator":
		cmdClusterCoordinator(args[1:])
	case "worker":
		cmdClusterWorker(args[1:])
	case "-h", "--help", "help":
		clusterUsage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "fak cluster: unknown subcommand %q\n", args[0])
		clusterUsage(os.Stderr)
		os.Exit(2)
	}
}

func clusterUsage(w *os.File) {
	fmt.Fprint(w, `usage: fak cluster <subcommand>

Run a real cross-node collective over fak's DistComm process group (host float32).

subcommands:
  selftest       spin the ranks over a loopback socket on this box and assert the
                 result is bit-exact vs the in-process LocalCollective reference
  coordinator    rank 0: listen, accept the workers, run the collective
  worker         rank r>=1: dial the coordinator, join, run the collective

common flags (coordinator/worker):
  --size N       total ranks in the group (>=1)
  --op OP        allreduce (sum, default) | allgather (rank-ordered concat)
  --vec a,b,c    this rank's float32 part (comma-separated)
  --widths w0,.. allgather only: every rank's part width in rank order (the shared
                 tiling; this rank's --vec must be exactly --widths[rank] long)

coordinator flags:
  --listen ADDR  bind address (default 0.0.0.0:7777)

worker flags:
  --coord ADDR   the coordinator's address to dial (host:port)
  --rank R       this worker's rank in [1,size)
  --timeout DUR  how long to keep retrying the dial (default 30s)
`)
}

// parseVec parses a comma-separated float32 vector. An empty string is the empty
// vector (a valid zero-width part). It fails closed on a non-numeric field.
func parseVec(s string) ([]float32, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []float32{}, nil
	}
	fields := strings.Split(s, ",")
	v := make([]float32, len(fields))
	for i, f := range fields {
		x, err := strconv.ParseFloat(strings.TrimSpace(f), 32)
		if err != nil {
			return nil, fmt.Errorf("field %d %q: %w", i, f, err)
		}
		v[i] = float32(x)
	}
	return v, nil
}

// parseWidths parses a comma-separated list of non-negative shard widths.
func parseWidths(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("empty widths")
	}
	fields := strings.Split(s, ",")
	w := make([]int, len(fields))
	for i, f := range fields {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			return nil, fmt.Errorf("width %d %q: %w", i, f, err)
		}
		if n < 0 {
			return nil, fmt.Errorf("width %d = %d, want >= 0", i, n)
		}
		w[i] = n
	}
	return w, nil
}

// planFromWidths builds a validated TPPlan whose shard r has width widths[r], in
// rank order — the shared tiling every rank passes to AllGather.
func planFromWidths(widths []int) (model.TPPlan, error) {
	shards := make([]model.TPShard, len(widths))
	lo := 0
	for r, w := range widths {
		shards[r] = model.TPShard{Rank: r, Lo: lo, Hi: lo + w}
		lo += w
	}
	p := model.TPPlan{Dim: lo, Shards: shards}
	if err := p.Validate(); err != nil {
		return model.TPPlan{}, err
	}
	return p, nil
}

// formatVec renders a float32 vector as a comma-separated line for stdout, so a
// caller can diff two nodes' output or pipe it onward.
func formatVec(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = strconv.FormatFloat(float64(x), 'g', -1, 32)
	}
	return strings.Join(parts, ",")
}

// runCollective runs one collective on g for the given op and prints the result.
func runCollective(g *model.DistComm, op string, myPart []float32, widths []int) error {
	var (
		out []float32
		err error
	)
	switch op {
	case "allreduce":
		out, err = g.AllReduceSum(myPart)
	case "allgather":
		if widths == nil {
			return fmt.Errorf("allgather needs --widths (the shared per-rank tiling)")
		}
		plan, perr := planFromWidths(widths)
		if perr != nil {
			return fmt.Errorf("build plan: %w", perr)
		}
		if g.Rank() < len(widths) && len(myPart) != widths[g.Rank()] {
			return fmt.Errorf("rank %d --vec has %d elements, want --widths[%d] = %d", g.Rank(), len(myPart), g.Rank(), widths[g.Rank()])
		}
		out, err = g.AllGather(myPart, plan)
	default:
		return fmt.Errorf("unknown --op %q (want allreduce|allgather)", op)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "fak cluster: rank %d/%d %s ok (%d elements)\n", g.Rank(), g.Size(), op, len(out))
	fmt.Println(formatVec(out))
	return nil
}

func cmdClusterCoordinator(args []string) {
	fs := flag.NewFlagSet("cluster coordinator", flag.ExitOnError)
	listen := fs.String("listen", "0.0.0.0:7777", "bind address (host:port)")
	size := fs.Int("size", 0, "total ranks in the group (>=1)")
	op := fs.String("op", "allreduce", "collective: allreduce | allgather")
	vec := fs.String("vec", "", "this rank's float32 part (comma-separated)")
	widthsStr := fs.String("widths", "", "allgather only: every rank's part width, rank order")
	_ = fs.Parse(args)

	if *size < 1 {
		fmt.Fprintln(os.Stderr, "fak cluster coordinator: --size must be >= 1")
		os.Exit(2)
	}
	myPart, widths := parseClusterParts("coordinator", *vec, *widthsStr, *op, *size)

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster coordinator: listen %s: %v\n", *listen, err)
		os.Exit(1)
	}
	defer ln.Close()
	fmt.Fprintf(os.Stderr, "fak cluster: coordinator listening on %s, waiting for %d worker(s)...\n", ln.Addr(), *size-1)

	g, err := model.Coordinate(ln, *size)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster coordinator: coordinate: %v\n", err)
		os.Exit(1)
	}
	defer g.Close()

	if err := runCollective(g, *op, myPart, widths); err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster coordinator: %v\n", err)
		os.Exit(1)
	}
}

func cmdClusterWorker(args []string) {
	fs := flag.NewFlagSet("cluster worker", flag.ExitOnError)
	coord := fs.String("coord", "", "coordinator address to dial (host:port)")
	rank := fs.Int("rank", 0, "this worker's rank in [1,size)")
	size := fs.Int("size", 0, "total ranks in the group (>=2)")
	op := fs.String("op", "allreduce", "collective: allreduce | allgather")
	vec := fs.String("vec", "", "this rank's float32 part (comma-separated)")
	widthsStr := fs.String("widths", "", "allgather only: every rank's part width, rank order")
	timeout := fs.Duration("timeout", 30*time.Second, "how long to keep retrying the dial")
	_ = fs.Parse(args)

	if *coord == "" {
		fmt.Fprintln(os.Stderr, "fak cluster worker: --coord is required")
		os.Exit(2)
	}
	if *size < 2 {
		fmt.Fprintln(os.Stderr, "fak cluster worker: --size must be >= 2 (rank 0 coordinates)")
		os.Exit(2)
	}
	if *rank < 1 || *rank >= *size {
		fmt.Fprintf(os.Stderr, "fak cluster worker: --rank %d out of [1,%d)\n", *rank, *size)
		os.Exit(2)
	}
	myPart, widths := parseClusterParts("worker", *vec, *widthsStr, *op, *size)

	conn, err := dialCoordinator(*coord, *timeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster worker: dial %s: %v\n", *coord, err)
		os.Exit(1)
	}
	g, err := model.Join(conn, *rank, *size)
	if err != nil {
		conn.Close()
		fmt.Fprintf(os.Stderr, "fak cluster worker: join: %v\n", err)
		os.Exit(1)
	}
	defer g.Close()
	fmt.Fprintf(os.Stderr, "fak cluster: worker rank %d/%d joined coordinator %s\n", *rank, *size, *coord)

	if err := runCollective(g, *op, myPart, widths); err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster worker: %v\n", err)
		os.Exit(1)
	}
}

// parseClusterParts parses this rank's --vec float32 part and, for allgather, every rank's
// --widths (validated against size). label names the subcommand in error messages
// ("coordinator" / "worker"); a malformed flag prints to stderr and exits 2.
func parseClusterParts(label, vec, widthsStr, op string, size int) ([]float32, []int) {
	myPart, err := parseVec(vec)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster %s: bad --vec: %v\n", label, err)
		os.Exit(2)
	}
	var widths []int
	if op == "allgather" {
		widths, err = parseWidths(widthsStr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak cluster %s: bad --widths: %v\n", label, err)
			os.Exit(2)
		}
		if len(widths) != size {
			fmt.Fprintf(os.Stderr, "fak cluster %s: --widths has %d entries, want --size = %d\n", label, len(widths), size)
			os.Exit(2)
		}
	}
	return myPart, widths
}

// dialCoordinator retries net.Dial until the coordinator is reachable or the
// deadline passes — a worker is routinely launched before the coordinator is up,
// so a single dial would race-lose. Each attempt has its own short connect timeout.
func dialCoordinator(addr string, timeout time.Duration) (net.Conn, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("gave up after %s: %w", timeout, lastErr)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func cmdClusterSelftest(args []string) {
	fs := flag.NewFlagSet("cluster selftest", flag.ExitOnError)
	maxSize := fs.Int("size", 4, "test rank counts 1..size over loopback")
	n := fs.Int("len", 17, "vector length per rank (a non-round length exercises the tail)")
	_ = fs.Parse(args)

	if *maxSize < 1 {
		fmt.Fprintln(os.Stderr, "fak cluster selftest: --size must be >= 1")
		os.Exit(2)
	}
	if *n < 1 {
		fmt.Fprintln(os.Stderr, "fak cluster selftest: --len must be >= 1")
		os.Exit(2)
	}
	if err := clusterSelftest(*maxSize, *n); err != nil {
		fmt.Fprintf(os.Stderr, "fak cluster selftest: FAIL: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("PASS: cross-process allreduce + allgather bit-exact vs LocalCollective for sizes 1..%d (max|Δ|=0)\n", *maxSize)
}

// clusterSelftest runs, for each rank count in 1..maxSize, a real cross-process
// AllReduceSum and AllGather over a loopback socket (the ranks as goroutines, each
// holding only its own part) and asserts every rank's result is byte-for-byte equal
// to the in-process LocalCollective reference. It is the runnable, hardware-free
// witness that the two-node path is correct before it is launched on two machines:
// the wire and the orchestration here are identical to the cross-host case; only the
// loopback address differs from a real interface.
func clusterSelftest(maxSize, n int) error {
	for size := 1; size <= maxSize; size++ {
		// Deterministic parts: rank r holds [r*n+1, r*n+2, ...]. No RNG (Math.random
		// is unavailable to the kernel's determinism contract anyway), and distinct
		// per rank so a dropped/reordered/double-counted rank changes the result.
		parts := make([][]float32, size)
		for r := 0; r < size; r++ {
			parts[r] = make([]float32, n)
			for i := 0; i < n; i++ {
				parts[r][i] = float32(r*n + i + 1)
			}
		}

		// AllReduceSum.
		wantReduce, err := model.LocalCollective{}.AllReduceSum(parts)
		if err != nil {
			return fmt.Errorf("size=%d local allreduce: %w", size, err)
		}
		gotReduce, errs := runLoopbackGroup(size, func(g *model.DistComm) ([]float32, error) {
			return g.AllReduceSum(parts[g.Rank()])
		})
		for r := 0; r < size; r++ {
			if errs[r] != nil {
				return fmt.Errorf("size=%d rank %d allreduce: %w", size, r, errs[r])
			}
			if err := assertBitExact(gotReduce[r], wantReduce); err != nil {
				return fmt.Errorf("size=%d rank %d allreduce: %w", size, r, err)
			}
		}

		// AllGather (near-even tiling of size*n into `size` shards).
		plan, err := model.NewTPPlan(size*n, size)
		if err != nil {
			return fmt.Errorf("size=%d plan: %w", size, err)
		}
		gParts := make([][]float32, size)
		for r := 0; r < size; r++ {
			gParts[r] = make([]float32, plan.Shards[r].Width())
			for i := range gParts[r] {
				gParts[r][i] = float32(plan.Shards[r].Lo + i + 1)
			}
		}
		wantGather, err := model.LocalCollective{}.AllGather(gParts, plan)
		if err != nil {
			return fmt.Errorf("size=%d local allgather: %w", size, err)
		}
		gotGather, errs := runLoopbackGroup(size, func(g *model.DistComm) ([]float32, error) {
			return g.AllGather(gParts[g.Rank()], plan)
		})
		for r := 0; r < size; r++ {
			if errs[r] != nil {
				return fmt.Errorf("size=%d rank %d allgather: %w", size, r, errs[r])
			}
			if err := assertBitExact(gotGather[r], wantGather); err != nil {
				return fmt.Errorf("size=%d rank %d allgather: %w", size, r, err)
			}
		}
	}
	return nil
}

// assertBitExact fails unless got == want bit-for-bit — the max|Δ|=0 bar a reorder,
// drop, or double-count of a rank over the wire would break.
func assertBitExact(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("[%d] = %v, want %v (not bit-exact vs LocalCollective)", i, got[i], want[i])
		}
	}
	return nil
}

// runLoopbackGroup spins a size-rank DistComm over a loopback TCP listener, runs fn
// on every rank concurrently (rank 0 coordinates; ranks 1.. dial in and Join), and
// returns each rank's (result, error) in rank order. It mirrors the idiom DistComm's
// own tests use — a genuine cross-process exchange on one box — so the selftest
// exercises the same code the two-node launch does.
func runLoopbackGroup(size int, fn func(g *model.DistComm) ([]float32, error)) ([][]float32, []error) {
	results := make([][]float32, size)
	errs := make([]error, size)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		errs[0] = err
		return results, errs
	}
	defer ln.Close()
	addr := ln.Addr().String()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		g, cerr := model.Coordinate(ln, size)
		if cerr != nil {
			errs[0] = cerr
			return
		}
		defer g.Close()
		results[0], errs[0] = fn(g)
	}()

	for r := 1; r < size; r++ {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			conn, derr := net.Dial("tcp", addr)
			if derr != nil {
				errs[rank] = derr
				return
			}
			g, jerr := model.Join(conn, rank, size)
			if jerr != nil {
				conn.Close()
				errs[rank] = jerr
				return
			}
			defer g.Close()
			results[rank], errs[rank] = fn(g)
		}(r)
	}
	wg.Wait()
	return results, errs
}
