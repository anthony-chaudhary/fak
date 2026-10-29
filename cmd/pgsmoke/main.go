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
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Closed contracts for the CLI seams, asserted by main_test.go via errors.Is.
var (
	// ErrInvalidConfig reports a flag value or combination the smoke cannot run with.
	ErrInvalidConfig = errors.New("pgsmoke: invalid configuration")
	// ErrBootstrapTimeout reports that a non-zero rank never saw a usable id file appear.
	ErrBootstrapTimeout = errors.New("pgsmoke: bootstrap id timeout")
	// ErrBootstrapMalformed reports an id file that exists but is not a 128-byte NCCL unique id.
	ErrBootstrapMalformed = errors.New("pgsmoke: bootstrap id malformed")
	// ErrSumMismatch reports that the all-reduce result deviates from -expect.
	ErrSumMismatch = errors.New("pgsmoke: expected sum mismatch")
)

// bootstrapIDLen is NCCL_UNIQUE_ID_BYTES, the fixed size of the out-of-band-distributed
// bootstrap id (mirrors compute.ncclUniqueIDBytes, which is unexported).
const bootstrapIDLen = 128

// bootstrapPollInterval is how often a waiting rank re-checks the id file.
const bootstrapPollInterval = 200 * time.Millisecond

// config is the parsed, validated pgsmoke command line.
type config struct {
	Rank   int
	World  int
	Device int
	N      int
	Val    float64
	Expect float64
	IDPath string
	Wait   time.Duration
}

// flagSpec holds the registered flag targets so parseConfig and printUsage share one
// definition and can never drift apart.
type flagSpec struct {
	rank   *int
	world  *int
	device *int
	id     *string
	n      *int
	val    *float64
	expect *float64
	wait   *time.Duration
}

func registerFlags(fs *flag.FlagSet) flagSpec {
	return flagSpec{
		rank:   fs.Int("rank", 0, "this process's rank"),
		world:  fs.Int("world", 2, "world size"),
		device: fs.Int("device", 0, "CUDA device index for this process"),
		id:     fs.String("id", "/tmp/pgsmoke.id", "shared file rank 0 publishes its NCCL bootstrap id to"),
		n:      fs.Int("n", 4096, "tensor element count"),
		val:    fs.Float64("val", 1, "constant value this rank's tensor is filled with"),
		expect: fs.Float64("expect", 0, "expected all-reduce sum across every rank (bit-exact check)"),
		wait:   fs.Duration("wait", 60*time.Second, "how long a non-zero rank waits for the bootstrap id file"),
	}
}

// parseConfig parses args into a validated config. Misuse wraps ErrInvalidConfig so callers
// branch on the contract, not on message prose.
func parseConfig(args []string) (*config, error) {
	fs := flag.NewFlagSet("pgsmoke", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	p := registerFlags(fs)
	if err := fs.Parse(args); err != nil {
		// Double %w keeps flag.ErrHelp in the chain so main can still distinguish a
		// help request from genuine misuse.
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	c := &config{
		Rank:   *p.rank,
		World:  *p.world,
		Device: *p.device,
		IDPath: *p.id,
		N:      *p.n,
		Val:    *p.val,
		Expect: *p.expect,
		Wait:   *p.wait,
	}
	switch {
	case c.World < 2:
		return nil, fmt.Errorf("%w: world=%d must be >= 2", ErrInvalidConfig, c.World)
	case c.Rank < 0 || c.Rank >= c.World:
		return nil, fmt.Errorf("%w: rank=%d outside [0,%d)", ErrInvalidConfig, c.Rank, c.World)
	case c.N <= 0:
		return nil, fmt.Errorf("%w: n=%d must be > 0", ErrInvalidConfig, c.N)
	case c.Wait <= 0:
		return nil, fmt.Errorf("%w: wait=%s must be > 0", ErrInvalidConfig, c.Wait)
	}
	return c, nil
}

// printUsage renders the flag help to stderr.
func printUsage() {
	fs := flag.NewFlagSet("pgsmoke", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	registerFlags(fs)
	fmt.Fprintln(os.Stderr, "usage: pgsmoke [flags]")
	fs.PrintDefaults()
}

// publishBootstrapID writes rank 0's freshly-minted id to path. The length guard keeps a
// truncated or wrong-sized id from ever reaching a peer's InitProcessGroup.
func publishBootstrapID(path string, id []byte) error {
	if len(id) != bootstrapIDLen {
		return fmt.Errorf("%w: id is %d bytes, want %d", ErrBootstrapMalformed, len(id), bootstrapIDLen)
	}
	if err := os.WriteFile(path, id, 0o600); err != nil {
		return fmt.Errorf("write id file %s: %w", path, err)
	}
	return nil
}

// waitForBootstrapID polls path until it holds exactly one 128-byte id or timeout elapses.
// A short read is treated as a still-in-progress write and re-polled; at the deadline the
// outcome is typed as malformed (a file was present but never reached 128 bytes) or timeout
// (no usable file ever appeared).
func waitForBootstrapID(path string, timeout time.Duration) ([]byte, error) {
	deadline := time.Now().Add(timeout)
	sawMalformed := false
	for {
		b, err := os.ReadFile(path)
		if err == nil {
			if len(b) == bootstrapIDLen {
				return b, nil
			}
			sawMalformed = true
		}
		if time.Now().After(deadline) {
			if sawMalformed {
				return nil, fmt.Errorf("%w: %s never reached %d bytes within %s", ErrBootstrapMalformed, path, bootstrapIDLen, timeout)
			}
			return nil, fmt.Errorf("%w: %s not published within %s", ErrBootstrapTimeout, path, timeout)
		}
		time.Sleep(bootstrapPollInterval)
	}
}

// checkExpectedSum compares the reduced result element-by-element against want and reports
// up to three offending elements on w. Integer-valued f32 sums are bit-exact, so any
// deviation is a genuine collective failure, not floating-point noise.
func checkExpectedSum(w io.Writer, rank int, result []float32, want float32) error {
	bad := 0
	for i, v := range result {
		if v != want {
			bad++
			if bad <= 3 {
				fmt.Fprintf(w, "pgsmoke: rank %d result[%d]=%v != want=%v\n", rank, i, v, want)
			}
		}
	}
	if bad == 0 {
		return nil
	}
	return fmt.Errorf("%w: %d of %d elements differ from %v", ErrSumMismatch, bad, len(result), want)
}

func main() {
	cfg, err := parseConfig(os.Args[1:])
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			printUsage()
			return
		}
		fmt.Fprintf(os.Stderr, "pgsmoke: %v\n\n", err)
		printUsage()
		os.Exit(2)
	}

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
	if cfg.Rank == 0 {
		id, err = pg.ProcessGroupUniqueID()
		if err != nil {
			fmt.Fprintf(os.Stderr, "pgsmoke: ProcessGroupUniqueID: %v\n", err)
			os.Exit(1)
		}
		if err := publishBootstrapID(cfg.IDPath, id); err != nil {
			fmt.Fprintf(os.Stderr, "pgsmoke: %v\n", err)
			os.Exit(1)
		}
	} else {
		id, err = waitForBootstrapID(cfg.IDPath, cfg.Wait)
		if err != nil {
			fmt.Fprintf(os.Stderr, "pgsmoke: %v\n", err)
			os.Exit(1)
		}
	}

	if err := pg.InitProcessGroup(id, cfg.World, cfg.Rank, cfg.Device); err != nil {
		fmt.Fprintf(os.Stderr, "pgsmoke: InitProcessGroup: %v\n", err)
		os.Exit(1)
	}
	defer pg.DestroyProcessGroup()

	data := make([]float32, cfg.N)
	for i := range data {
		data[i] = float32(cfg.Val)
	}
	host := compute.NewF32(be, []int{cfg.N}, data)
	dev := be.Upload(host, compute.F32)

	out, err := pg.AllReduceSumPG(dev)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pgsmoke: AllReduceSumPG: %v\n", err)
		os.Exit(1)
	}

	result := be.Read(out)
	want := float32(cfg.Expect)
	pass := checkExpectedSum(os.Stderr, cfg.Rank, result, want) == nil
	fmt.Printf("PGSMOKE rank=%d world=%d device=%d n=%d got=%v want=%v pass=%v\n", cfg.Rank, cfg.World, cfg.Device, cfg.N, result[0], want, pass)
	if !pass {
		os.Exit(1)
	}
}
