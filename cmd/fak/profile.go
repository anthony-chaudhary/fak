package main

// `fak profile` -- the host-aware developer profiler. It is the thin convenience
// verb the developer-tooling guide (docs/dev-tooling.md) specs as a pair with
// `fak test`: one command that resolves the right `go test -bench` invocation to
// capture a CPU + allocation profile of a package's benchmarks, AND encodes the one
// piece of host knowledge that bites newcomers -- on Windows, native `go test` is
// blocked by an OS Application-Control policy, so the benchmark run must execute
// inside WSL. `fak profile` routes there automatically (via the repo-root test.ps1
// wrapper), then points you at the `go tool pprof` command to read the result (or
// runs `-top` for you).
//
//	fak profile ./internal/ctxmmu/        CPU+mem profile of the package's benchmarks
//	fak profile ./internal/recall/ -bench BenchmarkDigest
//	fak profile ./internal/ctxmmu/ -benchtime 2s -top   profile, then print pprof -top
//	fak profile ./internal/ctxmmu/ -n     print the resolved command without running it
//	fak profile --list                    show what the verb captures and exit
//
// The pure planProfile below resolves (host, package, flags) -> the exact command to
// exec, so the decision is testable and reproducible; cmdProfile is the impure shell
// that prints the resolved command, runs it, and optionally shells `go tool pprof`.
// Go's own pprof + the `fak benchmarks` / `fak bench` / `fak ablate` verbs remain the
// authoritative perf surfaces -- this verb is the host-routing convenience layer over
// the same `go test -cpuprofile`, not a replacement.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdProfile(argv []string) { os.Exit(runProfile(os.Stdout, os.Stderr, argv)) }

// profilePlan is the resolved, reproducible command a `fak profile` invocation will run.
type profilePlan struct {
	Pkg        string   // the package target being profiled
	GoArgs     []string // the args handed to `go test`, e.g. ["-run=^$", "-bench=.", ...]
	Argv       []string // the actual command to exec, host-resolved
	ViaWSL     bool     // true when routed through test.ps1 (Windows -> WSL)
	CPUProfile string   // the CPU profile path (so the shell can offer `go tool pprof`)
	Note       string   // a one-line human note about the routing
}

// profileOpts is the post-flag input to the pure planner.
type profileOpts struct {
	pkg        string
	bench      string
	cpuProfile string
	memProfile string
	benchtime  string
}

// planProfile is the pure resolver: given the host GOOS and the profile options, it
// returns the command to run. No I/O, no exec -- so a peer can reproduce the exact
// invocation from the same inputs, and the table test pins the routing. It builds the
// canonical `go test -run=^$ -bench=<pat> -benchmem -cpuprofile <f> -memprofile <f>
// <pkg>` line (the form the dev-tooling guide documents) and, on Windows, routes it
// through test.ps1 so the OS-policy-blocked native `go test` never gets hit.
func planProfile(goos string, o profileOpts) (profilePlan, error) {
	pkg := strings.TrimSpace(o.pkg)
	if pkg == "" {
		return profilePlan{}, fmt.Errorf("a package target is required (e.g. fak profile ./internal/ctxmmu/)")
	}
	if !looksLikePackage(pkg) {
		return profilePlan{}, fmt.Errorf("%q does not look like a Go package target (try ./internal/foo/ or an import path)", pkg)
	}
	bench := o.bench
	if strings.TrimSpace(bench) == "" {
		bench = "." // every benchmark in the package
	}
	cpu := o.cpuProfile
	if strings.TrimSpace(cpu) == "" {
		cpu = "cpu.out"
	}
	mem := o.memProfile
	if strings.TrimSpace(mem) == "" {
		mem = "mem.out"
	}

	p := profilePlan{Pkg: pkg, CPUProfile: cpu}
	// -run=^$ disables the package's unit tests so only the benchmarks run; -benchmem
	// adds the allocations/op column that drives a hot-path change toward zero.
	p.GoArgs = []string{"-run=^$", "-bench=" + bench, "-benchmem", "-cpuprofile", cpu, "-memprofile", mem}
	if bt := strings.TrimSpace(o.benchtime); bt != "" {
		p.GoArgs = append(p.GoArgs, "-benchtime", bt)
	}
	p.GoArgs = append(p.GoArgs, pkg)

	if goos == "windows" {
		// Native `go test` is OS-policy-blocked here; route through test.ps1, which
		// forwards every arg verbatim to `go test` inside WSL.
		p.ViaWSL = true
		p.Note = "windows host: routing the benchmark run to WSL via test.ps1 (native go test is OS-policy-blocked)"
		p.Argv = append([]string{"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1"}, p.GoArgs...)
	} else {
		p.Note = goos + " host: running go test -bench directly"
		p.Argv = append([]string{"go", "test"}, p.GoArgs...)
	}
	return p, nil
}

func runProfile(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak profile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		bench      = fs.String("bench", ".", "benchmark name regexp to profile (passed to go test -bench)")
		cpuProfile = fs.String("cpuprofile", "cpu.out", "CPU profile output path")
		memProfile = fs.String("memprofile", "mem.out", "memory (allocation) profile output path")
		benchtime  = fs.String("benchtime", "", "go test -benchtime value (e.g. 2s or 1000x; empty = go default)")
		top        = fs.Bool("top", false, "after profiling, run `go tool pprof -top <cpuprofile>` to print the hottest functions")
		dry        = fs.Bool("n", false, "print the resolved command without running it")
		list       = fs.Bool("list", false, "explain what the verb captures and exit")
	)
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak profile -- host-aware developer profiler

  fak profile ./internal/ctxmmu/        CPU+mem profile of the package's benchmarks
  fak profile ./internal/recall/ -bench BenchmarkDigest
  fak profile ./internal/ctxmmu/ -benchtime 2s -top
  fak profile ./internal/ctxmmu/ -n     print the resolved command, do not run
  fak profile --list                    explain what the verb captures

On Windows, the benchmark run is routed to WSL via test.ps1 (native go test is OS-policy-blocked).
`)
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *list {
		fmt.Fprint(stdout, "fak profile captures, for one package's benchmarks:\n"+
			"  -cpuprofile cpu.out   CPU profile (read with `go tool pprof -top cpu.out`)\n"+
			"  -memprofile mem.out   allocation profile (allocs/op via -benchmem)\n"+
			"It is the host-routing layer over `go test -bench -cpuprofile`; `fak benchmarks`\n"+
			"and `fak ablate` remain the curated benchmark surfaces.\n")
		return 0
	}

	o := profileOpts{
		pkg:        strings.TrimSpace(strings.Join(fs.Args(), " ")),
		bench:      *bench,
		cpuProfile: *cpuProfile,
		memProfile: *memProfile,
		benchtime:  *benchtime,
	}
	p, err := planProfile(runtime.GOOS, o)
	if err != nil {
		fmt.Fprintf(stderr, "fak profile: %v\n", err)
		return 2
	}
	fmt.Fprintf(stdout, "# %s\n# pkg=%s -> %s\n", p.Note, p.Pkg, strings.Join(p.Argv, " "))
	if *dry {
		fmt.Fprintf(stdout, "# then: go tool pprof -top %s\n", p.CPUProfile)
		return 0
	}

	cmd := exec.Command(p.Argv[0], p.Argv[1:]...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = stdout, stderr, os.Stdin
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak profile: %v\n", err)
		return 1
	}

	if *top {
		// `go tool pprof` only reads the captured profile (no test binary is compiled),
		// so it is not subject to the test-binary OS block and runs directly on any host.
		fmt.Fprintf(stdout, "\n# go tool pprof -top %s\n", p.CPUProfile)
		pp := exec.Command("go", "tool", "pprof", "-top", p.CPUProfile)
		windowgate.ConfigureBackgroundCommand(pp)
		pp.Stdout, pp.Stderr = stdout, stderr
		if err := pp.Run(); err != nil {
			fmt.Fprintf(stderr, "fak profile: go tool pprof: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "\nprofiles written: %s (cpu), %s (mem)\nread with: go tool pprof -top %s   |   go tool pprof -http=:0 %s\n",
			p.CPUProfile, o.memProfile, p.CPUProfile, p.CPUProfile)
	}
	return 0
}
