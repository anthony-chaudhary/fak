package main

// fak backend scaffold — the dev-kit generator for issue #1685 (parent #1678, the binding-layer
// epic). Every C-series backend (ROCm #266/C-002, TPU/ANE #261/C-004, OpenVINO #257/C-006)
// hand-wrote the same shape: an always-compiled arch/device taxonomy, a cgo `//go:build <tag>`
// registration stub into the existing compute.Register/Pick seam, and a parity-test skeleton
// mirroring cuda_test.go/vulkan_test.go. `fak backend scaffold <name> --lane <lane>` emits that
// starting shape in one command so a vendor goes from nothing to a green, build-clean,
// taxonomy-tested scaffold without reverse-engineering four existing backends.
//
// The generation logic lives in internal/compute/scaffold.go (Generate/WriteScaffold); this file
// is only the CLI shell — flag parsing, usage, and calling into that package, the same split
// index.go uses for internal/devindex.
//
// Wiring note: this file intentionally does NOT edit main.go's command switch (main.go is a
// high-churn file other sessions are actively editing concurrently). cmdBackend below is fully
// implemented and ready to dispatch; a follow-up adds ONE line to main.go's switch:
//
//	case "backend":
//		cmdBackend(argv)
//
// Until that line lands, this verb is implemented but not reachable from `fak backend ...` on
// the command line — it can still be exercised directly via cmdBackend/runBackend from Go, or by
// a test.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// cmdBackend is the os.Exit-wrapping entry point main.go's switch should dispatch `fak backend
// ...` to (see the wiring note above for the exact missing line).
func cmdBackend(argv []string) { os.Exit(runBackend(os.Stdout, os.Stderr, argv)) }

func runBackend(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeBackendUsage(stderr)
		return 2
	}
	sub := argv[0]
	rest := argv[1:]
	switch sub {
	case "scaffold":
		return runBackendScaffold(stdout, stderr, rest)
	case "-h", "--help", "help":
		writeBackendUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak backend: unknown subcommand %q\n", sub)
		writeBackendUsage(stderr)
		return 2
	}
}

func writeBackendUsage(w io.Writer) {
	fmt.Fprintf(w, `fak backend scaffold <name> --lane <%s>

Generates a compute backend dev-kit scaffold: <name>_arch.go (the always-compiled
device/arch taxonomy), <name>_arch_test.go (host-tractable taxonomy tests, green
immediately), <name>_backend.go (a //go:build-gated cgo registration stub with TODO op
bodies), and <NAME>-NOTES.md (the C-series house-format hand-off notes).

Flags:
  --lane <lane>   required; one of: %s
  --dir <path>    output directory (default: internal/compute, the repo's compute package)

Example:
  fak backend scaffold mychip --lane custom
`, strings.Join(compute.KnownLanes(), "|"), strings.Join(compute.KnownLanes(), ", "))
}

func runBackendScaffold(stdout, stderr io.Writer, argv []string) int {
	// The <name> positional argument comes before any flags in the documented usage
	// (`fak backend scaffold <name> --lane <lane>`), but Go's flag.Parse stops consuming
	// flags at the first non-flag token — so <name> must be peeled off manually before the
	// FlagSet ever sees the remaining argv, or --lane/--dir are silently never parsed.
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		fmt.Fprintln(stderr, "fak backend scaffold: expected exactly one <name> argument")
		writeBackendUsage(stderr)
		return 2
	}
	name := argv[0]

	fs := flag.NewFlagSet("backend scaffold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lane := fs.String("lane", "", "backend lane: "+strings.Join(compute.KnownLanes(), "|"))
	dir := fs.String("dir", "", "output directory (default: internal/compute under the repo root)")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak backend scaffold: unexpected extra arguments: %s\n", strings.Join(fs.Args(), " "))
		writeBackendUsage(stderr)
		return 2
	}

	if *lane == "" {
		fmt.Fprintln(stderr, "fak backend scaffold: --lane is required")
		writeBackendUsage(stderr)
		return 2
	}
	if _, ok := compute.LookupLane(*lane); !ok {
		fmt.Fprintf(stderr, "fak backend scaffold: unknown --lane %q; known lanes: %s\n", *lane, strings.Join(compute.KnownLanes(), ", "))
		return 2
	}

	outDir := *dir
	if outDir == "" {
		root := findRepoRootForBackendScaffold(".")
		outDir = filepath.Join(root, "internal", "compute")
	}

	written, err := compute.WriteScaffold(outDir, compute.ScaffoldSpec{Name: name, Lane: compute.Lane(*lane)})
	if err != nil {
		fmt.Fprintf(stderr, "fak backend scaffold: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak backend scaffold: wrote %d files to %s\n", len(written), outDir)
	for _, p := range written {
		fmt.Fprintf(stdout, "  %s\n", p)
	}
	fmt.Fprintf(stdout, "\nnext: go build ./... && go test ./internal/compute/ -run %sArch\n", exportedPrefixForUsage(name))
	return 0
}

// exportedPrefixForUsage mirrors compute's unexported exportedPrefix (mychip -> Mychip) for the
// hint printed after generation; kept tiny and local rather than exporting a second copy of the
// same one-line titlecase helper from the compute package.
func exportedPrefixForUsage(name string) string {
	if name == "" {
		return name
	}
	r := []rune(name)
	if r[0] >= 'a' && r[0] <= 'z' {
		r[0] -= 'a' - 'A'
	}
	return string(r)
}

// findRepoRootForBackendScaffold walks up from start looking for go.mod, so `fak backend
// scaffold` writes into internal/compute at the repo root even when invoked from a subdirectory.
// Falls back to start if no go.mod is found (WriteScaffold will then simply create the
// requested-but-not-found path under start, same as any other relative --dir).
func findRepoRootForBackendScaffold(start string) string {
	dir := start
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return start
}
