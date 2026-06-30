package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

// `fak benchmarks`  -  the single discoverable door over every benchmark fak ships.
// Before this verb, "run a benchmark" meant already knowing which of 18 cmd/*bench*
// mains or five `fak` verbs to invoke, and which bespoke flags each took. This
// reads internal/benchcatalog (the in-binary registry) and answers the three
// questions a developer actually has:
//
//	fak benchmarks list [--offline] [--json]   what benchmarks exist, what each measures, cold-start cost
//	fak benchmarks describe <name>             the one benchmark's purpose, run command, key flags, doc
//	fak benchmarks run <name> [-- extra args]  run it (prints the resolved command; runs cmd/ benches via go run)
//
// The `run` path for a standalone cmd/<name> shells out to `go run ./cmd/<name>`
// so a developer never has to remember the path; for a `fak <verb>` bench it tells
// the developer the exact verb to type (we are already inside that binary, so we
// do not re-exec ourselves).
func cmdBenchmarks(argv []string) { os.Exit(runBenchmarks(os.Stdout, os.Stderr, argv)) }

func runBenchmarks(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return benchmarksList(stdout, stderr, nil)
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "list", "ls":
		return benchmarksList(stdout, stderr, rest)
	case "describe", "show":
		return benchmarksDescribe(stdout, stderr, rest)
	case "run":
		return benchmarksRun(stdout, stderr, rest)
	case "-h", "--help", "help":
		benchmarksUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak benchmarks: unknown subcommand %q\n", sub)
		benchmarksUsage(stderr)
		return 2
	}
}

func benchmarksUsage(w io.Writer) {
	fmt.Fprint(w, `fak benchmarks  -  the index of every benchmark fak ships

usage:
  fak benchmarks list [--offline] [--json]   list all benchmarks (offline = zero-asset only)
  fak benchmarks describe <name>             one benchmark: what it measures, how to run it, key flags
  fak benchmarks run <name> [-- args...]     run a benchmark (go run for cmd/ benches; prints fak verbs)

Start here if you have never run a fak benchmark:
  fak benchmarks list --offline              the benchmarks that run NOW with no weights, GPU, dataset, or key
`)
}

// benchmarksList prints the catalog as a table (or JSON). The table is the
// discovery surface: name, kind, cold-start need, and the one-line "what number
// does this produce."
func benchmarksList(stdout, stderr io.Writer, argv []string) int {
	offlineOnly := false
	asJSON := false
	for _, a := range argv {
		switch a {
		case "--offline", "-offline":
			offlineOnly = true
		case "--json", "-json":
			asJSON = true
		default:
			fmt.Fprintf(stderr, "fak benchmarks list: unknown flag %q\n", a)
			return 2
		}
	}

	list := benchcatalog.All()
	if offlineOnly {
		list = benchcatalog.Offline()
	}

	if asJSON {
		_ = writeIndentedJSON(stdout, list)
		return 0
	}

	off := len(benchcatalog.Offline())
	fmt.Fprintf(stdout, "%d benchmarks (%d run offline  -  no weights, GPU, dataset, or key).\n", len(benchcatalog.All()), off)
	if !offlineOnly {
		fmt.Fprintf(stdout, "Tip: `fak benchmarks list --offline` for the zero-asset set; `fak benchmarks describe <name>` for one.\n")
	}
	fmt.Fprintln(stdout)

	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tKIND\tNEEDS\tWHAT IT MEASURES")
	for _, b := range list {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", b.Name, b.Kind, b.Need, truncate(b.Summary, 72))
	}
	_ = tw.Flush()
	return 0
}

// benchmarksDescribe prints the full record for one benchmark: the run command a
// developer copy-pastes, the key flags, and the methodology doc.
func benchmarksDescribe(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak benchmarks describe <name>   (see `fak benchmarks list`)")
		return 2
	}
	name := argv[0]
	b, ok := benchcatalog.Get(name)
	if !ok {
		fmt.Fprintf(stderr, "fak benchmarks: no benchmark named %q. Run `fak benchmarks list`.\n", name)
		if s := nearest(name); s != "" {
			fmt.Fprintf(stderr, "Did you mean %q?\n", s)
		}
		return 1
	}

	fmt.Fprintf(stdout, "%s  (%s, needs: %s)\n", b.Name, b.Kind, b.Need)
	fmt.Fprintf(stdout, "%s\n\n", b.Summary)
	fmt.Fprintf(stdout, "Run:\n  %s\n", b.Run)
	if len(b.Flags) > 0 {
		fmt.Fprintln(stdout, "\nKey flags:")
		for _, f := range b.Flags {
			fmt.Fprintf(stdout, "  %s\n", f)
		}
	}
	if b.Doc != "" {
		fmt.Fprintf(stdout, "\nMethodology / authority: %s\n", b.Doc)
	} else {
		fmt.Fprintf(stdout, "\nMethodology: the source comment in cmd/%s (or this binary's verb).\n", b.Name)
	}
	if b.Need == benchcatalog.NeedNone {
		fmt.Fprintf(stdout, "\nThis benchmark runs with no external assets  -  try it now:\n  fak benchmarks run %s\n", b.Name)
	}
	return 0
}

// benchmarksRun resolves the benchmark to its real command and runs it. A cmd/
// bench is launched with `go run ./cmd/<name>` (plus any developer-supplied
// trailing args after `--`); a `fak <verb>` bench is NOT re-exec'd  -  we print the
// exact verb to type, since the developer is already running this binary.
func benchmarksRun(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak benchmarks run <name> [-- args...]")
		return 2
	}
	name := argv[0]
	var extra []string
	for i, a := range argv[1:] {
		if a == "--" {
			extra = argv[1+i+1:]
			break
		}
	}

	b, ok := benchcatalog.Get(name)
	if !ok {
		fmt.Fprintf(stderr, "fak benchmarks: no benchmark named %q. Run `fak benchmarks list`.\n", name)
		return 1
	}

	if b.Kind == benchcatalog.KindVerb {
		if code, ok := runBuiltInBenchmark(b.Name, stdout, stderr, extra); ok {
			return code
		}
		// We are this binary; don't re-exec. Tell the developer the exact command.
		fmt.Fprintf(stdout, "`%s` is a built-in fak verb. Run it directly:\n\n  %s\n", b.Name, b.Run)
		return 0
	}

	// cmd/<name>: go run ./cmd/<name> [extra...]
	args := []string{"run", "./cmd/" + b.Name}
	args = append(args, extra...)
	fmt.Fprintf(stdout, "running: go %s\n", strings.Join(args, " "))
	if b.Need != benchcatalog.NeedNone && len(extra) == 0 {
		fmt.Fprintf(stdout, "note: %s needs %s  -  see `fak benchmarks describe %s` for the asset flags.\n", b.Name, b.Need, b.Name)
	}
	cmd := exec.Command("go", args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(stderr, "fak benchmarks run %s: %v\n", b.Name, err)
		return 1
	}
	return 0
}

func runBuiltInBenchmark(name string, stdout, stderr io.Writer, extra []string) (int, bool) {
	switch name {
	case "vcache":
		args := []string{"bench", "--json", "--snapshot", "off"}
		args = append(args, extra...)
		return runVCache(stdout, stderr, args), true
	default:
		return 0, false
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "..."
}

// nearest returns the registered name with the longest shared prefix with q, for
// a "did you mean" hint. Cheap and deterministic  -  no edit-distance dependency.
func nearest(q string) string {
	best, bestLen := "", 0
	for _, n := range benchcatalog.Names() {
		l := commonPrefix(q, n)
		if l > bestLen {
			best, bestLen = n, l
		}
	}
	if bestLen >= 3 {
		return best
	}
	return ""
}

func commonPrefix(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}
