package main

// fak headroom — the kernel's CONTEXT-COMPRESSION control surface. Headroom-style
// compression (shrink tool outputs / logs / files before they reach the model,
// reversibly) is a pluggable AREA in the kernel: `internal/headroom` defines one
// generic Compressor interface and ships three plugins — noop (the off default),
// native (an in-process structural compressor, zero external deps), and headroom
// (a bridge to a running `headroom proxy`). The selected plugin is folded into the
// result path as a ResultAdmitter, so `fak guard`/`fak serve` compress in-stream
// with no extra wiring. This verb lets a human SEE and PROVE the seam without a
// model: list the plugins, compress a blob and read the savings, check status.
//
// Exit 0 = ok, 2 = usage error.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/headroom"
)

func cmdHeadroom(argv []string) {
	os.Exit(runHeadroom(os.Stdout, os.Stderr, argv))
}

// runHeadroom is the testable core: returns the exit code, takes its streams.
func runHeadroom(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		headroomUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "list":
		return runHeadroomList(stdout, argv[1:])
	case "status":
		return runHeadroomStatus(stdout, argv[1:])
	case "compress":
		return runHeadroomCompress(stdout, stderr, argv[1:])
	case "bench":
		return runHeadroomBench(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		headroomUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak headroom: unknown subcommand %q\n", argv[0])
		headroomUsage(stderr)
		return 2
	}
}

func headroomUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak headroom list                          list compressor plugins (* = selected)
  fak headroom status                        selected plugin, headroom proxy URL + reachability, KPI
  fak headroom compress [flags] [FILE|-]     compress a blob and print the savings (proof)
  fak headroom bench [flags] [FILE...]       measure savings on a corpus (built-in, or REAL files via --dir/FILE)

bench flags:
  --via NAME    compressor plugin to bench (default: native)
  --json        emit the report as JSON
  --dir DIR     measure REAL captured files in DIR (top-level) — the dogfood path; point it at your saved tool outputs
  --max-bytes N skip a corpus file larger than N bytes (default 4 MiB)

compress flags:
  --via NAME    use a specific plugin (default: selected / FAK_COMPRESSOR)
  --model ID    target model id for token-accounting plugins
  --emit        write the compressed bytes to stdout instead of a report

selection:
  FAK_COMPRESSOR=native|headroom|noop    pick the plugin the result path folds (default noop = off)
  FAK_HEADROOM_URL=http://127.0.0.1:8787 the running headroom proxy the bridge calls
`)
}

func runHeadroomList(stdout io.Writer, _ []string) int {
	sel := headroom.Selected().Name()
	for _, name := range headroom.Names() {
		marker := "  "
		if name == sel {
			marker = "* "
		}
		fmt.Fprintf(stdout, "%s%s\n", marker, name)
	}
	return 0
}

func runHeadroomStatus(stdout io.Writer, _ []string) int {
	sel := headroom.Selected()
	fmt.Fprintf(stdout, "selected:     %s\n", sel.Name())
	fmt.Fprintf(stdout, "plugins:      %s\n", strings.Join(headroom.Names(), ", "))
	fmt.Fprintf(stdout, "headroom url: %s\n", headroom.HeadroomURL())

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	fmt.Fprintf(stdout, "reachable:    %v\n", headroom.Reachable(ctx))

	st := headroom.Default.Stats()
	fmt.Fprintf(stdout, "considered:   %d\n", st.Considered)
	fmt.Fprintf(stdout, "compressed:   %d\n", st.Compressed)
	fmt.Fprintf(stdout, "bytes in/out: %d / %d\n", st.BytesIn, st.BytesOut)
	// The "when NOT to compress" decision breakdown — the governance, made auditable.
	fmt.Fprintf(stdout, "skipped:      empty=%d poison=%d no-saving=%d not-worth=%d\n",
		st.SkippedEmpty, st.SkippedPoison, st.SkippedNoSaving, st.SkippedNotWorth)
	return 0
}

func runHeadroomCompress(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("headroom compress", flag.ContinueOnError)
	fs.SetOutput(stderr)
	via := fs.String("via", "", "compressor plugin to use (default: selected / FAK_COMPRESSOR)")
	model := fs.String("model", "", "target model id for token-accounting plugins")
	emit := fs.Bool("emit", false, "write the compressed bytes to stdout instead of a report")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	data, err := readBlob(fs.Args())
	if err != nil {
		fmt.Fprintf(stderr, "fak headroom: %v\n", err)
		return 2
	}

	comp := headroom.Selected()
	if *via != "" {
		c, ok := headroom.Lookup(*via)
		if !ok {
			fmt.Fprintf(stderr, "fak headroom: unknown plugin %q (have %s)\n", *via, strings.Join(headroom.Names(), ", "))
			return 2
		}
		comp = c
	}

	in := headroom.Input{Kind: headroom.Detect(data), Model: *model, Bytes: data}
	out, cerr := comp.Compress(context.Background(), in)
	if cerr != nil {
		fmt.Fprintf(stderr, "fak headroom: %v\n", cerr)
	}

	if *emit {
		_, _ = stdout.Write(out.Bytes)
		return 0
	}
	fmt.Fprintf(stdout, "plugin:     %s\n", comp.Name())
	fmt.Fprintf(stdout, "kind:       %s\n", in.Kind)
	fmt.Fprintf(stdout, "codec:      %s\n", out.Codec)
	fmt.Fprintf(stdout, "orig bytes: %d\n", out.OrigLen)
	fmt.Fprintf(stdout, "new bytes:  %d\n", out.NewLen)
	fmt.Fprintf(stdout, "saved:      %.1f%%\n", out.SavedRatio()*100)
	fmt.Fprintf(stdout, "compressed: %v\n", out.Compressed)
	if out.Retrieval != "" {
		fmt.Fprintf(stdout, "ccr:        %s\n", out.Retrieval)
	}
	return 0
}

// runHeadroomBench replays the built-in representative corpus through a compressor
// (native by default) and reports the realized per-sample + aggregate savings — the
// no-model, deterministic witness that the context-savings transforms actually pay
// off on the tool-output shapes an agent streams into context.
func runHeadroomBench(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("headroom bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	via := fs.String("via", headroom.NativeName, "compressor plugin to bench (default: native)")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	dir := fs.String("dir", "", "measure REAL captured files in this directory (top-level) instead of the built-in corpus — point it at your own saved tool outputs to get the dogfood savings on real traffic")
	maxBytes := fs.Int("max-bytes", 4<<20, "skip a corpus file larger than this many bytes")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	comp, ok := headroom.Lookup(*via)
	if !ok {
		fmt.Fprintf(stderr, "fak headroom: unknown plugin %q (have %s)\n", *via, strings.Join(headroom.Names(), ", "))
		return 2
	}

	// Corpus selection: --dir or trailing file args read REAL captured bytes (the
	// dogfood path — measure on your own traffic); otherwise the built-in
	// representative corpus (the no-input demo).
	var inputs []headroom.BenchInput
	if *dir != "" || len(fs.Args()) > 0 {
		var err error
		inputs, err = benchInputsFromPaths(*dir, fs.Args(), *maxBytes)
		if err != nil {
			fmt.Fprintf(stderr, "fak headroom: %v\n", err)
			return 2
		}
		if len(inputs) == 0 {
			fmt.Fprintf(stderr, "fak headroom: no readable corpus files found\n")
			return 2
		}
	} else {
		inputs = headroom.BenchCorpus()
	}

	report := headroom.RunBench(comp, inputs)
	if *asJSON {
		b, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak headroom: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprint(stdout, report.Render())
	return 0
}

// benchInputsFromPaths builds a real-corpus bench input set from a directory's
// top-level files plus any explicit file args. It reads regular files (skipping
// subdirs, unreadable files, and any file larger than maxBytes), names each by its
// base name, and sorts by path so the report is deterministic. This is the dogfood
// path: point it at your own saved tool outputs to measure compression on real
// traffic instead of the built-in synthetic corpus.
func benchInputsFromPaths(dir string, args []string, maxBytes int) ([]headroom.BenchInput, error) {
	var paths []string
	if dir != "" {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, fmt.Errorf("read dir %q: %w", dir, err)
		}
		for _, e := range entries {
			if !e.IsDir() {
				paths = append(paths, filepath.Join(dir, e.Name()))
			}
		}
	}
	paths = append(paths, args...)
	sort.Strings(paths)

	var inputs []headroom.BenchInput
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil || fi.IsDir() {
			continue
		}
		if maxBytes > 0 && fi.Size() > int64(maxBytes) {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		inputs = append(inputs, headroom.BenchInput{Name: filepath.Base(p), Bytes: b})
	}
	return inputs, nil
}

// readBlob reads the compress input: the named file, or stdin when the arg is "-"
// or absent.
func readBlob(args []string) ([]byte, error) {
	if len(args) >= 1 && args[0] != "-" {
		return os.ReadFile(args[0])
	}
	return io.ReadAll(os.Stdin)
}
