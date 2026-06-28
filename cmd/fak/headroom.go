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
	"flag"
	"fmt"
	"io"
	"os"
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

// readBlob reads the compress input: the named file, or stdin when the arg is "-"
// or absent.
func readBlob(args []string) ([]byte, error) {
	if len(args) >= 1 && args[0] != "-" {
		return os.ReadFile(args[0])
	}
	return io.ReadAll(os.Stdin)
}
