package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/turnavoid"
)

func cmdTurnavoid(argv []string) { os.Exit(runTurnavoid(os.Stdin, os.Stdout, os.Stderr, argv)) }

func runTurnavoid(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, turnavoidUsage)
		return 2
	}
	switch argv[0] {
	case "replay":
		return runTurnavoidReplay(stdin, stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, turnavoidUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak turnavoid: unknown subcommand %q\n%s\n", argv[0], turnavoidUsage)
		return 2
	}
}

const turnavoidUsage = `fak turnavoid - replay whole-model-turn avoidance traces

  fak turnavoid replay --in TRACE.jsonl [--json]
      Strictly replay a fak.turnavoid.trace/v1 JSONL trace. The default output is
      concise text; --json emits the deterministic fak.turnavoid.report/v1 artifact.
      Use --in - to read JSONL from stdin.

Exit: 0 on a valid replay, 2 on usage, input, validation, or output errors.`

func runTurnavoidReplay(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak turnavoid replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	inPath := fs.String("in", "", "read a fak.turnavoid.trace/v1 JSONL trace from this path (- for stdin)")
	asJSON := fs.Bool("json", false, "emit the deterministic JSON report")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *inPath == "" {
		fmt.Fprintln(stderr, "fak turnavoid replay: --in TRACE.jsonl is required (use --in - for stdin)")
		return 2
	}

	input := stdin
	var file *os.File
	if *inPath != "-" {
		var err error
		file, err = os.Open(*inPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak turnavoid replay: open input: %v\n", err)
			return 2
		}
		defer file.Close()
		input = file
	}
	report, err := turnavoid.Replay(input)
	if err != nil {
		fmt.Fprintf(stderr, "fak turnavoid replay: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak turnavoid replay")
	}
	if _, err := io.WriteString(stdout, turnavoid.RenderText(report)); err != nil {
		fmt.Fprintf(stderr, "fak turnavoid replay: write output: %v\n", err)
		return 2
	}
	return 0
}
