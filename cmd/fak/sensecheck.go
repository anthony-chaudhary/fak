package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/sensecheck"
)

func cmdSenseCheck(argv []string) {
	os.Exit(runSenseCheck(os.Stdin, os.Stdout, os.Stderr, argv))
}

func runSenseCheck(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak sensecheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	text := fs.String("text", "", "text content to evaluate")
	filePath := fs.String("file", "", "file to read and evaluate")
	kind := fs.String("kind", "text", "subject kind (commit, log, session, text)")
	ref := fs.String("ref", "", "provenance reference (sha, path, id)")
	asJSON := fs.Bool("json", false, "emit sensecheck report as JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	var content string
	if *filePath != "" {
		b, err := os.ReadFile(*filePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak sensecheck: read file: %v\n", err)
			return 2
		}
		content = string(b)
	} else if *text != "" {
		content = *text
	} else if f, ok := stdin.(*os.File); ok {
		stat, err := f.Stat()
		if err == nil && (stat.Mode()&os.ModeCharDevice) != 0 {
			content = ""
		} else {
			b, err := io.ReadAll(stdin)
			if err != nil {
				fmt.Fprintf(stderr, "fak sensecheck: read stdin: %v\n", err)
				return 2
			}
			content = string(b)
		}
	} else {
		b, err := io.ReadAll(stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak sensecheck: read stdin: %v\n", err)
			return 2
		}
		content = string(b)
	}

	subj := sensecheck.Subject{
		Kind: *kind,
		Ref:  *ref,
		Segments: []sensecheck.Segment{
			{Label: "input", Text: content},
		},
	}

	report := sensecheck.Check(subj)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak sensecheck: json encode: %v\n", err)
			return 2
		}
		return 0
	}

	fmt.Fprint(stdout, sensecheck.Render(report))
	return 0
}
