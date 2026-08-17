package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessverify"
)

func runHarnessVerifyRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness verify-run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "verified harness lock promised for the run")
	observationPath := fs.String("observation", "", "runtime capability and decision observation JSON")
	jsonView := fs.Bool("json", false, "emit machine-readable verification JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lockPath == "" || *observationPath == "" {
		fmt.Fprintln(stderr, "fak harness verify-run: --lock and --observation are required")
		return 2
	}
	lock, err := readHarnessPreviewLock(*lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: lock: %v\n", err)
		return 1
	}
	raw, err := os.ReadFile(*observationPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: observation: %v\n", err)
		return 1
	}
	var observation harnessverify.Observation
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&observation); err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: observation: %v\n", err)
		return 1
	}
	report, err := harnessverify.Verify(*lock, observation)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: %v\n", err)
		return 1
	}
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak harness verify-run: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, harnessverify.Render(report))
	}
	if report.Verdict == "deviation" {
		return 3
	}
	return 0
}
