// localappcert validates a captured Mac local-app certification matrix.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/localappcert"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("localappcert", flag.ContinueOnError)
	flags.SetOutput(stderr)
	matrix := flags.String("matrix", "", "captured fak.local-app-certification/1 JSON")
	capture := flags.String("capture", "", "fak.local-app-certification-capture/1 JSON")
	evidenceDir := flags.String("evidence-dir", "", "directory for per-scenario combined-output evidence")
	runtimeRevision := flags.String("runtime-revision", "", "exact runtime revision (required unless present in capture specification)")
	artifact := flags.String("artifact", "", "exact tested artifact (required unless present in capture specification)")
	asJSON := flags.Bool("json", false, "emit a machine-readable validation verdict")
	if err := flags.Parse(args); errors.Is(err, flag.ErrHelp) {
		return 0
	} else if err != nil {
		return 2
	}
	if *matrix != "" && *capture != "" {
		fmt.Fprintln(stderr, "localappcert: --matrix and --capture are mutually exclusive")
		return 2
	}
	if *capture != "" {
		if flags.NArg() != 0 {
			fmt.Fprintln(stderr, "localappcert: capture mode does not support positional arguments")
			return 2
		}
		return runCapture(*capture, *evidenceDir, *runtimeRevision, *artifact, stdout, stderr)
	}
	if *matrix == "" {
		fmt.Fprintln(stderr, "localappcert: --matrix is required")
		return 2
	}
	return runValidation(*matrix, *asJSON, stdout, stderr)
}

func runValidation(matrix string, asJSON bool, stdout, stderr io.Writer) int {
	m, err := localappcert.Load(matrix)
	if err == nil {
		err = localappcert.Validate(m)
	}
	if asJSON {
		verdict := map[string]any{"schema": "fak.local-app-certification-verdict/1", "ok": err == nil, "matrix": matrix}
		if err != nil {
			verdict["error"] = err.Error()
		}
		_ = json.NewEncoder(stdout).Encode(verdict)
	} else if err == nil {
		fmt.Fprintln(stdout, "localappcert: PASS")
	} else {
		fmt.Fprintln(stderr, err)
	}
	if err != nil {
		return 1
	}
	return 0
}

func runCapture(specPath, evidenceDir, runtimeRevision, artifact string, stdout, stderr io.Writer) int {
	spec, err := localappcert.LoadCaptureSpec(specPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	rows, err := localappcert.Capture(context.Background(), spec, localappcert.CaptureOptions{
		EvidenceDir: evidenceDir, RuntimeRevision: runtimeRevision, Artifact: artifact,
	})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(rows); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, row := range rows {
		if row.Status != localappcert.StatusPass {
			return 1
		}
	}
	return 0
}
