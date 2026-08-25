package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectoryassurance"
)

// runTrajectoryAssurance is the package-level CLI seam. It deliberately only
// decodes typed evidence and writes a shadow receipt; it has no action callback.
func runTrajectoryAssurance(stdin io.Reader, stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("trajectory assurance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	statusFile := fs.String("ultracode-status", "", "strict fak.ultracode_status.v1 receipt")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak trajectory assurance [--ultracode-status FILE] < input.json")
		return 2
	}
	var input trajectoryassurance.Input
	if *statusFile != "" {
		file, err := os.Open(*statusFile)
		if err != nil {
			fmt.Fprintf(stderr, "trajectory assurance: open ultracode status: %v\n", err)
			return 2
		}
		defer file.Close()
		input, err = trajectoryassurance.DecodeUltracodeStatus(file, time.Now())
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
	} else {
		decoder := json.NewDecoder(stdin)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			fmt.Fprintf(stderr, "trajectory assurance: decode input: %v\n", err)
			return 2
		}
		var extra any
		if err := decoder.Decode(&extra); err != io.EOF {
			if err == nil {
				fmt.Fprintln(stderr, "trajectory assurance: decode input: multiple JSON values")
			} else {
				fmt.Fprintf(stderr, "trajectory assurance: decode input: %v\n", err)
			}
			return 2
		}
	}
	payload, err := trajectoryassurance.Marshal(trajectoryassurance.Assess(input))
	if err != nil {
		fmt.Fprintf(stderr, "%v\n", err)
		return 1
	}
	if _, err := fmt.Fprintln(stdout, string(payload)); err != nil {
		fmt.Fprintf(stderr, "trajectory assurance: write receipt: %v\n", err)
		return 1
	}
	return 0
}
