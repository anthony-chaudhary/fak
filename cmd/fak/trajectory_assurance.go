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
	trajctlFile := fs.String("trajctl-curve", "", "fak-trajctl-curve/1 objective progress receipt")
	auditFile := fs.String("trajectory-audit", "", "fak-trajectory-audit/1 JSONL diagnostics")
	dojoFile := fs.String("dojo-receipt", "", "fak-dojo-rsi/1 efficiency receipt")
	effectsFile := fs.String("effect-receipts", "", "fak.orchestration_effect_receipt.v1 JSON stream")
	trajectoryID := fs.String("trajectory-id", "", "trajectory/session identity used to select audit diagnostics")
	maxAge := fs.Duration("max-age", 24*time.Hour, "maximum age for timestamped receipts")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak trajectory assurance [--trajctl-curve FILE] [--ultracode-status FILE] [--effect-receipts FILE] [--trajectory-audit FILE] [--dojo-receipt FILE] [--trajectory-id ID] [--max-age DURATION] < input.json")
		return 2
	}
	var input trajectoryassurance.Input
	declared := []string{*statusFile, *trajctlFile, *auditFile, *dojoFile, *effectsFile}
	usingReceipts := false
	for _, path := range declared {
		if path != "" {
			usingReceipts = true
		}
	}
	if usingReceipts {
		now := time.Now()
		adapters := []struct {
			kind   string
			path   string
			decode func(io.Reader) (trajectoryassurance.Input, error)
		}{
			{"trajctl", *trajctlFile, func(r io.Reader) (trajectoryassurance.Input, error) {
				return trajectoryassurance.DecodeTrajctlCurve(r, now, *maxAge)
			}},
			{"ultracode", *statusFile, func(r io.Reader) (trajectoryassurance.Input, error) {
				return trajectoryassurance.DecodeUltracodeStatus(r, now)
			}},
			{"effects", *effectsFile, func(r io.Reader) (trajectoryassurance.Input, error) {
				return trajectoryassurance.DecodeEffectReceipts(r, now, *maxAge)
			}},
			{"audit", *auditFile, func(r io.Reader) (trajectoryassurance.Input, error) {
				return trajectoryassurance.DecodeTrajectoryAudit(r, *trajectoryID)
			}},
			{"dojo", *dojoFile, trajectoryassurance.DecodeDojoIteration},
		}
		for _, adapter := range adapters {
			if adapter.path == "" {
				continue
			}
			file, err := os.Open(adapter.path)
			if err != nil {
				part := trajectoryassurance.UnavailableInput(adapter.kind, err.Error())
				if err := trajectoryassurance.MergeInput(&input, part); err != nil {
					fmt.Fprintln(stderr, err)
					return 2
				}
				continue
			}
			part, err := adapter.decode(file)
			closeErr := file.Close()
			if err == nil {
				err = closeErr
			}
			if err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
			if err := trajectoryassurance.MergeInput(&input, part); err != nil {
				fmt.Fprintln(stderr, err)
				return 2
			}
		}
		if input.TrajectoryID == "" {
			input.TrajectoryID = *trajectoryID
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
