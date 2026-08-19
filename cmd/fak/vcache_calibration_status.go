package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
)

func runVCacheCalibrationStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak vcache calibration-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("file", nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel), "provider calibration JSONL")
	providersCSV := fs.String("providers", "anthropic,openai", "comma-separated providers expected to be calibrated")
	maxAge := fs.Duration("max-age", vcachecalibration.DefaultCalibrationTTL, "maximum trusted calibration age")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak vcache calibration-status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	statuses, err := vcachecalibration.CalibrationStatuses(*path, strings.Split(*providersCSV, ","), time.Now(), *maxAge)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache calibration-status: %v\n", err)
		return 2
	}
	ok := true
	for _, status := range statuses {
		if status.State != "fresh" {
			ok = false
		}
	}
	if *asJSON {
		payload := struct {
			Schema   string                                `json:"schema"`
			OK       bool                                  `json:"ok"`
			Path     string                                `json:"path"`
			Statuses []vcachecalibration.CalibrationStatus `json:"statuses"`
		}{Schema: "fak.vcache.calibration-status.v1", OK: ok, Path: *path, Statuses: statuses}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak vcache calibration-status: encode: %v\n", err)
			return 2
		}
	} else {
		for _, status := range statuses {
			fmt.Fprintf(stdout, "%-12s %-7s %s", status.Provider, strings.ToUpper(status.State), status.Reason)
			if status.Row != nil {
				fmt.Fprintf(stdout, " predictions=%d false_warm=%.4f age=%.1fh", status.Row.Predictions, status.Row.FalseWarmRate, status.AgeHours)
			}
			fmt.Fprintln(stdout)
			if status.Action != "" {
				fmt.Fprintf(stdout, "  action: %s\n", status.Action)
			}
		}
	}
	if ok {
		return 0
	}
	return 1
}
