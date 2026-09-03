package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectoryassurance"
)

// runTrajectoryAssurance is the package-level CLI seam. It deliberately only
// decodes typed evidence and writes a shadow receipt; it has no action callback.
func runTrajectoryAssurance(stdin io.Reader, stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 && args[0] == "gym" {
		return runTrajectoryAssuranceGym(stdout, stderr, args[1:])
	}
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
		fmt.Fprintln(stderr, "usage: fak trajectory assurance [--trajctl-curve FILE] [--ultracode-status FILE] [--effect-receipts FILE] [--trajectory-audit FILE] [--dojo-receipt FILE] [--trajectory-id ID] [--max-age DURATION] < input.json\n       fak trajectory assurance gym --corpus <file> [--thresholds <file>] [--json] [--report <file>]")
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

func runTrajectoryAssuranceGym(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("trajectory assurance gym", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpusFile := fs.String("corpus", "", "path to gym corpus JSON")
	thresholdsFile := fs.String("thresholds", "", "path to custom gym thresholds JSON")
	jsonOut := fs.Bool("json", false, "format report as JSON")
	reportFile := fs.String("report", "", "write report JSON to file")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *corpusFile == "" {
		fmt.Fprintln(stderr, "usage: fak trajectory assurance gym --corpus <file> [--thresholds <file>] [--json] [--report <file>]")
		return 2
	}

	corpus, raw, err := trajectoryassurance.LoadGym(*corpusFile)
	if err != nil {
		fmt.Fprintf(stderr, "trajectory assurance gym: load corpus %s: %v\n", *corpusFile, err)
		return 1
	}

	var report trajectoryassurance.GymReport
	if *thresholdsFile != "" {
		b, err := os.ReadFile(*thresholdsFile)
		if err != nil {
			fmt.Fprintf(stderr, "trajectory assurance gym: read thresholds %s: %v\n", *thresholdsFile, err)
			return 1
		}
		threshold := trajectoryassurance.DefaultGymThreshold
		if err := json.Unmarshal(b, &threshold); err != nil {
			fmt.Fprintf(stderr, "trajectory assurance gym: parse thresholds %s: %v\n", *thresholdsFile, err)
			return 1
		}
		report = trajectoryassurance.EvaluateGymWithThresholds(corpus, raw, threshold)
	} else {
		report = trajectoryassurance.EvaluateGym(corpus, raw)
	}

	reportBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "trajectory assurance gym: marshal report: %v\n", err)
		return 1
	}
	reportBytes = append(reportBytes, '\n')

	if *reportFile != "" {
		if err := os.WriteFile(*reportFile, reportBytes, 0644); err != nil {
			fmt.Fprintf(stderr, "trajectory assurance gym: write report %s: %v\n", *reportFile, err)
			return 1
		}
	}

	if *jsonOut {
		if _, err := stdout.Write(reportBytes); err != nil {
			fmt.Fprintf(stderr, "trajectory assurance gym: write output: %v\n", err)
			return 1
		}
	} else {
		summary := formatGymSummary(report)
		if _, err := fmt.Fprint(stdout, summary); err != nil {
			fmt.Fprintf(stderr, "trajectory assurance gym: write output: %v\n", err)
			return 1
		}
	}

	return 0
}

func formatGymSummary(r trajectoryassurance.GymReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Gym Report: %s (corpus %s)\n", r.Promotion.Verdict, r.CorpusVersion)
	fmt.Fprintf(&b, "Digest:             %s\n", r.CorpusDigest)
	fmt.Fprintf(&b, "Verdict:            %s\n", r.Promotion.Verdict)
	if len(r.Promotion.Reasons) > 0 {
		fmt.Fprintf(&b, "Reasons:            %s\n", strings.Join(r.Promotion.Reasons, "; "))
	}
	fmt.Fprintf(&b, "Runs:               %d (%d cases)\n", r.Overall.Runs, r.Overall.Cases)
	fmt.Fprintf(&b, "Utility Success:    %.1f%% (CI95: %.1f%% - %.1f%%)\n",
		r.Overall.UtilitySuccess.Value*100,
		r.Overall.UtilitySuccess.CI95.Low*100,
		r.Overall.UtilitySuccess.CI95.High*100)
	fmt.Fprintf(&b, "Security Success:   %.1f%% (CI95: %.1f%% - %.1f%%)\n",
		r.Overall.SecuritySuccess.Value*100,
		r.Overall.SecuritySuccess.CI95.Low*100,
		r.Overall.SecuritySuccess.CI95.High*100)
	fmt.Fprintf(&b, "False Hold Rate:    %.2f%%\n", r.Overall.FalseHoldRate*100)
	fmt.Fprintf(&b, "Intervention Regret: %.4f\n", r.Overall.InterventionRegret)
	if r.WorstStratum.Key != "" {
		fmt.Fprintf(&b, "Worst Stratum:      %s\n", r.WorstStratum.Key)
	}
	return b.String()
}
