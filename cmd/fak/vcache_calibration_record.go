package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func runVCacheCalibrationRecord(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak vcache calibration-record", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider := fs.String("provider", "", "provider represented by the telemetry (required)")
	model := fs.String("model", "", "provider model represented by the telemetry")
	source := fs.String("source", "probe", "durable source label for this calibration run")
	telemetry := fs.String("telemetry", "-", "provider usage JSONL ('-' reads stdin)")
	output := fs.String("output", nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel), "calibration JSONL ledger")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak vcache calibration-record: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*provider) == "" {
		fmt.Fprintln(stderr, "fak vcache calibration-record: --provider is required")
		return 2
	}
	rows, err := readVCacheTelemetry(*telemetry, nil)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache calibration-record: read telemetry: %v\n", err)
		return 1
	}
	now := time.Now()
	turns := make([]vcacheobserve.Turn, 0, len(rows))
	for i, row := range rows {
		turns = append(turns, vcacheobserve.Turn{
			UnixMillis:    now.Add(time.Duration(i) * time.Millisecond).UnixMilli(),
			Family:        strings.ToLower(strings.TrimSpace(*provider)) + ":" + strings.TrimSpace(*model),
			InputTokens:   int64(row.InputTokens + row.CacheCreationInputTokens + row.CacheReadInputTokens),
			CacheCreation: int64(row.CacheCreationInputTokens),
			CacheRead:     int64(row.CacheReadInputTokens),
		})
	}
	calibration, ok := vcachecalibration.CalibrationFromTurns(*provider, *source, turns, now)
	if !ok {
		fmt.Fprintln(stderr, "fak vcache calibration-record: telemetry has no cache activity or no prediction window")
		return 1
	}
	calibration.Model = strings.TrimSpace(*model)
	if err := vcachecalibration.AppendCalibration(*output, calibration); err != nil {
		fmt.Fprintf(stderr, "fak vcache calibration-record: append: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "RECORDED provider=%s model=%s turns=%d predictions=%d false_warm_rate=%.4f false_cold_rate=%.4f path=%s\n", calibration.Provider, calibration.Model, calibration.Turns, calibration.Predictions, calibration.FalseWarmRate, calibration.FalseColdRate, *output)
	return 0
}
