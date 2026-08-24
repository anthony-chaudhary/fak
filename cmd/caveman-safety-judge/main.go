package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/cavemansafety"
)

type compactReceipt struct {
	Schema            string                          `json:"schema"`
	JudgeVersion      string                          `json:"judge_version"`
	SourceSHA256      string                          `json:"source_sha256"`
	RulesSHA256       string                          `json:"rules_sha256"`
	CalibrationSHA256 string                          `json:"calibration_sha256"`
	SourceSchema      string                          `json:"source_schema"`
	SourceRevision    string                          `json:"source_revision"`
	Thresholds        cavemansafety.Thresholds        `json:"thresholds"`
	Calibration       cavemansafety.CalibrationResult `json:"calibration"`
	PerArmCounts      []cavemansafety.ArmCounts       `json:"per_arm_counts"`
	TokenMetrics      []cavemansafety.TokenMetric     `json:"token_metrics,omitempty"`
	Verdict           cavemansafety.Verdict           `json:"verdict"`
	RawJudgmentCount  int                             `json:"raw_judgment_count"`
	FullReceipt       string                          `json:"full_receipt"`
}

func main() {
	source := flag.String("source", "", "source armbench manifest (input only)")
	calibration := flag.String("calibration", "internal/cavemansafety/testdata/calibration.json", "human-labeled calibration fixture")
	out := flag.String("out", "", "full output receipt path")
	compact := flag.String("compact-out", "", "compact output receipt path")
	expected := flag.String("expected-source-sha256", cavemansafety.ExpectedSourceSHA256, "pinned source digest")
	flag.Parse()
	if *source == "" || *out == "" {
		fatalf("-source and -out are required")
	}
	sourceBytes, err := os.ReadFile(*source)
	if err != nil {
		fatalf("read source: %v", err)
	}
	calibrationBytes, err := os.ReadFile(*calibration)
	if err != nil {
		fatalf("read calibration: %v", err)
	}
	normalDelta := -45.2
	nativeDelta := -14.4
	metrics := []cavemansafety.TokenMetric{{Arm: "normal", AveragePromptMedian: 976.5}, {Arm: "caveman", AveragePromptMedian: 535.2, DeltaPercent: &normalDelta}, {Arm: "native_medium", AveragePromptMedian: 836.2, DeltaPercent: &nativeDelta}}
	receipt, err := cavemansafety.Apply(sourceBytes, calibrationBytes, *expected, metrics)
	if err != nil {
		fatalf("judge: %v", err)
	}
	if err := writeJSON(*out, receipt); err != nil {
		fatalf("write full receipt: %v", err)
	}
	if *compact != "" {
		c := compactReceipt{Schema: receipt.Schema, JudgeVersion: receipt.JudgeVersion, SourceSHA256: receipt.SourceSHA256, RulesSHA256: receipt.RulesSHA256, CalibrationSHA256: receipt.CalibrationSHA256, SourceSchema: receipt.SourceSchema, SourceRevision: receipt.SourceRevision, Thresholds: receipt.Thresholds, Calibration: receipt.Calibration, PerArmCounts: receipt.PerArmCounts, TokenMetrics: receipt.TokenMetrics, Verdict: receipt.Verdict, RawJudgmentCount: len(receipt.RawJudgments), FullReceipt: filepath.Base(*out)}
		if err := writeJSON(*compact, c); err != nil {
			fatalf("write compact receipt: %v", err)
		}
	}
	fmt.Printf("safety_gate_pass=%t judgments=%d full=%s\n", receipt.Verdict.SafetyGatePass, len(receipt.RawJudgments), *out)
}
func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}
func fatalf(f string, a ...any) {
	fmt.Fprintf(os.Stderr, "caveman-safety-judge: "+f+"\n", a...)
	os.Exit(1)
}
