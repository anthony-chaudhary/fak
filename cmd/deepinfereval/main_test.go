package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func runCommand(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr strings.Builder
	code := run(args, &stdout, &stderr)
	return code, stdout.String(), stderr.String()
}

func decodeScore(t *testing.T, text string) scoreReceipt {
	t.Helper()
	var got scoreReceipt
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatalf("decode score receipt: %v; output=%q", err, text)
	}
	return got
}

func TestAIMEUnicodeDecimalThroughCommand(t *testing.T) {
	code, stdout, stderr := runCommand(t,
		"-benchmark", "aime", "-prediction", "scratch ١٢; final ٣٤", "-gold", "34", "-json")
	if code != 0 || stderr != "" {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
	got := decodeScore(t, stdout)
	if got.Schema != receiptSchema || got.Benchmark != "aime" || got.Prediction != "34" || !got.Correct {
		t.Fatalf("receipt=%+v", got)
	}
}

func TestGPQAFinalChoiceThroughCommand(t *testing.T) {
	code, stdout, stderr := runCommand(t,
		"-benchmark", "gpqa", "-prediction", "A is tempting. Final answer: D", "-gold", "D", "-json")
	if code != 0 || stderr != "" {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
	got := decodeScore(t, stdout)
	if got.Benchmark != "gpqa" || got.Prediction != "D" || !got.Correct {
		t.Fatalf("receipt=%+v", got)
	}
}

func TestHLEJudgeRequiredThroughCommand(t *testing.T) {
	code, stdout, stderr := runCommand(t,
		"-benchmark", "hle", "-prediction", "candidate", "-gold", "reference", "-json")
	if code == 0 || stdout != "" {
		t.Fatalf("code=%d stdout=%q", code, stdout)
	}
	var got refusalReceipt
	if err := json.Unmarshal([]byte(stderr), &got); err != nil {
		t.Fatalf("decode refusal: %v; stderr=%q", err, stderr)
	}
	if got.Schema != receiptSchema || got.Benchmark != "hle" || got.Code != "JUDGE_REQUIRED" {
		t.Fatalf("refusal=%+v", got)
	}
}
