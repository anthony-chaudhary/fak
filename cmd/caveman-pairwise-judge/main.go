package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/cavemanpairwise"
)

func main() {
	source := flag.String("source", "", "immutable source manifest")
	prompts := flag.String("prompts", "docs/_witnesses/armbench-caveman-native/inputs/prompts.json", "prompt input")
	protocol := flag.Int("protocol", 1, "judge protocol version (1 or 2)")
	v1Receipt := flag.String("v1-receipt", "docs/_witnesses/caveman-pairwise-judge/receipt.json", "immutable protocol-v1 receipt required by protocol 2")
	diagnosisOut := flag.String("diagnosis-out", "docs/_witnesses/caveman-pairwise-judge-v2/diagnosis.json", "protocol-v1 flip diagnosis output")
	calibration := flag.String("calibration", "internal/cavemanpairwise/testdata/calibration.json", "held-out human labels")
	safety := flag.String("safety-receipt", "docs/_witnesses/caveman-safety-judge/compact.json", "independent deterministic safety receipt")
	out := flag.String("out", "docs/_witnesses/caveman-pairwise-judge/receipt.json", "receipt output")
	base := flag.String("base-url", os.Getenv("OPENAI_BASE_URL"), "OpenAI-compatible base URL")
	model := flag.String("model", "gpt-5.6-sol", "pinned judge model")
	flag.Parse()
	if *source == "" || *base == "" || os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "source, base URL, and OPENAI_API_KEY are required")
		os.Exit(2)
	}
	sb, pb, cb := mustRead(*source), mustRead(*prompts), mustRead(*calibration)
	client := cavemanpairwise.Client{BaseURL: *base, APIKey: os.Getenv("OPENAI_API_KEY"), Model: *model}
	if *protocol == 2 {
		runV2(client, sb, pb, cb, *v1Receipt, *diagnosisOut, *safety, *out)
		return
	}
	if *protocol != 1 {
		fmt.Fprintln(os.Stderr, "protocol must be 1 or 2")
		os.Exit(2)
	}
	r, runErr := cavemanpairwise.Run(context.Background(), client, sb, pb, cb)
	if runErr == nil {
		runErr = bindSafety(&r, sb, mustRead(*safety))
	}
	writeJSON(*out, r)
	fmt.Printf("protocol=1 calibration_pass=%v application_calls=%d token_eligible=%v receipt=%s\n", r.Calibration.Pass, len(r.Application.Pairs)*2, r.TokenEligible, *out)
	exitFor(r.Calibration.Pass, r.Application.NonInferiority, r.TokenEligible, runErr)
}

func runV2(client cavemanpairwise.Client, source, prompts, calibration []byte, v1Path, diagnosisPath, safetyPath, out string) {
	v1 := mustRead(v1Path)
	diagnosis, err := cavemanpairwise.DiagnoseV1(v1)
	if err != nil {
		panic(err)
	}
	writeJSON(diagnosisPath, diagnosis)
	r, runErr := cavemanpairwise.RunV2(context.Background(), client, source, prompts, calibration, v1)
	if runErr == nil {
		runErr = bindSafety(&r.Receipt, source, mustRead(safetyPath))
	}
	writeJSON(out, r)
	fmt.Printf("protocol=2 calibration_pass=%v application_calls=%d token_eligible=%v receipt=%s\n", r.Calibration.Pass, len(r.Application.Pairs)*cavemanpairwise.RepeatsPerOrderV2*2, r.TokenEligible, out)
	exitFor(r.Calibration.Pass, r.Application.NonInferiority, r.TokenEligible, runErr)
}

func bindSafety(r *cavemanpairwise.Receipt, source, safety []byte) error {
	var meta struct {
		Upstream struct {
			SavedPercent float64 `json:"saved_percent"`
		} `json:"upstream"`
	}
	if err := json.Unmarshal(source, &meta); err != nil {
		return err
	}
	return cavemanpairwise.BindSafety(r, safety, meta.Upstream.SavedPercent)
}

func exitFor(calibrationPass bool, nonInferiority *bool, tokenEligible bool, runErr error) {
	if runErr != nil {
		fmt.Fprintln(os.Stderr, runErr)
		os.Exit(1)
	}
	if !calibrationPass || nonInferiority == nil || !*nonInferiority || !tokenEligible {
		os.Exit(1)
	}
}

func writeJSON(path string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		panic(err)
	}
	b = append(b, '\n')
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		panic(err)
	}
	if err = os.WriteFile(path, b, 0644); err != nil {
		panic(err)
	}
}

func mustRead(path string) []byte {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	return b
}
