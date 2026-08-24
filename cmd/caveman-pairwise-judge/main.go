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
	calibration := flag.String("calibration", "internal/cavemanpairwise/testdata/calibration.json", "human labels")
	safety := flag.String("safety-receipt", "docs/_witnesses/caveman-safety-judge/compact.json", "independent deterministic safety receipt")
	out := flag.String("out", "docs/_witnesses/caveman-pairwise-judge/receipt.json", "receipt output")
	base := flag.String("base-url", os.Getenv("OPENAI_BASE_URL"), "OpenAI-compatible base URL")
	model := flag.String("model", "gpt-5.6-sol", "pinned judge model")
	flag.Parse()
	if *source == "" || *base == "" || os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "source, base URL, and OPENAI_API_KEY are required")
		os.Exit(2)
	}
	sb := mustRead(*source)
	pb := mustRead(*prompts)
	cb := mustRead(*calibration)
	r, runErr := cavemanpairwise.Run(context.Background(), cavemanpairwise.Client{BaseURL: *base, APIKey: os.Getenv("OPENAI_API_KEY"), Model: *model}, sb, pb, cb)
	if runErr == nil {
		var sourceMeta struct {
			Upstream struct {
				SavedPercent float64 `json:"saved_percent"`
			} `json:"upstream"`
		}
		if err := json.Unmarshal(sb, &sourceMeta); err != nil {
			runErr = err
		} else if err := cavemanpairwise.BindSafety(&r, mustRead(*safety), sourceMeta.Upstream.SavedPercent); err != nil {
			runErr = err
		}
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		panic(err)
	}
	b = append(b, '\n')
	if err = os.MkdirAll(filepath.Dir(*out), 0755); err != nil {
		panic(err)
	}
	if err = os.WriteFile(*out, b, 0644); err != nil {
		panic(err)
	}
	fmt.Printf("calibration_pass=%v application_calls=%d token_eligible=%v receipt=%s\n", r.Calibration.Pass, len(r.Application.Pairs)*2, r.TokenEligible, *out)
	if runErr != nil {
		fmt.Fprintln(os.Stderr, runErr)
		os.Exit(1)
	}
	if !r.Calibration.Pass || r.Application.NonInferiority == nil || !*r.Application.NonInferiority || !r.TokenEligible {
		os.Exit(1)
	}
}
func mustRead(p string) []byte {
	b, e := os.ReadFile(p)
	if e != nil {
		panic(e)
	}
	return b
}
