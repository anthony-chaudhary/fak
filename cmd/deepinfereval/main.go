package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/reasoneval"
)

const receiptSchema = "fak.deepinfer-eval.v1"

type scoreReceipt struct {
	Schema    string `json:"schema"`
	Benchmark string `json:"benchmark"`
	reasoneval.RuleResult
}

type refusalReceipt struct {
	Schema    string `json:"schema"`
	Benchmark string `json:"benchmark"`
	Code      string `json:"code"`
	Detail    string `json:"detail"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("deepinfereval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	benchmark := fs.String("benchmark", "", "benchmark scorer: aime, gpqa, or hle")
	prediction := fs.String("prediction", "", "model response text to score")
	gold := fs.String("gold", "", "reference answer")
	jsonOut := fs.Bool("json", false, "emit a versioned JSON receipt")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *benchmark == "" || *prediction == "" || *gold == "" {
		fmt.Fprintln(stderr, "deepinfereval: -benchmark, -prediction, and -gold are required")
		return 2
	}

	result, err := reasoneval.ScoreRule(*benchmark, *prediction, *gold)
	if errors.Is(err, reasoneval.ErrJudgeRequired) {
		_ = json.NewEncoder(stderr).Encode(refusalReceipt{
			Schema: receiptSchema, Benchmark: *benchmark, Code: "JUDGE_REQUIRED", Detail: err.Error(),
		})
		return 3
	}
	if err != nil {
		fmt.Fprintln(stderr, "deepinfereval:", err)
		return 2
	}

	receipt := scoreReceipt{Schema: receiptSchema, Benchmark: *benchmark, RuleResult: result}
	if *jsonOut {
		if err := json.NewEncoder(stdout).Encode(receipt); err != nil {
			fmt.Fprintln(stderr, "deepinfereval:", err)
			return 2
		}
		return 0
	}
	fmt.Fprintf(stdout, "%s prediction=%q source=%s correct=%t\n", receipt.Benchmark, receipt.Prediction, receipt.Source, receipt.Correct)
	return 0
}
