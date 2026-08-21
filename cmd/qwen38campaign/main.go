// Command qwen38campaign runs the frozen Qwen3.8 evidence campaign.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("qwen38campaign", flag.ContinueOnError)
	fs.SetOutput(stderr)
	config := fs.String("config", "", "campaign adapter JSON")
	corpus := fs.String("corpus", "docs/benchmarks/qwen38-quant/corpus.json", "frozen corpus JSON")
	report := fs.String("report", "", "validator-clean report output")
	archive := fs.String("archive", "", "secret-scrubbed raw archive output")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *config == "" || *report == "" || *archive == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: qwen38campaign --config CONFIG.json --report REPORT.json --archive ARCHIVE.json [--corpus CORPUS.json]")
		return 2
	}
	if err := qwen38quantrun.RunAdapter(context.Background(), *config, *corpus, *report, *archive); err != nil {
		fmt.Fprintf(stderr, "qwen38campaign: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "PASS report=%s archive=%s\n", *report, *archive)
	return 0
}
