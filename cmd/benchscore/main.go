package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/benchscore"
)

func main() {
	root := flag.String("root", "experiments/benchmark/runs", "root to scan for score.json files")
	jsonOut := flag.Bool("json", false, "emit JSON instead of a markdown matrix")
	flag.Parse()

	report, err := benchscore.Scan(*root)
	if err != nil {
		fmt.Fprintln(os.Stderr, "benchscore:", err)
		os.Exit(2)
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintln(os.Stderr, "benchscore:", err)
			os.Exit(2)
		}
	} else {
		fmt.Print(benchscore.RenderMarkdown(report))
	}
	if len(report.Issues) > 0 {
		os.Exit(1)
	}
}
