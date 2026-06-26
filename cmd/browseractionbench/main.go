// Command browseractionbench runs a browser/computer-use action-mediation smoke
// through fak adjudication.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/browseraction"
)

func main() {
	suitePath := flag.String("suite", filepath.FromSlash("testdata/webbench/action_mediation_smoke.json"), "browser action mediation suite JSON")
	outPath := flag.String("out", "", "write report JSON to this path (default stdout)")
	mdPath := flag.String("md", "", "write markdown summary to this path")
	flag.Parse()

	suite, err := browseraction.LoadActionMediationSuite(*suitePath)
	if err != nil {
		fatal(err)
	}
	report, err := browseraction.RunActionMediation(context.Background(), suite, time.Now().UTC())
	if err != nil {
		fatal(err)
	}
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fatal(err)
	}
	b = append(b, '\n')
	if *outPath == "" {
		if _, err := os.Stdout.Write(b); err != nil {
			fatal(err)
		}
	} else if err := writeFile(*outPath, b); err != nil {
		fatal(err)
	}
	if *mdPath != "" {
		if err := writeFile(*mdPath, []byte(browseraction.RenderActionMediationMarkdown(report))); err != nil {
			fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "\n== browseractionbench ==\n")
	fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
	fmt.Fprintf(os.Stderr, "tasks        : %d\n", report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "raw safe     : %d/%d\n", report.Summary.Raw.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak safe     : %d/%d\n", report.Summary.Fak.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak denied   : %d\n", report.Summary.Fak.DeniedActions)
	if *outPath != "" {
		fmt.Fprintf(os.Stderr, "json         : %s\n", *outPath)
	}
	if *mdPath != "" {
		fmt.Fprintf(os.Stderr, "markdown     : %s\n", *mdPath)
	}
}

func writeFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "browseractionbench: %v\n", err)
	os.Exit(1)
}
