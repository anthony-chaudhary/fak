// Command toolsandboxbench runs a tau3/ToolSandbox-shaped policy-state adapter
// smoke through fak adjudication.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolsandbox"
)

func main() {
	suitePath := flag.String("suite", filepath.FromSlash("testdata/toolsandbox/policy_state_smoke.json"), "ToolSandbox/tau3-shaped suite JSON")
	outPath := flag.String("out", "", "write report JSON to this path (default stdout)")
	mdPath := flag.String("md", "", "write markdown summary to this path")
	flag.Parse()

	suite, err := toolsandbox.Load(*suitePath)
	if err != nil {
		fatal(err)
	}
	report, err := toolsandbox.Run(context.Background(), suite, time.Now().UTC())
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
		if err := writeFile(*mdPath, []byte(toolsandbox.RenderMarkdown(report))); err != nil {
			fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "\n== toolsandboxbench ==\n")
	fmt.Fprintf(os.Stderr, "suite        : %s\n", *suitePath)
	fmt.Fprintf(os.Stderr, "tasks        : %d\n", report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "raw safe     : %d/%d\n", report.Summary.Raw.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak safe     : %d/%d\n", report.Summary.Fak.SafeSuccesses, report.Summary.TaskCount)
	fmt.Fprintf(os.Stderr, "fak denied   : %d\n", report.Summary.Fak.DeniedCalls)
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
	fmt.Fprintf(os.Stderr, "toolsandboxbench: %v\n", err)
	os.Exit(1)
}
