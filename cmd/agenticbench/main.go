// Command agenticbench emits the #868 parent rollup over committed agentic
// benchmark artifacts.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agenticbench"
)

func main() {
	root := flag.String("root", ".", "repo root containing the benchmark artifacts")
	out := flag.String("out", "", "write rollup JSON to this path (default stdout)")
	md := flag.String("md", "", "write markdown summary to this path")
	externalQueue := flag.String("external-queue", "", "write pending external harness queue JSON to this path")
	externalQueueMD := flag.String("external-queue-md", "", "write pending external harness queue markdown to this path")
	strict := flag.Bool("strict", false, "exit nonzero unless the epic result gate is complete")
	flag.Parse()

	now := time.Now().UTC()
	report, err := agenticbench.Build(*root, now)
	if err != nil {
		fatal(err)
	}
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fatal(err)
	}
	b = append(b, '\n')
	if *out == "" {
		if _, err := os.Stdout.Write(b); err != nil {
			fatal(err)
		}
	} else if err := writeFile(*out, b); err != nil {
		fatal(err)
	}
	if *md != "" {
		if err := writeFile(*md, []byte(agenticbench.RenderMarkdown(report))); err != nil {
			fatal(err)
		}
	}
	var queue *agenticbench.ExternalHarnessQueue
	if *externalQueue != "" || *externalQueueMD != "" {
		queue, err = agenticbench.WriteExternalHarnessQueue(*root, *externalQueue, *externalQueueMD, now)
		if err != nil {
			fatal(err)
		}
	}
	fmt.Fprintf(os.Stderr, "\n== agenticbench ==\n")
	fmt.Fprintf(os.Stderr, "epic         : #%d\n", report.Epic)
	fmt.Fprintf(os.Stderr, "status       : %s\n", report.Status)
	fmt.Fprintf(os.Stderr, "children     : %d/%d parsed\n", report.Summary.ChildrenParsed, report.Summary.ChildrenTotal)
	fmt.Fprintf(os.Stderr, "result claims: %d\n", report.Summary.ResultClaimArtifacts)
	if queue != nil {
		fmt.Fprintf(os.Stderr, "queue items  : %d\n", queue.Summary.ItemsTotal)
	}
	if *out != "" {
		fmt.Fprintf(os.Stderr, "json         : %s\n", *out)
	}
	if *md != "" {
		fmt.Fprintf(os.Stderr, "markdown     : %s\n", *md)
	}
	if *externalQueue != "" {
		fmt.Fprintf(os.Stderr, "queue json   : %s\n", *externalQueue)
	}
	if *externalQueueMD != "" {
		fmt.Fprintf(os.Stderr, "queue md     : %s\n", *externalQueueMD)
	}
	if *strict && !report.ResultClaimAllowed {
		os.Exit(2)
	}
}

func writeFile(path string, b []byte) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, b, 0o644)
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "agenticbench: %v\n", err)
	os.Exit(1)
}
