package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuecentrality"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdCentralityAudit(args []string) {
	if err := runCentralityAudit(args, os.Stdout, os.Stderr, time.Now); err != nil {
		fmt.Fprintf(os.Stderr, "centrality-audit: %v\n", err)
		os.Exit(1)
	}
}

func runCentralityAudit(args []string, stdout, stderr io.Writer, now func() time.Time) error {
	fs := flag.NewFlagSet("centrality-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "read a gh-compatible issue JSON array from file instead of GitHub")
	repo := fs.String("repo", "", "GitHub OWNER/REPO (default: current repository)")
	limit := fs.Int("limit", 5000, "maximum open issues to collect")
	jsonOut := fs.Bool("json", false, "emit the stable JSON report")
	selectionsPath := fs.String("selections", "", "preview exact body patches for explicitly selected issue classifications")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *limit < 1 {
		return fmt.Errorf("--limit must be positive")
	}

	var data []byte
	var err error
	scope, provenance := "open issues", "gh issue list"
	if *input != "" {
		data, err = os.ReadFile(*input)
		scope, provenance = "fixture issues", *input
	} else {
		argv := []string{"issue", "list", "--state", "open", "--limit", fmt.Sprint(*limit), "--json", "number,title,body"}
		if *repo != "" {
			argv = append(argv, "--repo", *repo)
			scope = "open issues in " + *repo
		}
		cmd := exec.Command("gh", argv...)
		windowgate.ConfigureBackgroundCommand(cmd)
		data, err = cmd.Output()
	}
	if err != nil {
		return fmt.Errorf("collect portfolio: %w", err)
	}
	issues, err := issuecentrality.Decode(data)
	if err != nil {
		return err
	}
	if *selectionsPath != "" {
		selectionData, readErr := os.ReadFile(*selectionsPath)
		if readErr != nil {
			return fmt.Errorf("read selections: %w", readErr)
		}
		var selections []issuecentrality.Selection
		if decodeErr := json.Unmarshal(selectionData, &selections); decodeErr != nil {
			return fmt.Errorf("decode selections: %w", decodeErr)
		}
		plan, planErr := issuecentrality.PreviewMigration(issues, selections)
		if planErr != nil {
			return planErr
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(plan)
	}
	report := issuecentrality.Audit(issues, scope, provenance, now(), nil)
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}
	_, err = io.WriteString(stdout, report.Text())
	return err
}
