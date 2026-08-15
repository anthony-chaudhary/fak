package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessclassify"
	"github.com/anthony-chaudhary/fak/internal/harnesspreview"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func runHarnessPreview(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness preview", flag.ContinueOnError)
	fs.SetOutput(stderr)
	currentPath := fs.String("current", "", "last admitted product lock")
	candidatePath := fs.String("candidate", "", "candidate product lock")
	currentDomain := fs.String("current-domain", "", "last admitted contextual domain")
	candidateDomain := fs.String("candidate-domain", "", "candidate contextual domain")
	classificationPath := fs.String("classification", "", "classification result JSON")
	conflict := fs.String("conflict", "", "resolver conflict to preview without a candidate lock")
	view := fs.String("view", "cli", "cli, tui, or json")
	headless := fs.Bool("headless", false, "fail closed with a JSON recovery action when a decision is required")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *candidatePath == "" && strings.TrimSpace(*conflict) == "" && *classificationPath == "" {
		fmt.Fprintln(stderr, "fak harness preview: --candidate or --conflict is required")
		return 2
	}
	current, err := readHarnessPreviewLock(*currentPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness preview: current: %v\n", err)
		return 1
	}
	candidate, err := readHarnessPreviewLock(*candidatePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness preview: candidate: %v\n", err)
		return 1
	}
	classification, err := readHarnessClassification(*classificationPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness preview: classification: %v\n", err)
		return 1
	}
	preview := harnesspreview.Compare(harnesspreview.Input{Current: current, Candidate: candidate, CurrentDomain: *currentDomain, CandidateDomain: *candidateDomain, Classification: classification, Conflict: *conflict})
	if *headless || *view == "json" {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(preview); err != nil {
			fmt.Fprintf(stderr, "fak harness preview: %v\n", err)
			return 1
		}
	} else {
		switch *view {
		case "cli":
			fmt.Fprint(stdout, harnesspreview.RenderCLI(preview))
		case "tui":
			fmt.Fprint(stdout, harnesspreview.RenderTUI(preview))
		default:
			fmt.Fprintln(stderr, "fak harness preview: --view must be cli, tui, or json")
			return 2
		}
	}
	if preview.RequiresDecision {
		return 3
	}
	return 0
}

func readHarnessPreviewLock(path string) (*harnessresolve.Lock, error) {
	if path == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var lock harnessresolve.Lock
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		var wrapped struct {
			Lock harnessresolve.Lock `json:"lock"`
		}
		if wrapErr := json.Unmarshal(raw, &wrapped); wrapErr != nil || wrapped.Lock.Schema == "" {
			return nil, err
		}
		lock = wrapped.Lock
	}
	if err := harnessresolve.VerifyLock(lock); err != nil {
		return nil, err
	}
	return &lock, nil
}

func readHarnessClassification(path string) (*harnessclassify.Result, error) {
	if path == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var result harnessclassify.Result
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&result); err != nil {
		return nil, err
	}
	if result.Schema != harnessclassify.Schema {
		return nil, fmt.Errorf("invalid schema %q", result.Schema)
	}
	return &result, nil
}
