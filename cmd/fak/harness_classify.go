package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessclassify"
)

func runHarnessClassify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness classify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("path", "", "current project or task path")
	task := fs.String("task", "", "plain-language current task")
	taskDomain := fs.String("task-domain", "", "explicit task domain")
	projectDomain := fs.String("project-domain", "", "explicit project domain")
	choiceFile := fs.String("choice-file", "", "scoped remembered-choice JSON")
	choose := fs.String("choose", "", "operator domain choice to remember")
	choiceOut := fs.String("choice-out", "", "write a scoped remembered choice")
	reason := fs.String("reason", "", "why the operator selected --choose")
	ttl := fs.Duration("ttl", 24*time.Hour, "remembered-choice lifetime")
	var signals harnessSignalFlag
	fs.Var(&signals, "signal", "named signal KEY=DOMAIN (repeatable)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	choice, err := readHarnessChoice(*choiceFile)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness classify: %v\n", err)
		return 1
	}
	input := harnessclassify.Input{Path: *path, Task: *task, TaskDomain: *taskDomain, ProjectDomain: *projectDomain, Signals: signals.values(), Choice: choice}
	result, err := harnessclassify.Classify(input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness classify: %v\n", err)
		return 1
	}
	if *choose != "" {
		if *choiceOut == "" || strings.TrimSpace(*reason) == "" || *ttl <= 0 {
			fmt.Fprintln(stderr, "fak harness classify: --choose requires --choice-out, --reason, and positive --ttl")
			return 2
		}
		choice := harnessclassify.Choice{Domain: *choose, Scope: result.ContextKey, ExpiresAt: time.Now().UTC().Add(*ttl), Reason: *reason}
		raw, err := harnessclassify.EncodeChoice(choice)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness classify: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*choiceOut, append(raw, '\n'), 0o600); err != nil {
			fmt.Fprintf(stderr, "fak harness classify: write choice: %v\n", err)
			return 1
		}
		input.Choice = &choice
		input.TaskDomain, input.ProjectDomain = "", ""
		result, err = harnessclassify.Classify(input)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness classify: %v\n", err)
			return 1
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness classify: %v\n", err)
		return 1
	}
	if result.NeedsDecision {
		return 3
	}
	return 0
}

func readHarnessChoice(path string) (*harnessclassify.Choice, error) {
	if path == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read choice: %w", err)
	}
	var choice harnessclassify.Choice
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&choice); err != nil {
		return nil, fmt.Errorf("parse choice: %w", err)
	}
	return &choice, nil
}

type harnessSignalFlag map[string]string

func (f *harnessSignalFlag) String() string { return "" }
func (f *harnessSignalFlag) Set(value string) error {
	key, domain, ok := strings.Cut(value, "=")
	if !ok || strings.TrimSpace(key) == "" || strings.TrimSpace(domain) == "" {
		return fmt.Errorf("signal must be KEY=DOMAIN")
	}
	if *f == nil {
		*f = map[string]string{}
	}
	(*f)[strings.TrimSpace(key)] = strings.TrimSpace(domain)
	return nil
}
func (f harnessSignalFlag) values() map[string]string { return map[string]string(f) }
