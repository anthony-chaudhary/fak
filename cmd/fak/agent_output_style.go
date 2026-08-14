package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func resolveAgentOutputStyle(raw string) (syspromptmmu.StyleReadout, error) {
	style := syspromptmmu.DescribeStyle(raw)
	if !style.Known {
		return style, fmt.Errorf("agent: invalid --output-style %q; supported: %s", raw, strings.Join(syspromptmmu.StyleNames(), ", "))
	}
	return style, nil
}

func applyAgentOutputStyle(style syspromptmmu.StyleReadout) (func(), error) {
	previous, hadPrevious := os.LookupEnv(syspromptmmu.StyleEnvVar)
	if err := os.Setenv(syspromptmmu.StyleEnvVar, style.Style); err != nil {
		return nil, err
	}
	return func() {
		if hadPrevious {
			_ = os.Setenv(syspromptmmu.StyleEnvVar, previous)
		} else {
			_ = os.Unsetenv(syspromptmmu.StyleEnvVar)
		}
	}, nil
}

type agentOutputProfile struct {
	Selection      string `json:"selection"`
	Canonical      string `json:"canonical"`
	Family         string `json:"family"`
	Implementation string `json:"implementation"`
	Intensity      string `json:"intensity"`
	Status         string `json:"status"`
	Meaning        string `json:"meaning"`
}

func agentOutputProfiles() []agentOutputProfile {
	return []agentOutputProfile{
		{Selection: "full", Canonical: "full", Family: "native", Implementation: "native", Intensity: "off", Status: "shipped", Meaning: "No response-shape steering."},
		{Selection: "native:low", Canonical: "native:low", Family: "native", Implementation: "native", Intensity: "low", Status: "shipped", Meaning: "Trim filler; retain full explanation where useful."},
		{Selection: "native:medium", Canonical: "native:medium", Family: "native", Implementation: "native", Intensity: "medium", Status: "shipped", Meaning: "Answer directly; keep only needed explanation."},
		{Selection: "native:high", Canonical: "native:high", Family: "native", Implementation: "native", Intensity: "high", Status: "shipped", Meaning: "Essential content only; no preamble or recap."},
		{Selection: "caveman:low", Canonical: "caveman:native:low", Family: "caveman", Implementation: "native", Intensity: "low", Status: "shipped", Meaning: "Caveman-compatible shape using fak-authored safe bytes."},
		{Selection: "caveman:medium", Canonical: "caveman:native:medium", Family: "caveman", Implementation: "native", Intensity: "medium", Status: "shipped", Meaning: "Recommended Caveman-compatible balance."},
		{Selection: "caveman:high", Canonical: "caveman:native:high", Family: "caveman", Implementation: "native", Intensity: "high", Status: "shipped", Meaning: "Strong Caveman-compatible response compression."},
		{Selection: "caveman:original:*", Family: "caveman", Implementation: "original", Intensity: "low|medium|high", Status: "not-yet", Meaning: "Reserved for a pinned, attributed upstream adapter (#6706)."},
	}
}

func printAgentOutputProfiles(w io.Writer, argv []string) error {
	fs := flag.NewFlagSet("agent profiles", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOut := fs.Bool("json", false, "emit stable JSON")
	if err := fs.Parse(argv); err != nil {
		return fmt.Errorf("agent profiles: %w", err)
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("agent profiles: unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}
	profiles := agentOutputProfiles()
	if *jsonOut {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(profiles)
	}
	fmt.Fprintln(w, "Response profiles (opt-in; default is full):")
	for _, p := range profiles {
		fmt.Fprintf(w, "  %-24s %-8s %s\n", p.Selection, p.Status, p.Meaning)
	}
	fmt.Fprintln(w, "\nExamples:")
	fmt.Fprintln(w, "  fak agent --output-style caveman:medium")
	fmt.Fprintln(w, "  fak agent --output-style native:high")
	fmt.Fprintln(w, "  fak agent --output-style full        # disable")
	fmt.Fprintln(w, "\nResponse shape does not change work policy. Ponytail composition is tracked in #6700/#6707.")
	return nil
}
