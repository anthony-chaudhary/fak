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

const agentDefaultOutputStyle = "caveman:medium"

const (
	agentOutputStyleSourceCLI       = "cli"
	agentOutputStyleSourcePersisted = "persisted"
	agentOutputStyleSourceDefault   = "shipped-default"
)

type agentOutputStylePreference struct {
	Style  syspromptmmu.StyleReadout
	Source string
}

func resolveAgentOutputStyle(raw string) (syspromptmmu.StyleReadout, error) {
	style, err := syspromptmmu.ResolveStyle(raw)
	if err != nil {
		return style, fmt.Errorf("agent: invalid --output-style %q; supported: %s", raw, strings.Join(syspromptmmu.StyleNames(), ", "))
	}
	return style, nil
}

func resolveAgentOutputStylePreference(cliValue string, cliExplicit bool, configPath string) (agentOutputStylePreference, error) {
	if cliExplicit {
		style, err := resolveAgentOutputStyle(cliValue)
		return agentOutputStylePreference{Style: style, Source: agentOutputStyleSourceCLI}, err
	}
	cfg, source, err := loadTUIConsoleConfig(configPath)
	if err != nil {
		return agentOutputStylePreference{}, fmt.Errorf("agent: load persisted response profile: %w", err)
	}
	if raw := cfg.PaneDefaults["adapt"]["output-style"]; len(raw) > 0 {
		selection, err := rawTUIScalar(raw)
		if err != nil {
			return agentOutputStylePreference{}, fmt.Errorf("agent: persisted response profile in %s: %w", source, err)
		}
		style, err := syspromptmmu.ResolveStyle(selection)
		if err != nil {
			return agentOutputStylePreference{}, fmt.Errorf("agent: persisted response profile in %s: %w", source, err)
		}
		return agentOutputStylePreference{Style: style, Source: agentOutputStyleSourcePersisted}, nil
	}
	style, err := syspromptmmu.ResolveStyle(agentDefaultOutputStyle)
	if err != nil {
		return agentOutputStylePreference{}, fmt.Errorf("agent: shipped response profile: %w", err)
	}
	return agentOutputStylePreference{Style: style, Source: agentOutputStyleSourceDefault}, nil
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

type agentProfileSweepPlan struct {
	Schema          string                 `json:"schema"`
	Axes            string                 `json:"axes"`
	ResultSemantics string                 `json:"result_semantics"`
	Rows            []agentProfileSweepRow `json:"rows"`
}

type agentProfileSweepRow struct {
	Axis      string `json:"axis"`
	Rung      string `json:"rung"`
	Selection string `json:"selection"`
	Canonical string `json:"canonical"`
	Meaning   string `json:"meaning"`
	Command   string `json:"command"`
	Result    string `json:"result"`
}

func agentOutputProfiles() []agentOutputProfile {
	return []agentOutputProfile{
		{Selection: "full", Canonical: "full", Family: "native", Implementation: "native", Intensity: "off", Status: "shipped", Meaning: "No response-shape steering."},
		{Selection: "native:low", Canonical: "native:low", Family: "native", Implementation: "native", Intensity: "low", Status: "shipped", Meaning: "Trim filler; retain full explanation where useful."},
		{Selection: "native:medium", Canonical: "native:medium", Family: "native", Implementation: "native", Intensity: "medium", Status: "shipped", Meaning: "Answer directly; keep only needed explanation."},
		{Selection: "native:high", Canonical: "native:high", Family: "native", Implementation: "native", Intensity: "high", Status: "shipped", Meaning: "Essential content only; no preamble or recap."},
		{Selection: "caveman:low", Canonical: "caveman:native:low", Family: "caveman", Implementation: "native", Intensity: "low", Status: "shipped", Meaning: "Caveman-compatible shape using fak-authored safe bytes."},
		{Selection: "caveman:medium", Canonical: "caveman:native:medium", Family: "caveman", Implementation: "native", Intensity: "medium", Status: "default", Meaning: "Default concise response shape with correctness carve-outs."},
		{Selection: "caveman:high", Canonical: "caveman:native:high", Family: "caveman", Implementation: "native", Intensity: "high", Status: "shipped", Meaning: "Strong Caveman-compatible response compression."},
		{Selection: "caveman:original:*", Family: "caveman", Implementation: "original", Intensity: "low|medium|high", Status: "not-yet", Meaning: "Reserved for a pinned, attributed upstream adapter (#6706)."},
	}
}

func agentWorkProfiles() []agentOutputProfile {
	return []agentOutputProfile{
		{Selection: "standard", Canonical: "standard", Family: "standard", Implementation: "native", Intensity: "off", Status: "shipped", Meaning: "Explicitly disable implementation-policy steering."},
		{Selection: "ponytail:low", Canonical: "ponytail:native:low", Family: "ponytail", Implementation: "native", Intensity: "low", Status: "shipped", Meaning: "Briefly check for a simpler route before adding machinery."},
		{Selection: "ponytail:medium", Canonical: "ponytail:native:medium", Family: "ponytail", Implementation: "native", Intensity: "medium", Status: "default", Meaning: "Simplicity ladder with full correctness carve-outs."},
		{Selection: "ponytail:high", Canonical: "ponytail:native:high", Family: "ponytail", Implementation: "native", Intensity: "high", Status: "shipped", Meaning: "Actively resist avoidable complexity; require justification for machinery."},
		{Selection: "ponytail:original:*", Family: "ponytail", Implementation: "original", Intensity: "low|medium|high", Status: "not-yet", Meaning: "Reserved for a pinned, attributed upstream adapter."},
	}
}

func agentProfileSweepRows() []agentProfileSweepRow {
	return []agentProfileSweepRow{
		{Axis: "response", Rung: "off", Selection: "full", Canonical: "full", Meaning: "No response-shape steering.", Command: "fak agent --output-style full --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "response", Rung: "low", Selection: "caveman:low", Canonical: "caveman:native:low", Meaning: "Light response compression.", Command: "fak agent --output-style caveman:low --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "response", Rung: "medium", Selection: "caveman:medium", Canonical: "caveman:native:medium", Meaning: "Concise response shape with correctness carve-outs.", Command: "fak agent --output-style caveman:medium --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "response", Rung: "high", Selection: "caveman:high", Canonical: "caveman:native:high", Meaning: "Strong response compression.", Command: "fak agent --output-style caveman:high --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "work", Rung: "off", Selection: "standard", Canonical: "standard", Meaning: "No implementation-policy steering.", Command: "fak agent --work-profile standard --output-style caveman:medium", Result: "not-measured"},
		{Axis: "work", Rung: "low", Selection: "ponytail:low", Canonical: "ponytail:native:low", Meaning: "Brief simplicity check before adding machinery.", Command: "fak agent --work-profile ponytail:low --output-style caveman:medium", Result: "not-measured"},
		{Axis: "work", Rung: "medium", Selection: "ponytail:medium", Canonical: "ponytail:native:medium", Meaning: "Simplicity ladder with correctness carve-outs.", Command: "fak agent --work-profile ponytail:medium --output-style caveman:medium", Result: "not-measured"},
		{Axis: "work", Rung: "high", Selection: "ponytail:high", Canonical: "ponytail:native:high", Meaning: "Actively resist avoidable complexity.", Command: "fak agent --work-profile ponytail:high --output-style caveman:medium", Result: "not-measured"},
	}
}

func printAgentProfileSweep(w io.Writer, jsonOut bool) error {
	rows := agentProfileSweepRows()
	if jsonOut {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(agentProfileSweepPlan{
			Schema:          "fak-agent-profile-sweep/1",
			Axes:            "independent",
			ResultSemantics: "not-measured means this row is a stable control plan, not a benchmark result",
			Rows:            rows,
		})
	}
	fmt.Fprintln(w, "Profile sweep plan (independent controls; stable off/low/medium/high rows):")
	for _, row := range rows {
		fmt.Fprintf(w, "  %-8s %-6s %-18s %-13s %s\n", row.Axis, row.Rung, row.Selection, row.Result, row.Command)
	}
	fmt.Fprintln(w, "Run the same recorded task or fixture with fixed options for each row, then compare its receipts.")
	fmt.Fprintln(w, "No benchmark results are bundled; every row is not-measured until you run it.")
	fmt.Fprintln(w, "Machine-readable plan: fak agent profiles --sweep --json")
	return nil
}

func printAgentOutputProfiles(w io.Writer, argv []string) error {
	fs := flag.NewFlagSet("agent profiles", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOut := fs.Bool("json", false, "emit stable JSON")
	sweep := fs.Bool("sweep", false, "emit the stable independent-axis sweep plan")
	if err := fs.Parse(argv); err != nil {
		return fmt.Errorf("agent profiles: %w", err)
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("agent profiles: unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *sweep {
		return printAgentProfileSweep(w, *jsonOut)
	}
	profiles := agentOutputProfiles()
	if *jsonOut {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(profiles)
	}
	fmt.Fprintln(w, "Response profiles (independent axis; default is caveman:medium):")
	for _, p := range profiles {
		fmt.Fprintf(w, "  %-24s %-8s %s\n", p.Selection, p.Status, p.Meaning)
	}
	fmt.Fprintln(w, "\nWork profiles (independent axis; default is ponytail:medium):")
	for _, p := range agentWorkProfiles() {
		fmt.Fprintf(w, "  %-24s %-8s %s\n", p.Selection, p.Status, p.Meaning)
	}
	fmt.Fprintln(w, "\nExamples:")
	fmt.Fprintln(w, "  fak agent --output-style caveman:medium")
	fmt.Fprintln(w, "  fak agent --work-profile ponytail:medium")
	fmt.Fprintln(w, "  fak agent --output-style caveman:high --work-profile ponytail:low  # mix independently")
	fmt.Fprintln(w, "  fak agent --output-style full --work-profile standard             # disable both")
	fmt.Fprintln(w, "\nPersistent response preference:")
	fmt.Fprintln(w, "  fak console settings --set-default adapt.output-style=caveman:medium  # set")
	fmt.Fprintln(w, "  fak console settings --json                                           # show canonical value + source")
	fmt.Fprintln(w, "  fak console settings --set-default adapt.output-style=full             # disable")
	fmt.Fprintln(w, "  precedence: CLI --output-style > persisted preference > shipped default")
	fmt.Fprintln(w, "\nSweep controls: fak agent profiles --sweep")
	fmt.Fprintln(w, "\nPrecedence: policy and explicit requirements > repository instructions > work profile > response profile.")
	return nil
}
