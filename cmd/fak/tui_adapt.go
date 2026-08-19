package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

const tuiAdaptSchema = "fak.console.adapt/1"

type tuiAdaptAxis struct {
	Axis           string `json:"axis"`
	Selection      string `json:"selection"`
	Canonical      string `json:"canonical"`
	Status         string `json:"status"`
	Activation     string `json:"activation"`
	DisableCommand string `json:"disable_command"`
}

type tuiAdaptReport struct {
	Schema string         `json:"schema"`
	Axes   []tuiAdaptAxis `json:"axes"`
}

func init() {
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "adapt",
		Summary: "Default-on response and work adaptations with explicit ablation controls",
		Usage:   "fak console adapt [--output-style NAME] [--work-profile NAME] [--json]",
		Schema:  tuiAdaptSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "output-style", Label: "Response shape", Kind: "select", Flag: "--output-style", Default: agentDefaultOutputStyle, Options: []string{"full", "caveman:low", "caveman:medium", "caveman:high"}, Detail: "full ablates the default Caveman response shaping"},
			{ID: "work-profile", Label: "Work policy", Kind: "select", Flag: "--work-profile", Default: agentDefaultWorkProfile, Options: []string{"standard", "ponytail:low", "ponytail:medium", "ponytail:high"}, Detail: "standard ablates the default Ponytail work policy"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit typed adaptation posture"},
		},
		Run: runTUIAdapt,
	})
}

func runTUIAdapt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("console adapt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	outputStyle := fs.String("output-style", agentDefaultOutputStyle, "response shape; full disables")
	workProfile := fs.String("work-profile", agentDefaultWorkProfile, "work policy; standard disables")
	asJSON := fs.Bool("json", false, "emit typed JSON")
	width := fs.Int("width", 80, "render width")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console adapt: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	style, err := resolveAgentOutputStyle(*outputStyle)
	if err != nil {
		fmt.Fprintln(stderr, "fak console adapt:", err)
		return 2
	}
	work, err := resolveAgentWorkProfile(*workProfile)
	if err != nil {
		fmt.Fprintln(stderr, "fak console adapt:", err)
		return 2
	}
	report := buildTUIAdaptReport(style, work)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintln(stderr, "fak console adapt:", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, formatTUIAdaptPane(report, *width))
	return 0
}

func buildTUIAdaptReport(style syspromptmmu.StyleReadout, work syspromptmmu.WorkProfileReadout) tuiAdaptReport {
	styleStatus := "ON · default"
	if style.Level == 0 {
		styleStatus = "OFF · ablated"
	}
	workStatus := "ON · default"
	if !work.Applied {
		workStatus = "OFF · ablated"
	}
	return tuiAdaptReport{Schema: tuiAdaptSchema, Axes: []tuiAdaptAxis{
		{Axis: "response", Selection: style.Style, Canonical: style.Style, Status: styleStatus, Activation: "fak agent --output-style " + style.Style, DisableCommand: "fak agent --output-style full"},
		{Axis: "work", Selection: work.Profile, Canonical: work.Profile, Status: workStatus, Activation: "fak agent --work-profile " + work.Profile, DisableCommand: "fak agent --work-profile standard"},
	}}
}

func formatTUIAdaptPane(report tuiAdaptReport, width int) string {
	var b strings.Builder
	fmt.Fprintln(&b, "Agent Adaptations · default posture and ablation controls")
	fmt.Fprintln(&b, "axis      status          selection                 ablate")
	for _, axis := range report.Axes {
		fmt.Fprintf(&b, "%-9s %-15s %-25s %s\n",
			axis.Axis,
			axis.Status,
			trimTUI(axis.Canonical, 25),
			trimTUI(axis.DisableCommand, maxTUI(16, width-53)),
		)
	}
	fmt.Fprintln(&b, "measure effects separately: fak console ablate")
	return b.String()
}
