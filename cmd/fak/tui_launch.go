package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

func init() {
	tuiplugin.Register(tuiplugin.Pane{
		ID: "launch", Summary: "Provider interception and one-shot direct launch posture",
		Usage: "fak console launch [--direct-next] [--json]", Schema: tuiLaunchSchema, BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "direct-next", Label: "Direct next", Kind: "toggle", Flag: "--direct-next", Detail: "one-shot pass-through; does not persist disable"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit typed launch posture"},
		}, Run: runTUILaunch,
	})
}

const tuiLaunchSchema = "fak.console.launch/1"

type tuiLaunchReport struct {
	Schema            string `json:"schema"`
	Default           string `json:"default,omitempty"`
	PersistedDisabled bool   `json:"persisted_disabled"`
	NextLaunchDirect  bool   `json:"next_launch_direct"`
	Interception      string `json:"interception"`
	KeyHint           string `json:"key_hint"`
}

func runTUILaunch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("console launch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	direct := fs.Bool("direct-next", false, "launch the next provider directly without persisting disable")
	asJSON := fs.Bool("json", false, "emit typed JSON")
	width := fs.Int("width", 80, "render width")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintln(stderr, "fak console launch:", err)
		return 1
	}
	report := tuiLaunchReport{Schema: tuiLaunchSchema, Default: c.Default, PersistedDisabled: c.Disabled, NextLaunchDirect: *direct, KeyHint: "[d] direct next launch (one-shot) | persisted disable: fak launch disable"}
	switch {
	case c.Disabled:
		report.Interception = "DISABLED (persisted pass-through)"
	case *direct:
		report.Interception = "DIRECT NEXT (one-shot; persisted setting unchanged)"
	default:
		report.Interception = "ENABLED (fak guard intercepts)"
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, formatTUILaunchPane(report, *width))
	return 0
}

func formatTUILaunchPane(r tuiLaunchReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Provider Launch  interception=%s\n", r.Interception)
	fmt.Fprintf(&b, "default=%s\n", firstNonEmpty(r.Default, "(unset)"))
	fmt.Fprintln(&b, r.KeyHint)
	return b.String()
}
