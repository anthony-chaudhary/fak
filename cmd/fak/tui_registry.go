package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

const tuiRegistrySchema = "fak.tui.registry.v1"

type tuiPaneRegistryReport struct {
	Schema string                 `json:"schema"`
	Counts tuiPaneRegistryCounts  `json:"counts"`
	Panes  []tuiplugin.Descriptor `json:"panes"`
}

type tuiPaneRegistryCounts struct {
	Panes    int `json:"panes"`
	BuiltIn  int `json:"built_in"`
	Overview int `json:"overview"`
	Controls int `json:"controls"`
}

func init() {
	registerBuiltinTUIPanes()
}

func registerBuiltinTUIPanes() {
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "agent",
		Summary: "launch a guarded Claude Code backend with account, target, and gateway controls",
		Usage:   "fak console agent [<target> | --target NAME | --auto] [--dry-run] [--json] [--] [claude args...]",
		Schema:  tuiAgentSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "account", Label: "Account", Kind: "input", Flag: "--account", Detail: "select a named Claude account from fak accounts"},
			{ID: "auto-target", Label: "Auto Target", Kind: "action", Flag: "--auto", Detail: "pick the first launchable registered compute target"},
			{ID: "dry-run", Label: "Dry Run", Kind: "toggle", Flag: "--dry-run", Detail: "render launch/env without spawning the agent"},
			{ID: "effort", Label: "Reasoning Effort", Kind: "select", Flag: "--effort", Options: []string{"low", "medium", "high", "xhigh", "max"}, Detail: "Claude reasoning effort for the next managed launch"},
			{ID: "gateway-url", Label: "Gateway URL", Kind: "input", Flag: "--gateway-url", Detail: "launch against an already-running fak serve gateway"},
			{ID: "model", Label: "Model / Version", Kind: "select", Flag: "--model", Options: []string{"claude-opus-5", "claude-opus-4-8", "claude-fable-5", "claude-sonnet-5"}, Detail: "standard Claude model/version for the next managed launch; direct --model still accepts arbitrary IDs"},
			{ID: "permission-mode", Label: "Permission Mode", Kind: "input", Flag: "--permission-mode", Default: "bypassPermissions"},
			{ID: "target", Label: "Target", Kind: "input", Flag: "--target", Detail: "select a named compute target"},
		},
		Run: runTUIAgent,
	})
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "guard",
		Summary: "render guard/adjudication proof artifacts or the live decision journal",
		Usage:   "fak console guard --guard-json FILE [--journal FILE|--tail] [--follow] [--color auto|always|never]",
		Schema:  tuiGuardSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "color", Label: "Color", Kind: "flag", Flag: "--color", Default: "auto", Options: []string{"auto", "always", "never"}, Detail: "auto, always, or never; NO_COLOR disables color"},
			{ID: "follow", Label: "Follow", Kind: "toggle", Flag: "--follow", Detail: "keep printing new journal decisions"},
			{ID: "journal", Label: "Journal", Kind: "input", Flag: "--journal", Detail: "read the hash-chained guard journal"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed pane model"},
			{ID: "rows", Label: "Rows", Kind: "input", Flag: "--rows", Default: "50", Detail: "cap rendered journal rows"},
			{ID: "tail", Label: "Tail", Kind: "action", Flag: "--tail", Detail: "resolve and read the canonical guard journal"},
		},
		Run:      runTUIGuard,
		Overview: tuiOverviewAdapter(buildTUIOverviewGuardCard),
	})
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "overview",
		Summary: "compose selected pane models into one ranked operator spine",
		Usage:   "fak console overview [--pane ID ...] [--console-config FILE] [--issues-json FILE] [--ledger FILE] [--sessions-json FILE] [--garden-json FILE] [--savings-ledger FILE] [--guard-json FILE ...]",
		Schema:  tuiOverviewSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "check", Label: "Check Garden", Kind: "toggle", Flag: "--check"},
			{ID: "console-config", Label: "Console Config", Kind: "input", Flag: "--console-config", Default: "~/.fak/console.json", Detail: "loads saved overview_panes unless --pane is set"},
			{ID: "garden-json", Label: "Garden JSON", Kind: "input", Flag: "--garden-json"},
			{ID: "guard-json", Label: "Guard JSON", Kind: "input", Flag: "--guard-json"},
			{ID: "issues-json", Label: "Issues JSON", Kind: "input", Flag: "--issues-json"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed overview model"},
			{ID: "ledger", Label: "Loop Ledger", Kind: "input", Flag: "--ledger"},
			{ID: "pane", Label: "Pane", Kind: "input", Flag: "--pane", Detail: "repeat to choose overview pane subset and display order"},
			{ID: "savings-ledger", Label: "Savings Ledger", Kind: "input", Flag: "--savings-ledger", Detail: "Track-2 OBSERVED-$ ledger for the above-the-fold savings hero"},
			{ID: "sessions-json", Label: "Sessions JSON", Kind: "input", Flag: "--sessions-json"},
		},
		Run: runTUIOverview,
	})
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "panes",
		Summary: "list registered console panes and their operator controls",
		Usage:   "fak console panes [--json]",
		Schema:  tuiRegistrySchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the pane registry model"},
		},
		Run: runTUIPanes,
	})
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "sessions",
		Summary: "render live gateway session state: budgets, pace, lineage, and reasons",
		Usage:   "fak console sessions [--sessions-json FILE] [--addr URL] [--key K] [--top N] [--json]",
		Schema:  tuiSessionsSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "addr", Label: "Gateway", Kind: "input", Flag: "--addr", Detail: "gateway base URL"},
			{ID: "at", Label: "At", Kind: "input", Flag: "--at", Detail: "snapshot time"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed pane model"},
			{ID: "key", Label: "Bearer", Kind: "input", Flag: "--key", Detail: "bearer credential when required"},
			{ID: "top", Label: "Top Rows", Kind: "input", Flag: "--top", Default: "25"},
		},
		Run:      runTUISessions,
		Overview: tuiOverviewAdapter(buildTUIOverviewSessionCard),
	})
}

func runTUIPanes(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui panes", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the console pane registry as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console panes: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	report := buildTUIPaneRegistryReport()
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console panes")
	}
	fmt.Fprint(stdout, renderTUIPaneRegistry(report, 120))
	return 0
}

func buildTUIPaneRegistryReport() tuiPaneRegistryReport {
	descs := tuiplugin.Descriptors()
	counts := tuiPaneRegistryCounts{Panes: len(descs)}
	for _, d := range descs {
		if d.BuiltIn {
			counts.BuiltIn++
		}
		if d.Overview {
			counts.Overview++
		}
		counts.Controls += len(d.Controls)
	}
	return tuiPaneRegistryReport{
		Schema: tuiRegistrySchema,
		Counts: counts,
		Panes:  descs,
	}
}

func renderTUIPaneRegistry(report tuiPaneRegistryReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console panes  schema=%s\n", report.Schema)
	fmt.Fprintf(&b, "panes=%d  built_in=%d  overview=%d  controls=%d\n", report.Counts.Panes, report.Counts.BuiltIn, report.Counts.Overview, report.Counts.Controls)
	if len(report.Panes) == 0 {
		fmt.Fprintln(&b, "\nno panes registered")
		return b.String()
	}
	fmt.Fprintln(&b, "\nPanes")
	fmt.Fprintln(&b, "pane          schema                  ov controls summary")
	for _, pane := range report.Panes {
		overview := "-"
		if pane.Overview {
			overview = "yes"
		}
		fmt.Fprintf(&b, "%s %s %s %8d %s\n",
			padRightTUI(trimTUI(pane.ID, 12), 12),
			padRightTUI(trimTUI(pane.Schema, 23), 23),
			padRightTUI(overview, 3),
			len(pane.Controls),
			trimTUI(pane.Summary, maxTUI(20, width-52)))
	}
	fmt.Fprintln(&b, "\nControls")
	for _, pane := range report.Panes {
		if len(pane.Controls) == 0 {
			continue
		}
		fmt.Fprintf(&b, "%s:\n", pane.ID)
		for _, c := range pane.Controls {
			target := c.Flag
			if target == "" {
				target = c.Command
			}
			if target == "" {
				target = "-"
			}
			detail := c.Detail
			if c.Default != "" {
				if detail != "" {
					detail += "; "
				}
				detail += "default=" + c.Default
			}
			if len(c.Options) > 0 {
				if detail != "" {
					detail += "; "
				}
				detail += "options=" + strings.Join(c.Options, "|")
			}
			fmt.Fprintf(&b, "  %s %s %s %s\n",
				padRightTUI(trimTUI(c.ID, 16), 16),
				padRightTUI(trimTUI(c.Kind, 8), 8),
				padRightTUI(trimTUI(target, 18), 18),
				trimTUI(detail, maxTUI(16, width-48)))
		}
	}
	return b.String()
}
