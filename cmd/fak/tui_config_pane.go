package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

const tuiConfigSchema = "fak.tui.config.v1"

type tuiConfigReport struct {
	Schema   string             `json:"schema"`
	At       string             `json:"at"`
	Path     string             `json:"path,omitempty"`
	Status   string             `json:"status"` // missing|loaded|error
	Updated  bool               `json:"updated,omitempty"`
	Error    string             `json:"error,omitempty"`
	Counts   tuiConfigCounts    `json:"counts"`
	Panes    []string           `json:"overview_panes,omitempty"`
	Defaults []tuiConfigDefault `json:"defaults,omitempty"`
}

type tuiConfigCounts struct {
	OverviewPanes int `json:"overview_panes"`
	PaneDefaults  int `json:"pane_defaults"`
}

type tuiConfigDefault struct {
	Pane    string   `json:"pane"`
	Control string   `json:"control"`
	Flag    string   `json:"flag,omitempty"`
	Value   string   `json:"value"`
	Tags    []string `json:"tags,omitempty"`
}

func init() {
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "config",
		Summary: "inspect console config, saved overview panes, and pane defaults",
		Usage:   "fak console config [--path FILE] [--json] [--set-overview ID,ID] [--set-default PANE.CONTROL=VALUE]",
		Schema:  tuiConfigSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "clear-overview", Label: "Clear Overview", Kind: "action", Flag: "--clear-overview", Detail: "remove the saved overview pane order"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed config model"},
			{ID: "path", Label: "Path", Kind: "input", Flag: "--path", Default: "~/.fak/console.json", Detail: "console config file to inspect"},
			{ID: "set-default", Label: "Set Default", Kind: "input", Flag: "--set-default", Detail: "save a pane control default as PANE.CONTROL=VALUE"},
			{ID: "set-overview", Label: "Set Overview", Kind: "input", Flag: "--set-overview", Detail: "save comma-separated overview pane order"},
			{ID: "unset-default", Label: "Unset Default", Kind: "input", Flag: "--unset-default", Detail: "remove a pane control default as PANE.CONTROL"},
		},
		Run:      runTUIConfig,
		Overview: tuiOverviewAdapter(buildTUIOverviewConfigCard),
	})
}

func runTUIConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("path", defaultTUIConsoleFile(), "console preference JSON (default: FAK_CONSOLE_FILE, else ~/.fak/console.json)")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the config TUI model as JSON")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	setOverview := fs.String("set-overview", "", "save comma-separated overview panes in display order")
	clearOverview := fs.Bool("clear-overview", false, "remove the saved overview pane order")
	var setDefaults multiFlag
	var unsetDefaults multiFlag
	fs.Var(&setDefaults, "set-default", "save a pane control default as PANE.CONTROL=VALUE (repeatable)")
	fs.Var(&unsetDefaults, "unset-default", "remove a pane control default as PANE.CONTROL (repeatable)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console config: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console config: %v\n", err)
		return 2
	}
	if configMutationRequested(*setOverview, *clearOverview, []string(setDefaults), []string(unsetDefaults)) {
		report, err := mutateTUIConfig(*path, tuiConfigMutation{
			SetOverview:   *setOverview,
			ClearOverview: *clearOverview,
			SetDefaults:   []string(setDefaults),
			UnsetDefaults: []string(unsetDefaults),
		}, at)
		if err != nil {
			fmt.Fprintf(stderr, "fak console config: %v\n", err)
			return 2
		}
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, report, "fak console config")
		}
		fmt.Fprint(stdout, renderTUIConfig(report, *width))
		return 0
	}
	report := buildTUIConfigReport(*path, at)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console config")
	}
	fmt.Fprint(stdout, renderTUIConfig(report, *width))
	return 0
}

type tuiConfigMutation struct {
	SetOverview   string
	ClearOverview bool
	SetDefaults   []string
	UnsetDefaults []string
}

func configMutationRequested(setOverview string, clearOverview bool, setDefaults, unsetDefaults []string) bool {
	return strings.TrimSpace(setOverview) != "" || clearOverview || len(setDefaults) > 0 || len(unsetDefaults) > 0
}

func mutateTUIConfig(path string, mut tuiConfigMutation, at time.Time) (tuiConfigReport, error) {
	if mut.ClearOverview && strings.TrimSpace(mut.SetOverview) != "" {
		return tuiConfigReport{}, fmt.Errorf("--clear-overview cannot be combined with --set-overview")
	}
	cfg, _, err := loadTUIConsoleConfig(path)
	if err != nil {
		return tuiConfigReport{}, err
	}
	if mut.ClearOverview {
		cfg.OverviewPanes = nil
	}
	if strings.TrimSpace(mut.SetOverview) != "" {
		cfg.OverviewPanes = normalizeTUIOverviewPaneList(splitTUIConfigCSV(mut.SetOverview))
	}
	if len(mut.SetDefaults) > 0 && cfg.PaneDefaults == nil {
		cfg.PaneDefaults = map[string]map[string]json.RawMessage{}
	}
	for _, spec := range mut.SetDefaults {
		pane, control, value, err := parseTUIConfigDefaultSet(spec)
		if err != nil {
			return tuiConfigReport{}, err
		}
		if cfg.PaneDefaults[pane] == nil {
			cfg.PaneDefaults[pane] = map[string]json.RawMessage{}
		}
		cfg.PaneDefaults[pane][control] = value
	}
	for _, spec := range mut.UnsetDefaults {
		pane, control, err := parseTUIConfigDefaultRef(spec)
		if err != nil {
			return tuiConfigReport{}, err
		}
		if controls := cfg.PaneDefaults[pane]; controls != nil {
			delete(controls, control)
			if len(controls) == 0 {
				delete(cfg.PaneDefaults, pane)
			}
		}
	}
	source, err := saveTUIConsoleConfig(path, cfg)
	if err != nil {
		return tuiConfigReport{}, err
	}
	report := buildTUIConfigReport(source, at)
	report.Updated = true
	return report, nil
}

func splitTUIConfigCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		out = append(out, strings.TrimSpace(part))
	}
	return out
}

func parseTUIConfigDefaultSet(spec string) (string, string, json.RawMessage, error) {
	key, rawValue, ok := strings.Cut(spec, "=")
	if !ok {
		return "", "", nil, fmt.Errorf("--set-default requires PANE.CONTROL=VALUE")
	}
	pane, control, err := parseTUIConfigDefaultRef(key)
	if err != nil {
		return "", "", nil, err
	}
	return pane, control, encodeTUIConfigDefaultValue(rawValue), nil
}

func parseTUIConfigDefaultRef(spec string) (string, string, error) {
	pane, control, ok := strings.Cut(strings.TrimSpace(spec), ".")
	if !ok {
		return "", "", fmt.Errorf("default reference %q must be PANE.CONTROL", spec)
	}
	pane = strings.TrimSpace(strings.ToLower(pane))
	control = strings.TrimSpace(strings.ToLower(control))
	if pane == "" || control == "" {
		return "", "", fmt.Errorf("default reference %q must be PANE.CONTROL", spec)
	}
	return pane, control, nil
}

func encodeTUIConfigDefaultValue(value string) json.RawMessage {
	value = strings.TrimSpace(value)
	if value == "" {
		return json.RawMessage(`""`)
	}
	if json.Valid([]byte(value)) {
		return json.RawMessage(value)
	}
	b, _ := json.Marshal(value)
	return json.RawMessage(b)
}

func buildTUIConfigReport(path string, at time.Time) tuiConfigReport {
	report := tuiConfigReport{
		Schema: tuiConfigSchema,
		At:     at.UTC().Format(time.RFC3339),
		Path:   strings.TrimSpace(pathutil.ExpandTilde(path)),
		Status: "missing",
	}
	if report.Path == "" {
		report.Status = "missing"
		return report
	}
	if _, err := os.Stat(report.Path); err != nil {
		if os.IsNotExist(err) {
			return report
		}
		report.Status = "error"
		report.Error = err.Error()
		return report
	}
	cfg, source, err := loadTUIConsoleConfig(report.Path)
	if err != nil {
		report.Status = "error"
		report.Error = err.Error()
		return report
	}
	if source != "" {
		report.Path = source
	}
	report.Status = "loaded"
	report.Panes = append([]string(nil), cfg.OverviewPanes...)
	report.Defaults = buildTUIConfigDefaults(cfg)
	report.Counts.OverviewPanes = len(report.Panes)
	report.Counts.PaneDefaults = len(report.Defaults)
	return report
}

func buildTUIConfigDefaults(cfg tuiConsoleConfig) []tuiConfigDefault {
	if len(cfg.PaneDefaults) == 0 {
		return nil
	}
	panes := map[string]tuiplugin.Descriptor{}
	for _, desc := range tuiplugin.Descriptors() {
		panes[desc.ID] = desc
	}
	paneIDs := make([]string, 0, len(cfg.PaneDefaults))
	for pane := range cfg.PaneDefaults {
		paneIDs = append(paneIDs, pane)
	}
	sort.Strings(paneIDs)
	rows := []tuiConfigDefault{}
	for _, paneID := range paneIDs {
		controls := map[string]tuiplugin.Control{}
		if desc, ok := panes[paneID]; ok {
			for _, c := range desc.Controls {
				controls[c.ID] = c
			}
		}
		controlIDs := make([]string, 0, len(cfg.PaneDefaults[paneID]))
		for id := range cfg.PaneDefaults[paneID] {
			controlIDs = append(controlIDs, id)
		}
		sort.Strings(controlIDs)
		for _, controlID := range controlIDs {
			c := controls[controlID]
			rows = append(rows, tuiConfigDefault{
				Pane:    paneID,
				Control: controlID,
				Flag:    c.Flag,
				Value:   compactTUIConfigDefaultValue(cfg.PaneDefaults[paneID][controlID]),
				Tags:    configDefaultTags(c),
			})
		}
	}
	return rows
}

func compactTUIConfigDefaultValue(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	raw = json.RawMessage([]byte(trimmed))
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return string(raw)
}

func configDefaultTags(c tuiplugin.Control) []string {
	tags := []string{}
	if c.ID == "" {
		tags = append(tags, "unknown-control")
	}
	if c.Flag == "" {
		tags = append(tags, "no-flag")
	}
	if c.Flag == "--console-config" {
		tags = append(tags, "dispatcher-only")
	}
	return tags
}

func renderTUIConfig(report tuiConfigReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console config  at=%s  status=%s\n", report.At, report.Status)
	if report.Updated {
		fmt.Fprintln(&b, "updated=yes")
	}
	if report.Path != "" {
		fmt.Fprintf(&b, "path=%s\n", trimTUI(report.Path, maxTUI(20, width-5)))
	}
	if report.Error != "" {
		fmt.Fprintf(&b, "error=%s\n", trimTUI(report.Error, maxTUI(20, width-6)))
	}
	fmt.Fprintf(&b, "overview_panes=%d  pane_defaults=%d\n", report.Counts.OverviewPanes, report.Counts.PaneDefaults)
	if len(report.Panes) > 0 {
		fmt.Fprintln(&b, "\nOverview Panes")
		for i, pane := range report.Panes {
			fmt.Fprintf(&b, "%2d  %s\n", i+1, pane)
		}
	}
	if len(report.Defaults) > 0 {
		fmt.Fprintln(&b, "\nPane Defaults")
		fmt.Fprintln(&b, "pane         control          flag               value tags")
		for _, row := range report.Defaults {
			tags := strings.Join(row.Tags, ",")
			if tags == "" {
				tags = "-"
			}
			fmt.Fprintf(&b, "%s %s %s %s %s\n",
				padRightTUI(trimTUI(row.Pane, 12), 12),
				padRightTUI(trimTUI(row.Control, 16), 16),
				padRightTUI(trimTUI(blankTUI(row.Flag), 18), 18),
				padRightTUI(trimTUI(row.Value, 16), 16),
				trimTUI(tags, maxTUI(8, width-68)))
		}
	}
	return b.String()
}

func buildTUIOverviewConfigCard(opt tuiOverviewOptions) (tuiOverviewCard, error) {
	report := buildTUIConfigReport(defaultTUIConsoleFile(), opt.At)
	counts := map[string]int{
		"overview_panes": report.Counts.OverviewPanes,
		"pane_defaults":  report.Counts.PaneDefaults,
	}
	status := "ok"
	tags := []string{"console-config"}
	attention := 0
	switch report.Status {
	case "missing":
		tags = append(tags, "defaults")
	case "error":
		status = "action"
		attention = 90
		tags = append(tags, "config-error")
	}
	summary := fmt.Sprintf("status=%s overview_panes=%d pane_defaults=%d", report.Status, report.Counts.OverviewPanes, report.Counts.PaneDefaults)
	if report.Error != "" {
		summary += " error=" + report.Error
	}
	return tuiOverviewCard{
		Pane:      "config",
		Status:    status,
		Source:    report.Path,
		Summary:   summary,
		Command:   "fak console config",
		Attention: attention,
		Counts:    counts,
		Tags:      tags,
	}, nil
}
