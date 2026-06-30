package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const windowgateSchema = "fak-windowgate/1"

type windowgatePayload struct {
	Schema      string         `json:"schema"`
	OK          bool           `json:"ok"`
	Verdict     string         `json:"verdict"`
	Finding     string         `json:"finding"`
	Reason      string         `json:"reason"`
	NextAction  string         `json:"next_action"`
	Workspace   string         `json:"workspace"`
	Counts      map[string]int `json:"counts"`
	Suppression map[string]int `json:"suppression,omitempty"`
	Violations  []string       `json:"violations"`
	Watchlist   []string       `json:"watchlist,omitempty"`
	Tools       map[string]int `json:"watchlist_tools,omitempty"`
	Files       map[string]int `json:"watchlist_files,omitempty"`
	Dirs        map[string]int `json:"watchlist_dirs,omitempty"`
}

func cmdWindowgate(argv []string) { os.Exit(runWindowgate(os.Stdout, os.Stderr, argv)) }

func runWindowgate(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && (argv[0] == "scan" || argv[0] == "report") {
		argv = argv[1:]
	}
	fs := flag.NewFlagSet("windowgate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	failCandidates := fs.Bool("fail-on-candidates", false, "also exit non-zero when advisory console-tool candidates remain")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak windowgate: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := strings.TrimSpace(*workspace)
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	rep, err := windowgate.ScanTree(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak windowgate: %v\n", err)
		return 1
	}
	payload := buildWindowgatePayload(root, rep, *failCandidates)
	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak windowgate: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderWindowgate(payload))
	}
	if !payload.OK {
		return 1
	}
	return 0
}

func buildWindowgatePayload(root string, rep windowgate.Report, failCandidates bool) windowgatePayload {
	violations := append([]string{}, rep.PSInstallers...)
	violations = append(violations, rep.PySpawns...)
	violations = append(violations, rep.GoExecs...)
	watchlist := append([]string{}, rep.PyCandidates...)
	watchlist = append(watchlist, rep.GoCandidates...)
	counts := map[string]int{
		"ps_installers":       len(rep.PSInstallers),
		"py_spawns":           len(rep.PySpawns),
		"py_watchlist":        len(rep.PyCandidates),
		"go_execs":            len(rep.GoExecs),
		"go_watchlist":        len(rep.GoCandidates),
		"py_explicit_modules": len(rep.PyExplicitModules),
		"py_default_modules":  len(rep.PyDefaultModules),
	}
	p := windowgatePayload{
		Schema:      windowgateSchema,
		OK:          len(violations) == 0,
		Verdict:     "OK",
		Finding:     "no_desktop_popup_clear",
		Reason:      "hard-ratcheted scheduled-task, Python, and Go background helper paths are window-suppressed",
		NextAction:  "keep new background subprocesses on windowgate.ConfigureBackgroundCommand or the Python no_window_creationflags helper",
		Workspace:   root,
		Counts:      counts,
		Suppression: suppressionCounts(rep),
		Violations:  violations,
		Watchlist:   watchlist,
		Tools:       watchlistToolCounts(watchlist),
		Files:       watchlistFileCounts(watchlist),
		Dirs:        watchlistDirCounts(watchlist),
	}
	if len(violations) > 0 {
		p.OK = false
		p.Verdict = "ACTION"
		p.Finding = "no_desktop_popup_regression"
		p.Reason = fmt.Sprintf("%d hard popup regression(s): background helpers can still flash visible console windows", len(violations))
		p.NextAction = "make scheduled tasks off-desktop/headless, add Python creationflags, or call windowgate.ConfigureBackgroundCommand before running Go helper commands"
		return p
	}
	if len(watchlist) > 0 {
		p.Finding = "no_desktop_popup_watchlist"
		p.Reason = fmt.Sprintf("hard popup gate is clean; %d console-tool launch(es) remain on the advisory watchlist", len(watchlist))
		p.NextAction = "review the watchlist and either classify each launch as foreground/interactive or route helper calls through windowgate.ConfigureBackgroundCommand"
		if failCandidates {
			p.OK = false
			p.Verdict = "ACTION"
		}
	}
	return p
}

func renderWindowgate(p windowgatePayload) string {
	var b strings.Builder
	status := "OK"
	if !p.OK {
		status = "ACTION"
	}
	fmt.Fprintf(&b, "windowgate: %s (%s)\n", status, p.Finding)
	fmt.Fprintf(&b, "workspace: %s\n", p.Workspace)
	fmt.Fprintf(&b, "hard: ps=%d py=%d go=%d  watchlist: py=%d go=%d\n",
		p.Counts["ps_installers"], p.Counts["py_spawns"], p.Counts["go_execs"], p.Counts["py_watchlist"], p.Counts["go_watchlist"])
	if len(p.Suppression) > 0 {
		fmt.Fprintf(&b, "suppression: %s\n", renderToolCounts(p.Suppression))
	}
	if len(p.Violations) > 0 {
		b.WriteString("\nviolations:\n")
		for _, row := range p.Violations {
			fmt.Fprintf(&b, "  - %s\n", row)
		}
	}
	if len(p.Watchlist) > 0 {
		if len(p.Tools) > 0 {
			fmt.Fprintf(&b, "watchlist tools: %s\n", renderToolCounts(p.Tools))
		}
		if len(p.Dirs) > 0 {
			fmt.Fprintf(&b, "watchlist dirs: %s\n", renderToolCounts(p.Dirs))
		}
		b.WriteString("\nwatchlist:\n")
		for _, row := range p.Watchlist {
			fmt.Fprintf(&b, "  - %s\n", row)
		}
	}
	fmt.Fprintf(&b, "\nreason: %s\nnext: %s", p.Reason, p.NextAction)
	return b.String()
}

func suppressionCounts(rep windowgate.Report) map[string]int {
	out := map[string]int{}
	if len(rep.PyExplicitModules) > 0 {
		out["py_explicit_modules"] = len(rep.PyExplicitModules)
	}
	if len(rep.PyDefaultModules) > 0 {
		out["py_default_modules"] = len(rep.PyDefaultModules)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

var (
	windowgatePyToolRE = regexp.MustCompile(`subprocess ([^ ]+) launch`)
	windowgateGoToolRE = regexp.MustCompile(`exec\.Command(?:Context)?\([^"]*"([^"]+)"`)
)

func watchlistToolCounts(rows []string) map[string]int {
	out := map[string]int{}
	for _, row := range rows {
		tool := ""
		if m := windowgatePyToolRE.FindStringSubmatch(row); m != nil {
			tool = m[1]
		} else if m := windowgateGoToolRE.FindStringSubmatch(row); m != nil {
			tool = filepath.Base(strings.ReplaceAll(m[1], "\\", "/"))
		}
		tool = strings.ToLower(strings.TrimSpace(tool))
		if tool != "" {
			out[tool]++
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func watchlistFileCounts(rows []string) map[string]int {
	out := map[string]int{}
	for _, row := range rows {
		file := watchlistRowFile(row)
		if file != "" {
			out[file]++
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func watchlistDirCounts(rows []string) map[string]int {
	out := map[string]int{}
	for _, row := range rows {
		file := watchlistRowFile(row)
		if file == "" {
			continue
		}
		dir := "."
		if i := strings.LastIndex(file, "/"); i >= 0 {
			dir = file[:i]
		}
		out[dir]++
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func watchlistRowFile(row string) string {
	i := strings.Index(row, ":")
	if i <= 0 {
		return ""
	}
	return filepath.ToSlash(row[:i])
}

func renderToolCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, counts[k]))
	}
	return strings.Join(parts, " ")
}
