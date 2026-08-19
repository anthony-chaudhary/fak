package main

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const infoFleetRowCap = 8

type infoFleetWorkspace struct {
	State       string                `json:"state"`
	GeneratedAt string                `json:"generated_at,omitempty"`
	Configured  int                   `json:"configured"`
	Shown       int                   `json:"shown"`
	Loops       []fleetpane.LoopCheck `json:"loops,omitempty"`
	Next        string                `json:"next,omitempty"`
}

func collectInfoFleetWorkspace(root string, runner fleetpane.Runner, now time.Time) *infoFleetWorkspace {
	out := &infoFleetWorkspace{State: "EMPTY", Next: "fak fleetpane loop-list"}
	cfg, err := fleetpane.LoadConfig(root)
	if err != nil {
		out.State = "UNAVAILABLE"
		out.Next = "fak fleetpane loop-list"
		return out
	}
	list := fleetpane.LoopList(cfg, fleetpane.Options{Runner: runner, Now: func() time.Time { return now }})
	var names []string
	for _, item := range list.Loops {
		if item.Enabled {
			names = append(names, item.Name)
		}
	}
	out.Configured = len(names)
	out.GeneratedAt = now.UTC().Format(time.RFC3339)
	if len(names) == 0 {
		return out
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	checks := make([]fleetpane.LoopCheck, len(names))
	var wg sync.WaitGroup
	for i, name := range names {
		wg.Add(1)
		go func(i int, name string) {
			defer wg.Done()
			checks[i] = fleetpane.CollectLoop(ctx, name, cfg.Loops[name], cfg, fleetpane.Options{Runner: runner, Now: func() time.Time { return now }})
		}(i, name)
	}
	wg.Wait()
	sort.SliceStable(checks, func(i, j int) bool {
		ai, aj := fleetpane.LoopCheckNeedsAction(checks[i]), fleetpane.LoopCheckNeedsAction(checks[j])
		if ai != aj {
			return ai
		}
		return checks[i].Name < checks[j].Name
	})
	if len(checks) > infoFleetRowCap {
		checks = checks[:infoFleetRowCap]
	}
	out.State = "READY"
	out.Loops = checks
	out.Shown = len(checks)
	out.Next = ""
	return out
}

func fleetWorkspaceRows(v guardInfoVars) []string {
	rows := []string{"FLEET WORKSPACE · read-only"}
	if v.Fleet != nil {
		rows = append(rows, guardInfoFleetHeadText(v.Fleet), guardInfoFleetTotalsText(v.Fleet))
	} else {
		rows = append(rows, "machine aggregate · unavailable · next: check /debug/vars.fleet")
	}
	w := v.FleetWorkspace
	if w == nil {
		return append(rows, "loops · unavailable · next: fak fleetpane loop-list")
	}
	rows = append(rows, fmt.Sprintf("loops · %s · configured %d · showing %d · source fleetpane @ %s", w.State, w.Configured, w.Shown, blankAs(w.GeneratedAt, "unknown")))
	if len(w.Loops) == 0 {
		return append(rows, fmt.Sprintf("no loop checks · next: %s", blankAs(w.Next, "fak fleetpane loop-list")))
	}
	for _, check := range w.Loops {
		detail := fleetpane.LoopAuditDetail(check)
		if detail == "" {
			detail = blankAs(check.Reason, "no progress evidence")
		}
		next := "none"
		if fleetpane.LoopCheckNeedsAction(check) {
			next = fmt.Sprintf("fak fleetpane loop-check %s", check.Name)
		}
		rows = append(rows, fmt.Sprintf("%s · %s · %s · next: %s", check.Name, check.State, oneLine(detail), next))
	}
	return rows
}
func blankAs(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return strings.TrimSpace(s)
}
func oneLine(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) > 120 {
		return s[:117] + "..."
	}
	return s
}

func runInfoFleetSelfcheck(stdout io.Writer, width, height int) int {
	if width <= 0 {
		width = 100
	}
	if height <= 0 {
		height = 16
	}
	v := guardInfoVars{
		Fleet: &gateway.SessionFleet{Verdict: "ACTION", Machines: 2, Action: 1, Sessions: 4},
		FleetWorkspace: &infoFleetWorkspace{State: "READY", GeneratedAt: "2026-08-18T20:00:00Z", Configured: 2, Shown: 2, Loops: []fleetpane.LoopCheck{
			{Name: "stuck-loop", Enabled: true, State: "ACTION", Detail: "no progress age=48m"},
			{Name: "healthy-loop", Enabled: true, State: "OK", Detail: "fresh progress age=2m"},
		}},
	}
	fmt.Fprintln(stdout, renderGuardInfoInteractiveBlock(infoViewState{active: viewFleet}, v, nil, width, height))
	fmt.Fprintln(stdout, "SELFCHECK OK — Fleet workspace is read-only, attention-first, and every action has an existing CLI next step")
	return 0
}
