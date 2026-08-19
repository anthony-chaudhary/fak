package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
	"github.com/anthony-chaudhary/fak/internal/fleetpane"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/waiting"
)

const infoFleetRowCap = 8

type infoFleetWorkspace struct {
	State       string                `json:"state"`
	GeneratedAt string                `json:"generated_at,omitempty"`
	Configured  int                   `json:"configured"`
	Shown       int                   `json:"shown"`
	Loops       []fleetpane.LoopCheck `json:"loops,omitempty"`
	Pressure    *infoFleetPressure    `json:"pressure,omitempty"`
	Next        string                `json:"next,omitempty"`
}

type infoFleetLaneFold struct {
	Lane    string
	Waiting waiting.Queue
	Aging   dispatchaging.Result
}

type infoFleetLanePressure struct {
	Lane              string  `json:"lane"`
	OldestWaitSeconds float64 `json:"oldest_wait_seconds"`
	Starved           int     `json:"starved"`
}

type infoFleetPressure struct {
	Verdict string                  `json:"verdict"`
	Lanes   []infoFleetLanePressure `json:"lanes"`
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
	if folds := infoFleetLaneFoldsFromChecks(checks); len(folds) > 0 {
		pressure := foldInfoFleetLanePressure(folds, dispatchaging.DefaultStarvationSeconds)
		out.Pressure = &pressure
	}
	if len(checks) > infoFleetRowCap {
		checks = checks[:infoFleetRowCap]
	}
	out.State = "READY"
	out.Loops = checks
	out.Shown = len(checks)
	out.Next = ""
	return out
}

// infoFleetLaneFoldsFromChecks reuses typed folds already returned by configured fleet
// loop checks. It performs no ledger read and ignores unrelated payloads.
func infoFleetLaneFoldsFromChecks(checks []fleetpane.LoopCheck) []infoFleetLaneFold {
	var out []infoFleetLaneFold
	for _, check := range checks {
		lane, _ := check.Payload["lane"].(string)
		lane = strings.TrimSpace(lane)
		if lane == "" {
			continue
		}
		var fold infoFleetLaneFold
		fold.Lane = lane
		waitingOK := decodeInfoFleetFold(check.Payload["waiting"], &fold.Waiting)
		agingOK := decodeInfoFleetFold(check.Payload["aging"], &fold.Aging)
		if waitingOK || agingOK {
			out = append(out, fold)
		}
	}
	return out
}

func decodeInfoFleetFold(src any, dst any) bool {
	if src == nil {
		return false
	}
	raw, err := json.Marshal(src)
	return err == nil && json.Unmarshal(raw, dst) == nil
}

// foldInfoFleetLanePressure joins waiting's per-lane oldest age with dispatchaging's
// closed Standing verdicts. The aggregate becomes STARVING at the hard deadline.
func foldInfoFleetLanePressure(folds []infoFleetLaneFold, starvationSeconds int64) infoFleetPressure {
	if starvationSeconds <= 0 {
		starvationSeconds = dispatchaging.DefaultStarvationSeconds
	}
	byLane := map[string]*infoFleetLanePressure{}
	for _, fold := range folds {
		lane := strings.TrimSpace(fold.Lane)
		if lane == "" {
			continue
		}
		row := byLane[lane]
		if row == nil {
			row = &infoFleetLanePressure{Lane: lane}
			byLane[lane] = row
		}
		if fold.Waiting.OldestAgeSeconds > row.OldestWaitSeconds {
			row.OldestWaitSeconds = fold.Waiting.OldestAgeSeconds
		}
		for _, unit := range fold.Aging.Order {
			if unit.Standing == dispatchaging.StandingStarved {
				row.Starved++
			}
		}
	}

	pressure := infoFleetPressure{Verdict: "OK", Lanes: make([]infoFleetLanePressure, 0, len(byLane))}
	for _, row := range byLane {
		pressure.Lanes = append(pressure.Lanes, *row)
		if row.OldestWaitSeconds >= float64(starvationSeconds) {
			pressure.Verdict = "STARVING"
		}
	}
	sort.SliceStable(pressure.Lanes, func(i, j int) bool {
		if pressure.Lanes[i].OldestWaitSeconds != pressure.Lanes[j].OldestWaitSeconds {
			return pressure.Lanes[i].OldestWaitSeconds > pressure.Lanes[j].OldestWaitSeconds
		}
		return pressure.Lanes[i].Lane < pressure.Lanes[j].Lane
	})
	return pressure
}

func fleetWorkspaceRows(v guardInfoVars) []string {
	heading := "FLEET WORKSPACE · read-only"
	if v.FleetWorkspace != nil && v.FleetWorkspace.Pressure != nil {
		heading = fmt.Sprintf("FLEET WORKSPACE · %s · read-only", v.FleetWorkspace.Pressure.Verdict)
	}
	rows := []string{heading}
	if v.Fleet != nil {
		rows = append(rows, guardInfoFleetHeadText(v.Fleet), guardInfoFleetTotalsText(v.Fleet))
	} else {
		rows = append(rows, "machine aggregate · unavailable · next: check /debug/vars.fleet")
	}
	w := v.FleetWorkspace
	if w == nil {
		return append(rows, "loops · unavailable · next: fak fleetpane loop-list")
	}
	if w.Pressure != nil {
		for _, lane := range w.Pressure.Lanes {
			rows = append(rows, fmt.Sprintf("%s · oldest-wait %s · starved %d", lane.Lane, humanDur(int64(lane.OldestWaitSeconds)), lane.Starved))
		}
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
		FleetWorkspace: &infoFleetWorkspace{State: "READY", GeneratedAt: "2026-08-18T20:00:00Z", Configured: 2, Shown: 2, Pressure: &infoFleetPressure{Verdict: "STARVING", Lanes: []infoFleetLanePressure{
			{Lane: "cmd", OldestWaitSeconds: 7 * 3600, Starved: 2},
			{Lane: "docs", OldestWaitSeconds: 2 * 3600},
		}}, Loops: []fleetpane.LoopCheck{
			{Name: "stuck-loop", Enabled: true, State: "ACTION", Detail: "no progress age=48m"},
			{Name: "healthy-loop", Enabled: true, State: "OK", Detail: "fresh progress age=2m"},
		}},
	}
	fmt.Fprintln(stdout, renderGuardInfoInteractiveBlock(infoViewState{active: viewFleet}, v, nil, width, height))
	fmt.Fprintln(stdout, "SELFCHECK OK — Fleet workspace is read-only, starvation-first, and every action has an existing CLI next step")
	return 0
}
