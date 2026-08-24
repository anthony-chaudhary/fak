package main

// dispatch_sweep.go — `fak dispatch sweep`, the queue-drain front door: find the next open
// issue, spawn one guarded worker, repeat — until a tick refuses (at cap / queue drained / no
// seat) or the best-effort --max-agents ceiling is hit.
//
//	fak dispatch sweep                                 # plan the next tick (dry-run)
//	fak dispatch sweep --live                          # drain to cap, best effort
//	fak dispatch sweep --live --max-agents 100 --max-workers 6
//	fak dispatch sweep --live --lane gateway --backend opencode --json
//
// The loop is pure (internal/dispatchsweep.RunSweep). This shell does only the wire: it builds
// the per-iteration tick command — one real `fak dispatch tick` evaluation — runs it and feeds
// the result to the loop. EVERY spawn therefore still passes the
// same dispatch_preflight DoS gate, switcher account pin, in-flight #N de-dup, and loop-ledger
// append the single tick already enforces; the sweep adds only the repeat, and stops the
// instant a tick refuses. DRY-RUN BY DEFAULT — --live is what actually spawns.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchsweep"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/seatpark"
)

// dispatchSweepLoopID is the loop identity the sweep front-door records its bounded
// no-seat park-and-retry (#3523) under — distinct from garden-issue-dispatch, so the
// two dispatch loops keep independent park tails. A no-seat wall is fleet-wide (an
// account-seat cap, not a lane property), so ONE id per sweep front-door suffices.
const dispatchSweepLoopID = "dispatch-issue-sweep"

// deriveSweepSeatParkState folds the loop ledger's dispatch-sweep run-ends
// (newest→oldest) into the consecutive no-seat park count + most-recent no-seat time
// seatpark.Decide keys on. It is the dispatch-sweep binding of the SHARED fold
// deriveSeatParkStateForLoop (garden_dispatch.go) — the loop id was always the only
// difference from the garden-dispatch tail, so the two now run the same code rather than
// mirroring it by hand: a deferred run (SEAT_PARKED) is neutral in the tail; a no-seat stop
// counts; anything else (progress, an exhausted park, another stop) ends the tail so a
// fresh cycle starts from zero. Pure: ledger in, counts out.
func deriveSweepSeatParkState(events []loopmgr.Event) (parks int, lastParkUnix int64) {
	return deriveSeatParkStateForLoop(events, dispatchSweepLoopID)
}

// recordSweepSeatPark appends a dispatch-sweep run-end to the loop ledger carrying the
// park-tail Reason token deriveSweepSeatParkState reads back: seatParkReasonNoSeat when a
// LIVE sweep stopped on a seat refuse, the seatpark.Status string when this run chose to
// park/exhaust, or "" otherwise (which ends the tail). Best-effort; a ledger write fault
// never fails the sweep.
func recordSweepSeatPark(ledgerPath, reason, summary string) {
	if ledgerPath == "" {
		return
	}
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  dispatchSweepLoopID,
		RunID:   firstNonEmpty(os.Getenv("FAK_LOOP_RUN_ID"), fmt.Sprintf("dispatch-sweep-%d", time.Now().UnixNano())),
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusWitnessedDone,
		Source:  "fak dispatch sweep",
		Reason:  reason,
		Summary: summary,
	})
}

// runDispatchSweep is the testable core of `fak dispatch sweep`: it returns the process exit
// code (0 ok, 1 a runtime/tooling fault, 2 a usage error) and takes its streams explicitly.
func runDispatchSweep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch sweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	maxAgents := fs.Int("max-agents", 100, "best-effort UPPER bound on workers to spawn this sweep (the preflight cap is the real limiter)")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "preflight worker cap passed to each tick (the real DoS limiter)")
	backend := fs.String("backend", "codex", "worker backend for every tick (claude|opencode|codex); default codex")
	lane := fs.String("lane", "", "pin every tick to this lane (default: largest step-budget lane pick)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the step-budget pick")
	settleS := fs.Float64("settle-s", 8.0, "seconds to wait after a live spawn before the next tick (lets the worker's de-dup log appear)")
	tickTimeoutS := fs.Int("tick-timeout-s", 300, "per-tick subprocess timeout in seconds")
	noLedger := fs.Bool("no-loop-ledger", false, "pass --no-loop-ledger to each tick AND skip the sweep-level no-seat park ledger (hermetic probes)")
	ledger := fs.String("ledger", "", "loop JSONL ledger path for the bounded no-seat park-and-retry (#3523); default: the loop ledger")
	live := fs.Bool("live", false, "actually spawn workers and drain the queue")
	asJSON := fs.Bool("json", false, "emit the raw sweep Record JSON instead of the human card")
	if !parseFlags(fs, argv) {
		return 2
	}

	root := *workspace
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch sweep: getwd: %v\n", err)
			return 1
		}
		root = wd
	}

	cfg := dispatchsweep.Config{
		MaxAgents:  *maxAgents,
		MaxWorkers: *maxWorkers,
		Backend:    *backend,
		Lane:       *lane,
		Live:       *live,
	}
	_ = tickTimeoutS // kept for CLI compatibility; the Go tick no longer shells a timeout-bound subprocess.

	tick := func(iter int) (dispatchsweep.TickResult, error) {
		payload, err := evaluateDispatchTick(dispatchTickOptions{
			Workspace:    root,
			MaxWorkers:   *maxWorkers,
			WorkKind:     dispatchtickWorkKind(*backend),
			Lane:         *lane,
			Backend:      *backend,
			ExcludeLanes: splitCommaList(*excludeLane),
			Live:         *live,
			// Refresh the fleet registry (the ~40s tools/fleet_sessions.py scan) only on the
			// FIRST tick of the drain, then reuse it -- mirroring the wave execution loop's
			// `i == 0` cadence (dispatch_wave.go). The scan's result is surfaced only as the
			// payload's registry_refresh provenance field; seat admission reads LIVE pidfile
			// leases (dispatchLiveSeatLeases) and the on-disk roster directly, and the worker
			// this sweep just spawned is counted via its own pidfile regardless. So re-scanning
			// before every spawn added (N-1)x~40s of serial wall-clock to an N-spawn sweep
			// without changing a single admission decision. One scan per drain suffices.
			Refresh:        dispatchSweepRefresh(iter),
			CooldownMin:    dispatchtick.DefaultCooldownMinutes,
			WorkerTimeoutS: dispatchtick.DefaultWorkerTimeoutS,
			SpawnProbeS:    dispatchtick.DefaultSpawnProbeS,
			RecordLoop:     !*noLedger,
		}, stderr)
		if err != nil {
			return dispatchsweep.TickResult{}, err
		}
		return tickResultFromJSON(payload), nil
	}

	settle := func() {
		if *settleS > 0 {
			time.Sleep(time.Duration(*settleS * float64(time.Second)))
		}
	}

	// Gate 1.5: bounded no-seat park-and-retry (#3523). When recent LIVE sweeps stopped on a
	// seat refuse (REFUSE_NO_ACCOUNT / WEEKLY_CAPPED — no free Claude seat), re-invoking the
	// sweep just bursts against a wall only a peer finishing can move, and the burst's
	// preflight load can turn a clean seat-refuse into a REFUSE_INSPECT. So park until a
	// bounded backoff window elapses, then retry, up to a bounded budget — the durable loop
	// ledger IS the park-state store. A parked invocation returns HERE, before RunSweep runs a
	// single tick, so it adds no preflight load. Only the LIVE path parks (dry-run inspects).
	ledgerPath := firstNonEmpty(*ledger, defaultLoopLedger())
	if *live && !*noLedger {
		events, _ := loopmgr.Load(ledgerPath)
		parks, lastPark := deriveSweepSeatParkState(events)
		seat := seatpark.Decide(seatpark.Input{
			TaskID:       dispatchSweepLoopID,
			Parks:        parks,
			LastParkUnix: lastPark,
			NowUnix:      time.Now().Unix(),
		})
		if !seat.ShouldAttempt() {
			recordSweepSeatPark(ledgerPath, string(seat.Status), seat.Detail)
			fmt.Fprintf(stdout, "issue-dispatch-sweep: %s — %s\n", seat.Status, seat.Detail)
			return 0
		}
	}

	rec := dispatchsweep.RunSweep(cfg, tick, settle)

	// Record this LIVE sweep's outcome in the park tail: a seat-refuse stop counts toward the
	// bounded budget; any other stop ends the tail so the next cycle starts fresh (#3523).
	if *live && !*noLedger {
		seatReason := ""
		if gardenDispatchSeatRefuses[rec.StopVerdict] {
			seatReason = seatParkReasonNoSeat
		}
		recordSweepSeatPark(ledgerPath, seatReason, fmt.Sprintf("dispatch sweep %s: %s", rec.StopVerdict, rec.StopReason))
	}

	if *asJSON {
		if rc := encodeJSONOrFailPrefixed(stdout, stderr, rec, "fak dispatch sweep: marshal"); rc != 0 {
			return rc
		}
	} else {
		fmt.Fprint(stdout, renderSweepCard(rec))
	}
	if rec.OK {
		return 0
	}
	return 1
}

// lastJSONObject returns the last top-level JSON object in out. The tick prints exactly one
// object under --json, but a loop-ledger append or an incidental log line could precede it, so
// we scan from the end for the last balanced {...} that decodes — that one can never be shadowed.
// dispatchSweepRefresh limits the expensive fleet-registry scan to the first tick in a
// drain. Later admission decisions read live pidfile leases and the roster directly.
func dispatchSweepRefresh(iter int) bool { return iter == 0 }

func lastJSONObject(out []byte) (map[string]any, error) {
	trimmed := bytes.TrimSpace(out)
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("no parseable JSON object in tick output")
	}
	var direct map[string]any
	if err := json.Unmarshal(trimmed, &direct); err == nil {
		return direct, nil
	}

	s := string(out)
	depth, start := 0, -1
	inString, escaped := false, false
	var last map[string]any
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			if depth == 0 {
				continue
			}
			depth--
			if depth == 0 && start >= 0 {
				var m map[string]any
				if err := json.Unmarshal([]byte(s[start:i+1]), &m); err == nil {
					last = m
				}
				start = -1
			}
		}
	}
	if last != nil {
		return last, nil
	}
	return nil, fmt.Errorf("no parseable JSON object in tick output")
}

// tickResultFromJSON projects a decoded tick object onto the loop's TickResult. JSON numbers
// arrive as float64; target_issue (preferred) or issue carries the issue number.
func tickResultFromJSON(m map[string]any) dispatchsweep.TickResult {
	tr := dispatchsweep.TickResult{
		Action:  jsonStr(m, "action"),
		Verdict: jsonStr(m, "verdict"),
		Reason:  jsonStr(m, "reason"),
		Lane:    jsonStr(m, "lane"),
		OK:      jsonBool(m, "ok"),
	}
	if iss := jsonInt(m, "target_issue"); iss != 0 {
		tr.Issue = iss
	} else {
		tr.Issue = jsonInt(m, "issue")
	}
	if acct, ok := m["account"].(map[string]any); ok {
		tr.Account = jsonStr(acct, "tag")
	}
	if pre, ok := m["preflight"].(map[string]any); ok {
		tr.PreflightVerdict = jsonStr(pre, "verdict")
	}
	return tr
}

// jsonStr is maputil.Str: read a decoded-JSON key as a string, "" when absent or not a
// string. One definition of that rule (internal/maputil), not a copy per decoder.
func jsonStr(m map[string]any, k string) string { return maputil.Str(m, k) }

func jsonBool(m map[string]any, k string) bool {
	b, _ := m[k].(bool)
	return b
}

func jsonInt(m map[string]any, k string) int {
	switch v := m[k].(type) {
	case float64:
		return int(v)
	case int:
		return v
	}
	return 0
}

// renderSweepCard is the human read-out: one header line, one line per tick, and the typed stop.
func renderSweepCard(rec dispatchsweep.Record) string {
	var b strings.Builder
	fmt.Fprintf(&b, "issue-dispatch-sweep: %s  spawned=%d/%d  cap(max-workers)=%d\n",
		rec.Mode, rec.SpawnedCount, rec.MaxAgents, rec.MaxWorkers)
	for _, t := range rec.Ticks {
		mark := "."
		if t.Action == "spawned" || t.Action == "would_spawn" {
			mark = "+"
		}
		issue := "--"
		if t.Issue != 0 {
			issue = fmt.Sprintf("#%d", t.Issue)
		}
		lane := t.Lane
		if lane == "" {
			lane = "-"
		}
		acct := t.Account
		if acct == "" {
			acct = "-"
		}
		fmt.Fprintf(&b, "  %s tick %d: %s  %s lane=%s acct=%s\n",
			mark, t.Iteration, t.Verdict, issue, lane, acct)
	}
	fmt.Fprintf(&b, "  STOP: %s — %s\n", rec.StopVerdict, rec.StopReason)
	if rec.Mode == "dry-run" {
		fmt.Fprintln(&b, "  (dry-run — re-run with --live to drain the queue)")
	}
	return b.String()
}
