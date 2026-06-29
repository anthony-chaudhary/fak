// Package dispatchsweep is the queue-drain loop core: find next issue -> spawn one worker ->
// repeat, until a tick refuses or the best-effort agent ceiling is hit.
//
// # Why this exists
//
// tools/issue_resolve_dispatch.py is a SINGLE guarded dispatch tick: it preflights the DoS
// gate (host clean AND a Claude seat free AND live workers < cap), picks the busiest lane's
// next open issue, and launches exactly ONE detached worker. The always-on FleetIssueDispatch
// task fires one such tick every few minutes. Nothing turns "do that repeatedly, right now,
// until the queue or the capacity runs out" into a single operator action — so a human who
// wants to fill the fleet has to hand-fire ticks. This is that missing keystone: the
// `fak dispatch sweep` shell drives RunSweep, which loops the Go `fak dispatch tick`.
//
// # Safety is delegated, not re-implemented
//
// RunSweep adds ONLY the loop. Every spawn still passes through the same Go tick — the
// same dispatch_preflight gate, the same switcher account pin, the same in-flight #N de-dup,
// the same loop-ledger append. The sweep STOPS the instant a tick refuses: REFUSE_AT_CAP (the
// fleet is full), NO_LANE/NO_ISSUE (the queue is drained), WEEKLY_CAPPED/REFUSE_NO_ACCOUNT (no
// seat), or BACKEND_UNHEALTHY. A refusal is the natural, witnessed terminator — the loop never
// needs to know the cap, so it can never exceed it.
//
// # The "up to N, best effort" contract
//
// MaxAgents (the --max-agents ceiling) is the best-effort UPPER bound, NOT a promise: the
// preflight cap (--max-workers) is the real limiter, so on a box with a cap of 2 the sweep
// spawns at most 2 and the third tick refuses REFUSE_AT_CAP and it stops cleanly. Ask for many,
// get exactly as many as the DoS gate allows, and a typed reason for every one it did not spawn.
//
// The loop is pure: tick and settle are injected, so it is unit-tested with no subprocess and
// no sleep. The production tick (cmd/fak) runs the Go dispatch tick evaluator directly.
package dispatchsweep

import "fmt"

// Schema tags the machine-readable sweep record.
const Schema = "fleet-issue-dispatch-sweep/1"

// TickResult is the slice of a single dispatch tick (tools/issue_resolve_dispatch.py --json)
// the sweep loop reasons over. The cmd/fak shell fills it from the tick's JSON; tests supply
// it directly.
type TickResult struct {
	Action           string // "spawned" | "would_spawn" | "refused" | "no_lane" | "no_issue" | ...
	Verdict          string // "SPAWNED" | "WOULD_SPAWN" | "REFUSE_AT_CAP" | "NO_LANE" | ...
	Reason           string
	Lane             string
	Account          string
	PreflightVerdict string
	Issue            int // target issue number, 0 if none
	OK               bool
}

// progressActions: a tick whose Action is one of these ADVANCED the sweep (a worker spawned,
// or in dry-run one would have). Anything else is a terminator that stops the loop.
var progressActions = map[string]bool{"spawned": true, "would_spawn": true}

// stopReasons maps a terminator verdict onto a one-line operator-facing stop reason. An unknown
// verdict falls through to "tick refused: <verdict>" so a new tick verdict is never silently
// swallowed as success.
var stopReasons = map[string]string{
	"REFUSE_AT_CAP":     "fleet is at the worker cap (raise --max-workers to grow)",
	"REFUSE_HOST_DIRTY": "host guard is dirty (a runaway/leak gate is tripped)",
	"REFUSE_NO_ACCOUNT": "no free Claude seat (all accounts busy or throttled)",
	"WEEKLY_CAPPED":     "account is weekly-quota-capped",
	"BACKEND_UNHEALTHY": "worker backend is spinning dead (banner-only/0-byte logs)",
	"NO_LANE":           "queue drained: no lane has an open issue",
	"NO_ISSUE":          "queue drained: every open issue already has a live worker",
	"SPAWN_FAILED":      "a worker failed to spawn (see the tick log)",
}

// faultVerdicts are the terminators that mean the TOOLING broke, not that a boundary was hit —
// they make the sweep record NOT ok. Everything else (a refusal, a drained queue, the dry-run
// plan, the agent ceiling) is an expected boundary and stays ok.
var faultVerdicts = map[string]bool{
	"TICK_ERROR":   true,
	"MAX_ITERS":    true,
	"SPAWN_FAILED": true,
	"UNKNOWN":      true,
}

// TickRecord is the per-iteration slice the sweep keeps for its record.
type TickRecord struct {
	Iteration        int    `json:"iteration"`
	Action           string `json:"action"`
	Verdict          string `json:"verdict"`
	OK               bool   `json:"ok"`
	Issue            int    `json:"issue,omitempty"`
	Lane             string `json:"lane,omitempty"`
	Account          string `json:"account,omitempty"`
	PreflightVerdict string `json:"preflight,omitempty"`
	Reason           string `json:"reason,omitempty"`
}

// Record is the machine-readable result of one sweep.
type Record struct {
	Schema        string       `json:"schema"`
	Mode          string       `json:"mode"`
	MaxAgents     int          `json:"max_agents"`
	MaxWorkers    int          `json:"max_workers"`
	Backend       string       `json:"backend"`
	Lane          string       `json:"lane,omitempty"`
	SpawnedCount  int          `json:"spawned_count"`
	SpawnedIssues []int        `json:"spawned_issues"`
	StopVerdict   string       `json:"stop_verdict"`
	StopReason    string       `json:"stop_reason"`
	Ticks         []TickRecord `json:"ticks"`
	OK            bool         `json:"ok"`
}

// Config is the sweep's bounds. Only MaxAgents and Live drive the loop; MaxWorkers/Backend/Lane
// are carried into the record for provenance (the cmd/fak shell threads them into the tick).
type Config struct {
	MaxAgents  int
	MaxWorkers int
	Backend    string
	Lane       string
	Live       bool
}

// TickFunc runs one dispatch tick (iter is 0-based) and returns its parsed result. An error
// means the tick could not be run or parsed — a tooling fault that stops the sweep NOT-ok.
type TickFunc func(iter int) (TickResult, error)

// RunSweep drives find->spawn->repeat. tick runs one dispatch tick; settle is called after each
// successful LIVE spawn (so the spawned worker's de-dup log lands before the next tick runs its
// in-flight check). Both are injected so the loop is tested with no subprocess and no sleep.
//
// It returns a fully-populated Record. The loop terminates on the FIRST of: a refusing tick,
// the dry-run single-plan rule, the MaxAgents ceiling, a tick error, or a hard iteration
// backstop (MaxAgents+2) that no healthy run can reach.
func RunSweep(cfg Config, tick TickFunc, settle func()) Record {
	maxIters := cfg.MaxAgents + 2 // backstop: at most MaxAgents progress ticks + a terminator
	ticks := make([]TickRecord, 0, maxIters)
	spawned := []int{}
	var stopVerdict, stopReason string

	for i := 0; i < maxIters; i++ {
		tr, err := tick(i)
		if err != nil {
			stopVerdict = "TICK_ERROR"
			stopReason = "tick failed to run: " + err.Error()
			ticks = append(ticks, TickRecord{Iteration: i + 1, Action: "tick_error",
				Verdict: "TICK_ERROR", Reason: err.Error()})
			break
		}
		ticks = append(ticks, TickRecord{
			Iteration: i + 1, Action: tr.Action, Verdict: tr.Verdict, OK: tr.OK,
			Issue: tr.Issue, Lane: tr.Lane, Account: tr.Account,
			PreflightVerdict: tr.PreflightVerdict, Reason: tr.Reason,
		})

		if !progressActions[tr.Action] {
			stopVerdict = firstNonEmpty(tr.Verdict, tr.Action, "UNKNOWN")
			if r, ok := stopReasons[stopVerdict]; ok {
				stopReason = r
			} else {
				stopReason = "tick refused: " + stopVerdict
			}
			break
		}

		spawned = append(spawned, tr.Issue)

		if !cfg.Live {
			// A dry-run tick changes no preflight/in-flight state, so iterating would re-plan
			// the identical issue forever. Report the single plan honestly and stop.
			stopVerdict = "DRY_RUN"
			stopReason = "dry-run: planned the next tick only; re-run with --live to drain the queue"
			break
		}
		if len(spawned) >= cfg.MaxAgents {
			stopVerdict = "MAX_AGENTS"
			stopReason = fmt.Sprintf("reached the --max-agents best-effort ceiling (%d)", cfg.MaxAgents)
			break
		}
		settle()
	}
	if stopVerdict == "" {
		stopVerdict = "MAX_ITERS"
		stopReason = fmt.Sprintf("hit the %d-iteration backstop without a terminator", maxIters)
	}

	mode := "dry-run"
	if cfg.Live {
		mode = "live"
	}
	return Record{
		Schema:        Schema,
		Mode:          mode,
		MaxAgents:     cfg.MaxAgents,
		MaxWorkers:    cfg.MaxWorkers,
		Backend:       cfg.Backend,
		Lane:          cfg.Lane,
		SpawnedCount:  len(spawned),
		SpawnedIssues: spawned,
		StopVerdict:   stopVerdict,
		StopReason:    stopReason,
		Ticks:         ticks,
		OK:            !faultVerdicts[stopVerdict],
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
