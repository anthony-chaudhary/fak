package main

// cachevalue_census.go — `fak cachevalue census` (#3650), the operator surface for the fleet
// managed-cache posture-adoption census under epic #3569 (trust-but-verify LOOPS for managed
// cache).
//
//	fak cachevalue census          # render the census: fleet %ACTIVE, and %upgrade-fired among ACTIVE
//	fak cachevalue census --json   # the same fold as JSON, for a periodic poster / dashboard
//
// The FOLD is pure and lives in internal/cachevaluereport (FoldCensus); this file is the only
// part that touches the live fleet. It reads the guard-session index every `fak guard` launch
// appends to (internal/guardsessions: handle, pid, and the published loopback gateway URL +
// read-scoped bearer), keeps the LIVE rows, GETs each one's <gateway_url>/debug/vars with that
// bearer — the same cross-process read `fak session status` performs — and reads the
// managed_cache posture block the gateway publishes (guardvars.ManagedCacheVars).
//
// Two honesty rules decide what a row contributes, both enforced by the fold:
//
//   - Ended and stale sessions are NOT fleet workers, so they are dropped before the fold
//     rather than folded in as UNKNOWN. The census answers "what does the fleet running right
//     NOW look like"; a recorded-but-dead session would otherwise dilute every ratio forever.
//   - A live worker that cannot be read (no published gateway, refused bearer, wedged or
//     unparseable answer) lands in UNKNOWN, never in PASSIVE — a scrape failure must not
//     manufacture a low adoption number. An absent managed_cache block on a worker that DID
//     answer is an affirmative PASSIVE witness, per that block's producer contract.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// censusVarsFetch is the per-worker /debug/vars read. It is the same bounded, bearer-
// authenticated cross-process GET `fak session status` uses; a var so tests can drive a
// synthetic fleet without binding gateways.
var censusVarsFetch = fetchSessionIndexStatus

// censusDebugVars is the narrow projection of a worker's /debug/vars document the census
// reads: the managed_cache posture block and nothing else. A nil block is the producer's
// "lever off and nothing observed" shape, which the fold reads as an affirmative PASSIVE.
type censusDebugVars struct {
	ManagedCache *guardvars.ManagedCacheVars `json:"managed_cache"`
}

// runCachevalueCensus handles `fak cachevalue census` — scrape the live fleet's posture and
// fold it into the adoption census. Exit 0 even when the fleet is empty or entirely dark: an
// INSUFFICIENT census is a valid, honest answer, not a failure.
func runCachevalueCensus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak cachevalue census", flag.ContinueOnError)
	fs.SetOutput(stderr)
	regDirFlag := fs.String("reg-dir", "", "registry dir holding guard_sessions.jsonl (default: $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	asJSON := fs.Bool("json", false, "emit the folded census as JSON instead of the rendered block")
	if !parseFlags(fs, argv) {
		return 2
	}
	regDir := resolveSweepRegDir(*regDirFlag)
	alive, aliveOK := sessionIndexAlivePIDs()
	rows := censusWorkerRows(guardsessions.Load(regDir), alive, aliveOK)
	report := cachevaluereport.FoldCensus(rows, time.Now())
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak cachevalue census")
	}
	fmt.Fprint(stdout, cachevaluereport.RenderCensus(report))
	if report.Workers == 0 {
		fmt.Fprintf(stdout, "\n  (no LIVE guard sessions in %s — start one with `fak guard -- <agent>`)\n",
			guardsessions.IndexPath(regDir))
	}
	return 0
}

// censusWorkerRows scrapes every LIVE session in the index and folds each into one census
// row. Sequential by design: the reads are loopback GETs against workers on this machine,
// bounded by the shared session HTTP client's timeout, and a census is an occasional
// operator/cadence read rather than a hot path.
func censusWorkerRows(rows []guardsessions.Row, alive map[int]bool, aliveOK bool) []cachevaluereport.WorkerRow {
	out := make([]cachevaluereport.WorkerRow, 0, len(rows))
	for _, r := range rows {
		if sessionIndexState(r, alive, aliveOK) != "live" {
			continue // ended or crashed: not a live fleet worker, so it carries no posture
		}
		out = append(out, censusWorkerRow(r))
	}
	return out
}

// censusWorkerRow reads ONE live worker's posture. Every failure mode along the way — no
// published gateway, a refused or wedged read, a body that is not the gateway's own JSON
// document — collapses to the same honest UNKNOWN row, which the fold excludes from both
// ratios instead of counting as PASSIVE.
func censusWorkerRow(r guardsessions.Row) cachevaluereport.WorkerRow {
	label := censusWorkerLabel(r)
	if strings.TrimSpace(r.GatewayURL) == "" {
		return cachevaluereport.UnreachedRow(label)
	}
	body, err := censusVarsFetch(r)
	if err != nil {
		return cachevaluereport.UnreachedRow(label)
	}
	var doc censusDebugVars
	if err := json.Unmarshal(body, &doc); err != nil {
		return cachevaluereport.UnreachedRow(label)
	}
	return cachevaluereport.RowFromVars(label, doc.ManagedCache)
}

// censusWorkerLabel names a worker in the census breakdown: its short guard-session handle,
// falling back to the trace id (then the fold's own "unknown") so every row stays
// addressable back to a session an operator can query with `fak session status`.
func censusWorkerLabel(r guardsessions.Row) string {
	if h := strings.TrimSpace(r.Handle); h != "" {
		return h
	}
	return strings.TrimSpace(r.TraceID)
}
