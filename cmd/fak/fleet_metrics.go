package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/providercost"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// fleet_metrics.go — `fak fleet metrics`, the FLEET-FIRST Prometheus surface.
//
// The stack already charts three things: a python engine's aggregate fleet_* scores
// (tools/fleet_bottleneck.py), ONE process's live gateway counters (`fak serve`
// /metrics), and a ledger-folded cache-value P&L (`fak cachevalue metrics`). None of
// them answers the operator's first question — "WHICH sessions are alive right now,
// and what is each one doing?" — because none carries a per-session dimension. The
// fleet_* families are fleet-wide scalars plus <=6-entry leaderboards; the gateway
// families describe the one process that happens to be scraped. So an operator could
// see that the fleet was slow and never see WHICH session was the reason.
//
// This exporter adds the missing axis: a bounded per-`session` label on top of the
// same rollups, so one Grafana dashboard starts at the fleet total and DRILLS DOWN to
// a named session without changing data source.
//
//	fak fleet metrics                                    # render the exposition once to stdout
//	fak fleet metrics --serve --addr 127.0.0.1:9098      # serve /metrics; re-fold per scrape
//	fak fleet metrics --textfile fleet.prom              # write a .prom for a textfile collector
//	fak fleet metrics --fleet                            # ALSO fold peer nodes' C2 session refs
//	fak fleet metrics --since 2026-08-01                 # bound the historical usage fold
//
// TWO PROVENANCE TIERS, NEVER BLENDED. The families are split by WHO authored the
// number, the same fence internal/claimcheck draws and info_fleet.go renders:
//
//   - fak_fleet_session_* / fak_fleet_sessions* — LIVE, read from the durable session
//     registry by an ORACLE (PCB state, heartbeat freshness, resume's idle-vs-TTL
//     posture), never from an agent's self-report. This is the "who is alive" tier.
//   - fak_fleet_usage_* — HISTORICAL, folded from the append-only gateway-usage ledger,
//     whose token/cache numbers are OBSERVED (provider-relayed) except the
//     kv_prefix_* pair, which is fak-authored. This is the "what did it cost" tier.
//
// Mixing them in one family would let a provider-relayed token count inherit the live
// oracle's authority, so they stay in separate names and the HELP line on each names
// its tier.
//
// CARDINALITY. Per-session labels are the reason this exporter exists and also its one
// real risk, so the session dimension is BOUNDED at --max-sessions (default 200) on
// both tiers, and the drop is REPORTED (fak_fleet_sessions_truncated,
// fak_fleet_usage_sessions_truncated) rather than silent — a truncated panel that reads
// as complete is worse than no panel.
//
//fak:ctxplan verb="fleet metrics" enters="nothing live — an offline fold over the durable session registry on disk plus the append-only gateway-usage JSONL ledger" pages="nothing into a model window — it renders a Prometheus text exposition to stdout, a .prom textfile, or a /metrics HTTP endpoint for Grafana to scrape" warms="nothing — it REPORTS which sessions are alive and what they spent; it warms no prompt cache or KV itself"
func runFleetMetrics(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak fleet metrics", flag.ContinueOnError)
	fs.SetOutput(stderr)
	registry := fs.String("registry", "", "durable session registry path (default: the same path `fak session ls --durable` reads)")
	fleet := fs.Bool("fleet", false, "also fold peer nodes' C2 session refs (each fold runs a BOUNDED git fetch — raise the scrape interval)")
	remote := fs.String("remote", "origin", "with --fleet, the git remote whose session refs are folded in")
	stale := fs.Duration("stale", defaultSessionStaleWindow, "heartbeat window past which a running-family session reads STALLED")
	usageLedger := fs.String("usage-ledger", gatewayusageledger.DefaultLedgerRel, "gateway-usage ledger folded into the historical fak_fleet_usage_* families")
	registrationLedger := fs.String("registration-ledger", sessionregistry.DefaultPath(), "durable run-registration ledger folded into the fak_fleet_run_* history families")
	cacheValueLedger := fs.String("cache-value-ledger", "", "fak-cache-value-ledger/1 JSONL with optional session_id (empty disables root cache-value metrics)")
	providerCostLedger := fs.String("provider-cost-ledger", "", "authoritative fak-provider-cost-ledger/1 JSONL (empty disables cost metrics)")
	since := fs.String("since", "", "fold only usage rows on or after this date (YYYY-MM-DD)")
	maxSessions := fs.Int("max-sessions", defaultFleetMetricsMaxSessions, "cardinality bound on the per-session label (0 disables the per-session tier entirely)")
	goalCoverageThreshold := fs.Float64("goal-coverage-threshold", 1, "minimum exact root-goal usage attribution ratio required for efficiency-ready=1 (0..1)")
	serve := fs.Bool("serve", false, "serve the exposition on --addr /metrics, re-folding on each scrape")
	addr := fs.String("addr", "127.0.0.1:9098", "with --serve, the host:port to bind /metrics on")
	textfile := fs.String("textfile", "", "write the exposition atomically to this .prom path (for a node_exporter textfile collector) instead of stdout")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "fak fleet metrics: --since must be YYYY-MM-DD: %v\n", err)
			return 2
		}
	}
	if *goalCoverageThreshold < 0 || *goalCoverageThreshold > 1 {
		fmt.Fprintln(stderr, "fak fleet metrics: --goal-coverage-threshold must be between 0 and 1")
		return 2
	}

	src := fleetMetricsSources{
		registryPath:          *registry,
		fleet:                 *fleet,
		remote:                *remote,
		staleWindow:           *stale,
		usageLedger:           *usageLedger,
		providerCostLedger:    *providerCostLedger,
		cacheValueLedger:      *cacheValueLedger,
		dispatchRunsDir:       filepath.Join(repoRoot(), ".dispatch-runs"),
		registrationLedger:    *registrationLedger,
		since:                 *since,
		maxSessions:           *maxSessions,
		goalCoverageThreshold: *goalCoverageThreshold,
		stderr:                stderr,
	}

	if *serve {
		return serveFleetMetrics(stdout, stderr, src, *addr)
	}

	exposition := src.render(time.Now())
	if *textfile != "" {
		if err := writeFileAtomicProm(*textfile, exposition); err != nil {
			fmt.Fprintf(stderr, "fak fleet metrics: write --textfile %s: %v\n", *textfile, err)
			return 1
		}
		fmt.Fprintf(stderr, "fak fleet metrics: wrote %d bytes to %s\n", len(exposition), *textfile)
		return 0
	}
	fmt.Fprint(stdout, exposition)
	return 0
}

// defaultFleetMetricsMaxSessions bounds the per-session label. 200 is deliberately far
// above any fleet size this repo's tooling plans for (`fak fleet capacity` reasons about
// account seats in the tens) so the cap never truncates a REAL fleet, while still
// refusing to hand Prometheus an unbounded series count if a registry is left uncollected
// and accumulates thousands of dead rows.
const defaultFleetMetricsMaxSessions = 200

// fleetMetricsSources is the resolved input set. Held as a value so the one-shot path and
// the serve handler fold from the SAME recipe — what Grafana scrapes is the projection of
// exactly what `fak session ls --durable` would print for the same instant.
type fleetMetricsSources struct {
	registryPath          string
	fleet                 bool
	remote                string
	staleWindow           time.Duration
	usageLedger           string
	providerCostLedger    string
	cacheValueLedger      string
	dispatchRunsDir       string
	registrationLedger    string
	since                 string
	maxSessions           int
	goalCoverageThreshold float64
	stderr                io.Writer
}

// render re-reads the registry and the ledger from disk and returns the exposition text.
// Re-reading per call is what makes --serve reflect a session starting or dying between
// scrapes; the fold itself is pure given the rows and the injected clock.
func (s fleetMetricsSources) render(now time.Time) string {
	w := newPromWriter()
	w.gauge("fak_fleet_up", "1 when the fleet exporter folded a snapshot (it always does; a missing series means the exporter is down).", 1)
	w.gauge("fak_fleet_snapshot_unixtime", "Unix time of the fold this exposition describes.", float64(now.Unix()))

	inv, byID, readable := s.liveInventory(now)
	renderFleetLiveExposition(w, inv, byID, s.maxSessions)
	writeFleetCommitThroughputMetrics(w, fleetmetrics.MeasureCommitThroughput(repoRoot(), now), inv.Count)

	// Dedupe BEFORE folding: a retried exit flush (or a periodic and an exit flush of one
	// snapshot landing in the same millisecond) writes the same row key twice, and a
	// double-counted session would inflate every historical panel by a silent amount.
	usageRows, dupDropped := gatewayusageledger.DedupeByKey(filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(s.usagePath()), s.since))
	usageFold := foldFleetUsage(usageRows)
	renderFleetUsageExposition(w, usageFold, s.maxSessions, dupDropped)

	registrations, registrationReadable := s.registrationInventory()
	renderFleetRunExposition(w, registrations)
	var costReport providercost.Report
	if s.providerCostLedger != "" {
		if rows, err := providercost.Read(s.providerCostLedger); err == nil {
			costReport = providercost.Fold(rows, registrations)
		}
	}
	cacheRows := []cachevalueledger.Row{}
	if s.cacheValueLedger != "" {
		cacheRows = cachevalueledger.ReadLedgerFile(s.cacheValueLedger)
	}
	renderFleetGoalExposition(w, registrations, usageFold, costReport, cacheRows, s.goalCoverageThreshold)
	writeRepoPulseMetrics(w, s.dispatchRunsDir)
	w.gauge("fak_fleet_registration_registry_readable", "1 when the child-registration lineage ledger was read successfully; 0 means goal-level attribution is unavailable, not that the fleet has no goals.", boolGauge(registrationReadable))

	w.gauge("fak_fleet_registry_readable", "1 when the durable session registry was read successfully; 0 when it could not be read (every live family then reads an honest zero, which is NOT the same as an empty fleet).", boolGauge(readable))
	return w.String()
}

// registrationInventory reads the execution lineage graph written before every guard /
// dispatchworker child starts. Missing is an honest empty graph; malformed or unreadable
// data flips the readability gauge so absence is never mistaken for "no goals".
func (s fleetMetricsSources) registrationInventory() ([]sessionregistry.Record, bool) {
	path := strings.TrimSpace(s.registrationLedger)
	if path == "" {
		return nil, true
	}
	rows, err := (sessionregistry.Store{Path: pathutil.ExpandTilde(path)}).ReadAll()
	if err != nil {
		s.note("read child-registration ledger %s: %v (goal families fold to zero)", path, err)
		return nil, false
	}
	return rows, true
}

// renderFleetRunExposition projects one row per durable ROOT registration. It is
// intentionally separate from renderFleetLiveExposition: a terminal registration is
// historical evidence that a run happened, while a descriptor is the oracle for what
// is alive now. Joining those stores into one family would make a completed run look
// live, or make an exited process disappear from history.
func renderFleetRunExposition(w *promWriter, rows []sessionregistry.Record) {
	roots := make([]sessionregistry.Record, 0, len(rows))
	byState := map[sessionregistry.State]int{}
	for _, r := range rows {
		if strings.TrimSpace(r.RootRegistrationID) == "" || r.RegistrationID != r.RootRegistrationID {
			continue
		}
		roots = append(roots, r)
		byState[r.State]++
	}
	sort.Slice(roots, func(i, j int) bool { return roots[i].RegistrationID < roots[j].RegistrationID })
	w.gauge("fak_fleet_registered_runs", "HISTORICAL REGISTRATION: durable root-run registrations in the lifecycle ledger, including open and terminal states; this is not live process inventory.", float64(len(roots)))
	states := []sessionregistry.State{sessionregistry.StateRegistered, sessionregistry.StateActive, sessionregistry.StateCompleted, sessionregistry.StateFailed, sessionregistry.StateCancelled, sessionregistry.StateLost, sessionregistry.StateReaped, sessionregistry.StateUnknown}
	for _, state := range states {
		w.gauge("fak_fleet_registered_runs_by_state", "HISTORICAL REGISTRATION: durable root-run registrations by latest lifecycle state; this is not live process liveness.", float64(byState[state]), "state", string(state))
	}
	for _, r := range roots {
		labels := []string{"run", r.RegistrationID, "session", strings.TrimSpace(r.Identity.SessionID)}
		w.gauge("fak_fleet_run_info", "HISTORICAL REGISTRATION: one series per durable root run with identity, terminal reason, and witness reference for review and drill-down; value is always 1 and does not assert process liveness.", 1,
			"run", r.RegistrationID,
			"session", strings.TrimSpace(r.Identity.SessionID),
			"root_issue", strings.TrimSpace(r.RootIssue),
			"task", strings.TrimSpace(r.TaskID),
			"goal_id", strings.TrimSpace(r.GoalID),
			"launch", strings.TrimSpace(r.LaunchKind),
			"runtime", strings.TrimSpace(r.Identity.Runtime),
			"state", string(r.State),
			"outcome", strings.TrimSpace(r.RootOutcome),
			"reason", strings.TrimSpace(r.Reason),
			"witness_ref", strings.TrimSpace(r.WitnessRef),
			"source", "durable_registration")
		w.gauge("fak_fleet_run_created_timestamp_seconds", "HISTORICAL REGISTRATION: Unix timestamp when the durable root run was registered; zero means unavailable.", unixTimestamp(r.CreatedAt), labels...)
		w.gauge("fak_fleet_run_started_timestamp_seconds", "HISTORICAL REGISTRATION: Unix timestamp when the durable root run started; zero means not recorded or not started.", unixTimestamp(r.StartedAt), labels...)
		w.gauge("fak_fleet_run_terminal_timestamp_seconds", "HISTORICAL REGISTRATION: Unix timestamp when the durable root run reached a terminal state; zero means open or unavailable.", unixTimestamp(r.TerminalAt), labels...)
		w.gauge("fak_fleet_run_duration_seconds", "HISTORICAL REGISTRATION: elapsed seconds from run start (or registration when start is unavailable) to terminal time; zero means open or unavailable.", fleetRunDurationSeconds(r), labels...)
	}
}
func unixTimestamp(at time.Time) float64 {
	if at.IsZero() {
		return 0
	}
	return float64(at.Unix())
}

func fleetRunDurationSeconds(r sessionregistry.Record) float64 {
	if r.TerminalAt.IsZero() {
		return 0
	}
	start := r.StartedAt
	if start.IsZero() {
		start = r.CreatedAt
	}
	if start.IsZero() || r.TerminalAt.Before(start) {
		return 0
	}
	return r.TerminalAt.Sub(start).Seconds()
}

type fleetGoalAgg struct {
	rootID, rootIssue, taskID string
	rootState                 sessionregistry.State
	rootOutcome               string
	attempts                  map[string]struct{}
	resumes                   int
	witnessed                 int
	wallSeconds               float64
	activeSeconds             float64
	states                    map[sessionregistry.State]int
	sessions                  map[string]struct{}
	usage                     fleetUsageAgg
}

// renderFleetGoalExposition joins durable lineage to usage. A session is attributable
// only when exactly one root claims it; unknown and cross-root claims stay visible.
func renderFleetGoalExposition(w *promWriter, rows []sessionregistry.Record, usage fleetUsageFold, cost providercost.Report, cacheRows []cachevalueledger.Row, coverageThreshold float64) {
	goals := map[string]*fleetGoalAgg{}
	sessionRoots := map[string]string{}
	ambiguous := map[string]bool{}
	for _, r := range rows {
		root := strings.TrimSpace(r.RootRegistrationID)
		if root == "" {
			continue
		}
		g := goals[root]
		if g == nil {
			g = &fleetGoalAgg{rootID: root, attempts: map[string]struct{}{}, states: map[sessionregistry.State]int{}, sessions: map[string]struct{}{}}
			goals[root] = g
		}
		if g.rootIssue == "" {
			g.rootIssue = strings.TrimSpace(r.RootIssue)
		}
		if g.taskID == "" {
			g.taskID = strings.TrimSpace(r.TaskID)
		}
		if r.RegistrationID == root {
			g.rootState = r.State
			g.rootOutcome = strings.TrimSpace(r.RootOutcome)
		}
		g.states[r.State]++
		if r.AttemptID != "" {
			g.attempts[r.AttemptID] = struct{}{}
		}
		if r.ResumeOfAttemptID != "" {
			g.resumes++
		}
		if strings.TrimSpace(r.WitnessRef) != "" {
			g.witnessed++
		}
		end := r.TerminalAt
		if end.IsZero() {
			end = r.HeartbeatAt
		}
		if end.IsZero() {
			end = r.StartedAt
		}
		if !r.CreatedAt.IsZero() && end.After(r.CreatedAt) {
			g.wallSeconds += end.Sub(r.CreatedAt).Seconds()
		}
		if !r.StartedAt.IsZero() && end.After(r.StartedAt) {
			g.activeSeconds += end.Sub(r.StartedAt).Seconds()
		}
		if sid := strings.TrimSpace(r.Identity.SessionID); sid != "" {
			g.sessions[sid] = struct{}{}
			if prior, ok := sessionRoots[sid]; ok && prior != root {
				ambiguous[sid] = true
			} else {
				sessionRoots[sid] = root
			}
		}
	}
	for sid, agg := range usage.BySession {
		if !ambiguous[sid] {
			if g := goals[sessionRoots[sid]]; g != nil {
				g.usage.merge(*agg)
			}
		}
	}
	type cacheRoot struct {
		rows           int
		prompt, reused uint64
	}
	cacheRoots := map[string]*cacheRoot{}
	cacheAttributed, cacheMissing, cacheAmbiguous := 0, 0, 0
	for _, row := range cacheRows {
		sid := strings.TrimSpace(row.SessionID)
		if sid == "" {
			cacheMissing++
			continue
		}
		if ambiguous[sid] {
			cacheAmbiguous++
			continue
		}
		root := sessionRoots[sid]
		if root == "" {
			cacheMissing++
			continue
		}
		x := cacheRoots[root]
		if x == nil {
			x = &cacheRoot{}
			cacheRoots[root] = x
		}
		x.rows++
		x.prompt += row.PromptTokens
		x.reused += row.ReusedTokens
		cacheAttributed++
	}
	ids := maputil.SortedKeys(goals)
	states := []sessionregistry.State{sessionregistry.StateRegistered, sessionregistry.StateActive, sessionregistry.StateCompleted, sessionregistry.StateFailed, sessionregistry.StateCancelled, sessionregistry.StateLost, sessionregistry.StateReaped, sessionregistry.StateUnknown}
	for _, id := range ids {
		g := goals[id]
		labels := []string{"root_registration", g.rootID, "root_issue", g.rootIssue, "task", g.taskID}
		w.gauge("fak_fleet_goal_info", "Root-goal identity from the durable child-registration ledger. Value is always 1.", 1, labels...)
		w.gauge("fak_fleet_goal_registrations", "Registrations attributed to this root goal, including the root registration.", float64(sumGoalStates(g.states)), labels...)
		w.gauge("fak_fleet_goal_sessions", "Distinct explicit gateway session IDs registered beneath this root goal.", float64(len(g.sessions)), labels...)
		w.gauge("fak_fleet_goal_attempts_total", "Distinct authoritative attempt IDs registered beneath this root goal.", float64(len(g.attempts)), labels...)
		w.gauge("fak_fleet_goal_resumes_total", "Registrations beneath this root goal carrying an explicit resume_of_attempt_id.", float64(g.resumes), labels...)
		w.gauge("fak_fleet_goal_witnessed_registrations", "Registrations beneath this root goal carrying an explicit witness_ref.", float64(g.witnessed), labels...)
		w.gauge("fak_fleet_goal_wall_seconds", "Summed durable created-to-latest-event elapsed seconds; absent endpoints contribute zero.", g.wallSeconds, labels...)
		w.gauge("fak_fleet_goal_active_seconds", "Summed durable started-to-latest-event active seconds; absent endpoints contribute zero.", g.activeSeconds, labels...)
		for _, state := range states {
			ls := append(append([]string{}, labels...), "state", string(state))
			w.gauge("fak_fleet_goal_registration_state", "Registrations beneath this root goal by latest durable lifecycle state.", float64(g.states[state]), ls...)
		}
		if g.rootState != "" {
			ls := append(append([]string{}, labels...), "state", string(g.rootState))
			w.gauge("fak_fleet_goal_terminal_state", "Latest authoritative lifecycle state of the root registration. No series means the root registration is absent.", 1, ls...)
		}
		if g.rootOutcome != "" {
			ls := append(append([]string{}, labels...), "outcome", g.rootOutcome)
			w.gauge("fak_fleet_goal_outcome_info", "Explicit root outcome recorded by the root registration; never inferred from child state.", 1, ls...)
		}
		w.gauge("fak_fleet_goal_observed_turns_total", "Historical gateway turns explicitly attributed to a registered session beneath this root goal.", float64(g.usage.ObservedTurns), labels...)
		w.gauge("fak_fleet_goal_input_tokens_total", "Historical gateway input tokens explicitly attributed to a registered session beneath this root goal.", float64(g.usage.InputTokens), labels...)
		w.gauge("fak_fleet_goal_output_tokens_total", "Historical gateway output tokens explicitly attributed to a registered session beneath this root goal.", float64(g.usage.OutputTokens), labels...)
		w.gauge("fak_fleet_goal_adjudications_total", "Historical gateway adjudications explicitly attributed to a registered session beneath this root goal.", float64(g.usage.Adjudications), labels...)
		w.gauge("fak_fleet_goal_tool_boundary_calls_total", "Historical kernel submissions explicitly attributed beneath this root goal; the governed tool boundary, not inferred provider calls.", float64(g.usage.Submits), labels...)
		w.gauge("fak_fleet_goal_cache_read_tokens_total", "Observed provider cache-read prompt tokens explicitly attributed beneath this root goal.", float64(g.usage.CachedPromptTokens), labels...)
		w.gauge("fak_fleet_goal_cache_write_tokens_total", "Observed provider cache-creation prompt tokens explicitly attributed beneath this root goal.", float64(g.usage.CacheCreationTokens), labels...)
		if cv := cacheRoots[id]; cv != nil {
			w.gauge("fak_fleet_goal_cache_value_prompt_tokens_total", "Cache-value ledger prompt tokens attributed to exactly one root goal.", float64(cv.prompt), labels...)
			w.gauge("fak_fleet_goal_cache_value_reused_tokens_total", "Cache-value ledger reused tokens attributed to exactly one root goal.", float64(cv.reused), labels...)
			reuseRatio := 0.0
			if cv.prompt > 0 {
				reuseRatio = float64(cv.reused) / float64(cv.prompt)
			}
			w.gauge("fak_fleet_goal_cache_value_reuse_ratio", "Reused cache-value tokens divided by prompt tokens attributed to exactly one root goal.", reuseRatio, labels...)
			w.gauge("fak_fleet_goal_cache_value_rows", "Cache-value rows attributed to exactly one root goal.", float64(cv.rows), labels...)
		}
	}
	attributed := fleetUsageAgg{}
	for _, g := range goals {
		attributed.merge(g.usage)
	}
	unattributed := usage.Total.subtract(attributed)
	w.gauge("fak_fleet_goal_usage_rows", "Gateway-usage rows by root-goal attribution; unattributed includes missing, unknown, and ambiguous lineage.", float64(attributed.Rows), "attribution", "attributed")
	w.gauge("fak_fleet_goal_usage_rows", "Gateway-usage rows by root-goal attribution; unattributed includes missing, unknown, and ambiguous lineage.", float64(unattributed.Rows), "attribution", "unattributed")
	ratio := coverageRatio(attributed.Rows, usage.Total.Rows)
	w.gauge("fak_fleet_goal_usage_attribution_ratio", "Fraction of usage rows attributable to exactly one root goal; 1 for an empty census.", ratio)
	w.gauge("fak_fleet_goal_efficiency_coverage_threshold", "Configured minimum exact usage attribution ratio for broad root-goal efficiency readiness.", coverageThreshold)
	w.gauge("fak_fleet_goal_efficiency_ready", "1 only when exact root-goal usage attribution meets the configured threshold and the lineage registry is readable.", boolGauge(ratio >= coverageThreshold))
	cacheTotal := len(cacheRows)
	w.gauge("fak_fleet_goal_cache_value_rows_total", "Cache-value rows by root-goal attribution outcome.", float64(cacheAttributed), "attribution", "attributed")
	w.gauge("fak_fleet_goal_cache_value_rows_total", "Cache-value rows by root-goal attribution outcome.", float64(cacheMissing), "attribution", "missing")
	w.gauge("fak_fleet_goal_cache_value_rows_total", "Cache-value rows by root-goal attribution outcome.", float64(cacheAmbiguous), "attribution", "ambiguous")
	cacheRatio := coverageRatio(cacheAttributed, cacheTotal)
	w.gauge("fak_fleet_goal_cache_value_attribution_ratio", "Fraction of cache-value rows attributed to exactly one root goal.", cacheRatio)
	w.gauge("fak_fleet_goal_cache_value_efficiency_ready", "1 only when cache-value rows exist and exact attribution meets the configured threshold.", boolGauge(cacheTotal > 0 && cacheRatio >= coverageThreshold))
	for _, root := range cost.Roots {
		g := goals[root.RootRegistrationID]
		if g == nil {
			continue
		}
		labels := []string{"root_registration", g.rootID, "root_issue", g.rootIssue, "task", g.taskID}
		w.gauge("fak_fleet_goal_provider_billed_micro_usd_total", "Authoritative provider-export billed micro-USD attributed to exactly one root goal; absent/unknown amounts are excluded.", float64(root.BilledMicroUSD), labels...)
		w.gauge("fak_fleet_goal_provider_cost_rows", "Provider billing-export rows attributed to exactly one root goal.", float64(root.Rows), labels...)
	}
	w.gauge("fak_fleet_goal_provider_cost_rows_total", "Provider billing-export rows by attribution outcome.", float64(cost.Coverage.TotalRows), "attribution", "all")
	w.gauge("fak_fleet_goal_provider_cost_rows_total", "Provider billing-export rows by attribution outcome.", float64(cost.Coverage.AttributedRows), "attribution", "attributed")
	w.gauge("fak_fleet_goal_provider_cost_rows_total", "Provider billing-export rows by attribution outcome.", float64(cost.Coverage.MissingRows), "attribution", "missing")
	w.gauge("fak_fleet_goal_provider_cost_rows_total", "Provider billing-export rows by attribution outcome.", float64(cost.Coverage.AmbiguousRows), "attribution", "ambiguous")
	costRatio := coverageRatio(cost.Coverage.AttributedAmountRows, cost.Coverage.AmountRows)
	w.gauge("fak_fleet_goal_provider_cost_attribution_ratio", "Fraction of known-amount provider rows attributed to exactly one root goal.", costRatio)
	costReady := cost.Coverage.AmountRows > 0 && costRatio >= coverageThreshold
	w.gauge("fak_fleet_goal_provider_cost_efficiency_ready", "1 only when at least one known provider amount exists and exact cost attribution meets the configured threshold.", boolGauge(costReady))
	w.gauge("fak_fleet_goal_input_tokens_by_attribution_total", "Gateway input tokens by root-goal attribution.", float64(attributed.InputTokens), "attribution", "attributed")
	w.gauge("fak_fleet_goal_input_tokens_by_attribution_total", "Gateway input tokens by root-goal attribution.", float64(unattributed.InputTokens), "attribution", "unattributed")
	w.gauge("fak_fleet_goal_output_tokens_by_attribution_total", "Gateway output tokens by root-goal attribution.", float64(attributed.OutputTokens), "attribution", "attributed")

	w.gauge("fak_fleet_goal_output_tokens_by_attribution_total", "Gateway output tokens by root-goal attribution.", float64(unattributed.OutputTokens), "attribution", "unattributed")
	renderFleetCanonicalGoalExposition(w, rows, goals, cost, cacheRows, coverageThreshold)
}

func renderFleetCanonicalGoalExposition(w *promWriter, rows []sessionregistry.Record, roots map[string]*fleetGoalAgg, cost providercost.Report, cacheRows []cachevalueledger.Row, coverageThreshold float64) {
	type canonicalAgg struct {
		rootIDs                          map[string]struct{}
		attempts                         map[string]struct{}
		sessions                         map[string]struct{}
		resumes, witnessed               int
		wallSeconds, activeSeconds       float64
		usage                            fleetUsageAgg
		completed, failed, other         int
		providerRows, providerAmountRows int
		providerBilledMicroUSD           int64
		cacheRows                        int
		cachePrompt, cacheReused         uint64
	}
	rootGoals := map[string]string{}
	rootConflict := map[string]bool{}
	for _, r := range rows {
		root, goalID := strings.TrimSpace(r.RootRegistrationID), strings.TrimSpace(r.GoalID)
		if root == "" || goalID == "" {
			continue
		}
		if prior := rootGoals[root]; prior != "" && prior != goalID {
			rootConflict[root] = true
		} else {
			rootGoals[root] = goalID
		}
	}
	canonical := map[string]*canonicalAgg{}
	boundRoots, conflictingRoots := 0, 0
	for root, g := range roots {
		if rootConflict[root] {
			conflictingRoots++
			continue
		}
		goalID := rootGoals[root]
		if goalID == "" {
			continue
		}
		boundRoots++
		c := canonical[goalID]
		if c == nil {
			c = &canonicalAgg{rootIDs: map[string]struct{}{}, attempts: map[string]struct{}{}, sessions: map[string]struct{}{}}
			canonical[goalID] = c
		}
		c.rootIDs[root] = struct{}{}
		for attempt := range g.attempts {
			c.attempts[root+"\x00"+attempt] = struct{}{}
		}
		for session := range g.sessions {
			c.sessions[session] = struct{}{}
		}
		c.resumes += g.resumes
		c.witnessed += g.witnessed
		c.wallSeconds += g.wallSeconds
		c.activeSeconds += g.activeSeconds
		c.usage.merge(g.usage)
		switch g.rootState {
		case sessionregistry.StateCompleted:
			c.completed++
		case sessionregistry.StateFailed, sessionregistry.StateCancelled, sessionregistry.StateLost, sessionregistry.StateReaped:
			c.failed++
		default:
			c.other++
		}
	}

	for _, rootCost := range cost.Roots {
		goalID := rootGoals[rootCost.RootRegistrationID]
		if goalID == "" || rootConflict[rootCost.RootRegistrationID] || canonical[goalID] == nil {
			continue
		}
		c := canonical[goalID]
		c.providerRows += rootCost.Rows
		c.providerAmountRows += rootCost.AmountRows
		c.providerBilledMicroUSD += int64(rootCost.BilledMicroUSD)
	}
	sessionRoot, sessionAmbiguous := map[string]string{}, map[string]bool{}
	for _, r := range rows {
		sid, root := strings.TrimSpace(r.Identity.SessionID), strings.TrimSpace(r.RootRegistrationID)
		if sid == "" || root == "" {
			continue
		}
		if prior := sessionRoot[sid]; prior != "" && prior != root {
			sessionAmbiguous[sid] = true
		} else {
			sessionRoot[sid] = root
		}
	}
	for _, row := range cacheRows {
		root := sessionRoot[strings.TrimSpace(row.SessionID)]
		goalID := rootGoals[root]
		if root == "" || sessionAmbiguous[row.SessionID] || goalID == "" || rootConflict[root] || canonical[goalID] == nil {
			continue
		}
		c := canonical[goalID]
		c.cacheRows++
		c.cachePrompt += row.PromptTokens
		c.cacheReused += row.ReusedTokens
	}
	ids := maputil.SortedKeys(canonical)
	for _, id := range ids {
		c, labels := canonical[id], []string{"goal_id", id}
		w.gauge("fak_fleet_canonical_goal_info", "Canonical goal with at least one explicitly bound execution root.", 1, labels...)
		w.gauge("fak_fleet_canonical_goal_execution_roots", "Execution roots explicitly bound to this canonical goal.", float64(len(c.rootIDs)), labels...)
		w.gauge("fak_fleet_canonical_goal_attempts_total", "Distinct root-qualified attempts across bound execution roots.", float64(len(c.attempts)), labels...)
		w.gauge("fak_fleet_canonical_goal_resumes_total", "Resume registrations across bound execution roots.", float64(c.resumes), labels...)
		w.gauge("fak_fleet_canonical_goal_sessions", "Distinct sessions across bound execution roots.", float64(len(c.sessions)), labels...)
		w.gauge("fak_fleet_canonical_goal_witnessed_registrations", "Witnessed registrations across bound execution roots.", float64(c.witnessed), labels...)
		w.gauge("fak_fleet_canonical_goal_wall_seconds", "Summed registration wall seconds across bound execution roots.", c.wallSeconds, labels...)
		w.gauge("fak_fleet_canonical_goal_active_seconds", "Summed registration active seconds across bound execution roots.", c.activeSeconds, labels...)
		w.gauge("fak_fleet_canonical_goal_prompt_tokens_total", "Prompt tokens from sessions attributed to bound execution roots.", float64(c.usage.InputTokens), labels...)
		w.gauge("fak_fleet_canonical_goal_output_tokens_total", "Output tokens from sessions attributed to bound execution roots.", float64(c.usage.OutputTokens), labels...)
		w.gauge("fak_fleet_canonical_goal_cache_read_tokens_total", "Cache-read tokens from sessions attributed to bound execution roots.", float64(c.usage.CachedPromptTokens), labels...)
		w.gauge("fak_fleet_canonical_goal_cache_write_tokens_total", "Cache-write tokens from sessions attributed to bound execution roots.", float64(c.usage.CacheCreationTokens), labels...)
		w.gauge("fak_fleet_canonical_goal_tool_boundary_calls_total", "Tool-boundary calls from sessions attributed to bound execution roots.", float64(c.usage.Submits), labels...)
		w.gauge("fak_fleet_canonical_goal_provider_billed_micro_usd_total", "Authoritative provider-export billed micro-USD folded from explicitly bound execution roots.", float64(c.providerBilledMicroUSD), labels...)
		w.gauge("fak_fleet_canonical_goal_provider_cost_rows", "Provider billing-export rows folded from explicitly bound execution roots.", float64(c.providerRows), labels...)
		w.gauge("fak_fleet_canonical_goal_provider_cost_amount_rows", "Provider billing-export rows with known amounts folded from explicitly bound execution roots.", float64(c.providerAmountRows), labels...)
		w.gauge("fak_fleet_canonical_goal_cache_value_prompt_tokens_total", "Cache-value prompt tokens folded from explicitly bound execution roots.", float64(c.cachePrompt), labels...)
		w.gauge("fak_fleet_canonical_goal_cache_value_reused_tokens_total", "Cache-value reused tokens folded from explicitly bound execution roots.", float64(c.cacheReused), labels...)
		cacheReuseRatio := 0.0
		if c.cachePrompt > 0 {
			cacheReuseRatio = float64(c.cacheReused) / float64(c.cachePrompt)
		}
		w.gauge("fak_fleet_canonical_goal_cache_value_reuse_ratio", "Reused cache-value tokens divided by prompt tokens across explicitly bound execution roots.", cacheReuseRatio, labels...)
		w.gauge("fak_fleet_canonical_goal_cache_value_rows", "Cache-value rows folded from explicitly bound execution roots.", float64(c.cacheRows), labels...)
		w.gauge("fak_fleet_canonical_goal_execution_roots_by_outcome", "Execution roots by terminal outcome class.", float64(c.completed), append(labels, "outcome", "completed")...)
		w.gauge("fak_fleet_canonical_goal_execution_roots_by_outcome", "Execution roots by terminal outcome class.", float64(c.failed), append(labels, "outcome", "failed")...)
		w.gauge("fak_fleet_canonical_goal_execution_roots_by_outcome", "Execution roots by terminal outcome class.", float64(c.other), append(labels, "outcome", "nonterminal_or_unknown")...)
	}
	totalRoots, unboundRoots := len(roots), len(roots)-boundRoots-conflictingRoots
	w.gauge("fak_fleet_canonical_goal_execution_roots_total", "Execution roots by explicit canonical-goal binding outcome.", float64(boundRoots), "attribution", "bound")
	w.gauge("fak_fleet_canonical_goal_execution_roots_total", "Execution roots by explicit canonical-goal binding outcome.", float64(unboundRoots), "attribution", "execution_root_only")
	w.gauge("fak_fleet_canonical_goal_execution_roots_total", "Execution roots by explicit canonical-goal binding outcome.", float64(conflictingRoots), "attribution", "conflicting")
	ratio := coverageRatio(boundRoots, totalRoots)
	w.gauge("fak_fleet_canonical_goal_binding_ratio", "Fraction of execution roots carrying one explicit canonical goal identity.", ratio)
	w.gauge("fak_fleet_canonical_goal_efficiency_ready", "1 only when execution roots exist and explicit canonical-goal binding meets the configured threshold.", boolGauge(totalRoots > 0 && ratio >= coverageThreshold))
}
func sumGoalStates(m map[sessionregistry.State]int) int {
	n := 0
	for _, v := range m {
		n += v
	}
	return n
}
func coverageRatio(a, t int) float64 {
	if t == 0 {
		return 1
	}
	return float64(a) / float64(t)
}

// usagePath resolves the ledger flag the same way every other nightrun reader does, so a
// relative default lands on the repo's ledger rather than the process's cwd.
func (s fleetMetricsSources) usagePath() string {
	p := strings.TrimSpace(s.usageLedger)
	if p == "" {
		return ""
	}
	if strings.HasPrefix(p, ".fak/") || strings.HasPrefix(p, "docs/") {
		return nightrunLedgerPath(p)
	}
	return p
}

// liveInventory reads the durable registry (and, under --fleet, the peer C2 refs) and
// folds them through the SAME buildSessionInventory `fak session ls` uses, so the
// exposition and the CLI can never disagree about whether a session is live. It also
// returns the raw local descriptors indexed by id, because the inventory row deliberately
// carries only the acceptance tuple and the drill-down wants the richer durable fields
// (pid, generation, priority, wall-clock budget) the descriptor already holds.
//
// A registry that cannot be read is NOT a fatal error: it folds to an empty inventory and
// flips fak_fleet_registry_readable to 0, so a dashboard can tell "no sessions" from
// "could not look".
func (s fleetMetricsSources) liveInventory(now time.Time) (sessionInventory, map[string]session.Descriptor, bool) {
	path := strings.TrimSpace(s.registryPath)
	if path == "" {
		path = defaultSessionRegistryPath()
	}
	path = pathutil.ExpandTilde(path)

	reg := session.NewRegistry(session.NewFileStore(path))
	local, err := reg.List(now)
	readable := err == nil
	if err != nil {
		s.note("read durable registry %s: %v (live families fold to zero)", path, err)
		local = nil
	}

	var fleetDescs []leaseref.SessionDescriptor
	if s.fleet {
		remote := strings.TrimSpace(s.remote)
		if remote == "" {
			remote = "origin"
		}
		fleetDescs = fleetSessionDescriptors(s.stderrOrDiscard(), leaseref.NewInDir(""), remote, now)
	}

	byID := make(map[string]session.Descriptor, len(local))
	for _, d := range local {
		id := sessionDescriptorID(d)
		if id != "" {
			byID[id] = d
		}
	}
	return buildSessionInventory(local, fleetDescs, now, s.staleWindow, s.fleet), byID, readable
}

func (s fleetMetricsSources) stderrOrDiscard() io.Writer {
	if s.stderr == nil {
		return io.Discard
	}
	return s.stderr
}

func (s fleetMetricsSources) note(format string, args ...any) {
	writeMetricsNote(s.stderr, "fak fleet metrics: ", format, args...)
}

func writeMetricsNote(stderr io.Writer, prefix, format string, args ...any) {
	if stderr == nil {
		return
	}
	fmt.Fprintf(stderr, prefix+format+"\n", args...)
}

// ---- live tier -----------------------------------------------------------------

// renderFleetLiveExposition projects the live inventory into the fak_fleet_* families:
// the fleet-level rollups first (the numbers a fleet overview leads with), then the
// bounded per-session series a drill-down selects one row of. PURE over its inputs so a
// test can pin the exact text.
func renderFleetLiveExposition(w *promWriter, inv sessionInventory, byID map[string]session.Descriptor, maxSessions int) {
	w.gauge("fak_fleet_sessions", "LIVE: sessions in the durable inventory (the same population `fak session ls --durable` prints).", float64(inv.Count))
	w.gauge("fak_fleet_scope_includes_peers", "1 when this fold included peer nodes' C2 session refs (--fleet); 0 when it is this node's durable registry only.", boolGauge(inv.Fleet))

	// The by_state rollup counts the EFFECTIVE status, so a RUNNING row whose heartbeat
	// lapsed lands in STALLED — the same projection the CLI headline makes. Emitting the
	// raw pcb_state here instead would make a wedged fleet read as a working one.
	for _, st := range sortedKeysByPCBOrder(inv.ByState) {
		w.gauge("fak_fleet_sessions_by_state", "LIVE: sessions per EFFECTIVE PCB status; a running-family session with a lapsed heartbeat rolls up as STALLED, not RUNNING.", float64(inv.ByState[st]), "state", st)
	}

	byLiveness := map[string]int{}
	byPosture := map[string]int{}
	bySource := map[string]int{}
	byHost := map[string]int{}
	for _, r := range inv.Sessions {
		byLiveness[r.LivenessClass]++
		byPosture[r.CachePosture]++
		bySource[r.Source]++
		byHost[fleetHostLabel(r.Host)]++
	}
	// The three liveness classes are emitted even at zero so a panel plots a real 0
	// instead of a gap when, say, nothing is stalled.
	for _, class := range []string{string(taskmgr.LivenessLive), string(taskmgr.LivenessIdle), string(taskmgr.LivenessStalled)} {
		w.gauge("fak_fleet_sessions_by_liveness", "LIVE: sessions per witnessed liveness class (live | idle | stalled), classified off the durable heartbeat.", float64(byLiveness[class]), "liveness", class)
	}
	for _, p := range []string{"warm", "cold", "unknown"} {
		w.gauge("fak_fleet_sessions_by_cache_posture", "LIVE: sessions per projected cache posture (warm | cold | unknown) — resume's idle-vs-TTL projection, NOT a witnessed provider-cache hit.", float64(byPosture[p]), "posture", p)
	}
	for _, src := range fleetSortedKeys(bySource) {
		w.gauge("fak_fleet_sessions_by_source", "LIVE: sessions per inventory source — local (this node's durable C1 registry) or fleet (a peer node's C2 ref).", float64(bySource[src]), "source", src)
	}
	for _, h := range fleetSortedKeys(byHost) {
		w.gauge("fak_fleet_sessions_by_host", "LIVE: sessions per host.", float64(byHost[h]), "host", h)
	}
	w.gauge("fak_fleet_hosts", "LIVE: distinct hosts carrying at least one session in this fold.", float64(len(byHost)))
	w.gauge("fak_fleet_sessions_warm", "LIVE: sessions whose projected cache posture is warm.", float64(inv.Warm))
	w.gauge("fak_fleet_sessions_cold", "LIVE: sessions whose projected cache posture is cold.", float64(inv.Cold))

	if maxSessions <= 0 {
		w.gauge("fak_fleet_sessions_truncated", "LIVE: sessions the --max-sessions cardinality bound kept OUT of the per-session families. Non-zero means a per-session panel is incomplete.", float64(inv.Count))
		return
	}

	// Rank before truncating so the cap keeps the sessions an operator is most likely to
	// be looking for: live first, then stalled (the wedged ones worth finding), then the
	// youngest — never an arbitrary map order.
	rows := append([]sessionInventoryRow(nil), inv.Sessions...)
	sort.SliceStable(rows, func(i, j int) bool {
		pi, pj := fleetSessionRank(rows[i]), fleetSessionRank(rows[j])
		if pi != pj {
			return pi < pj
		}
		if rows[i].AgeSeconds != rows[j].AgeSeconds {
			return rows[i].AgeSeconds < rows[j].AgeSeconds
		}
		return rows[i].ID < rows[j].ID
	})
	truncated := 0
	if len(rows) > maxSessions {
		truncated = len(rows) - maxSessions
		rows = rows[:maxSessions]
	}
	w.gauge("fak_fleet_sessions_truncated", "LIVE: sessions the --max-sessions cardinality bound kept OUT of the per-session families. Non-zero means a per-session panel is incomplete.", float64(truncated))

	// Emit in id order so the exposition is byte-stable across folds with the same input.
	sort.Slice(rows, func(i, j int) bool { return rows[i].ID < rows[j].ID })

	// FAMILY-MAJOR, NOT SESSION-MAJOR. Every sample of one metric family must be
	// contiguous in the text exposition — that is what `promtool check metrics` enforces
	// and what the OpenMetrics parser requires. Looping sessions on the outside would
	// interleave the families (info(a), live(a), info(b), live(b), …), which the lenient
	// scrape parser tolerates but the linter and any strict consumer reject. So the loop
	// nesting is inverted: one pass per family, all sessions inside it.
	for _, r := range rows {
		w.gauge("fak_fleet_session_info", "LIVE: one series per session carrying its descriptive labels; the value is always 1. Join a per-session gauge to this to label a drill-down panel.", 1, fleetInfoLabels(r, byID[r.ID])...)
	}
	for _, fam := range []struct {
		name, help string
		val        func(sessionInventoryRow, session.Descriptor) (float64, bool)
	}{
		{"fak_fleet_session_live", "LIVE: 1 when this session's heartbeat is fresh AND it claims a running-family state; 0 otherwise (idle or stalled).",
			func(r sessionInventoryRow, _ session.Descriptor) (float64, bool) {
				return boolGauge(r.LivenessClass == string(taskmgr.LivenessLive)), true
			}},
		{"fak_fleet_session_stalled", "LIVE: 1 when this session claims a running-family state but its heartbeat has not advanced within the stale window — it claims work and is not progressing.",
			func(r sessionInventoryRow, _ session.Descriptor) (float64, bool) {
				return boolGauge(r.LivenessClass == string(taskmgr.LivenessStalled)), true
			}},
		{"fak_fleet_session_age_seconds", "LIVE: seconds since this session was created (for a peer C2 row, seconds since its ref last updated — that is all the ref carries).",
			func(r sessionInventoryRow, _ session.Descriptor) (float64, bool) { return float64(r.AgeSeconds), true }},
		{"fak_fleet_session_cache_warm", "LIVE: 1 when this session's PROJECTED cache posture is warm (idle-since-last-stamp under the resume TTL). A projection, never a witnessed provider-cache hit.",
			func(r sessionInventoryRow, _ session.Descriptor) (float64, bool) {
				return boolGauge(r.CachePosture == "warm"), true
			}},
		// Both of these are emitted UNCONDITIONALLY, including at zero, because zero is a
		// real measured value here rather than a missing one. Generation 0 means "never
		// re-continued" and priority 0 is the DEFAULT scheduling priority (Pick sorts
		// ascending, so 0 is the most common value in a healthy fleet, not an unset one).
		// Suppressing them would leave an ordinary session's stat panel reading "No data",
		// which an operator reads as "the exporter does not know" — the opposite of the
		// truth. Contrast the wall-clock pair below, which is suppressed precisely because
		// an unbudgeted session has no limit to report.
		{"fak_fleet_session_generation", "LIVE: re-continuation depth — how many budget-reset generations this session has been through. 0 means it is still on its first.",
			func(_ sessionInventoryRow, d session.Descriptor) (float64, bool) {
				return float64(d.Generation), true
			}},
		{"fak_fleet_session_priority", "LIVE: the session's scheduling priority as persisted in the durable registry. Lower is picked first; 0 is the default.",
			func(_ sessionInventoryRow, d session.Descriptor) (float64, bool) {
				return float64(d.Priority), true
			}},
		// The wall-clock budget pair is emitted only when a LIMIT was actually set: an
		// unbudgeted session has no denominator, and a 0-limit series would render as a
		// fully-burned budget on any elapsed/limit panel.
		{"fak_fleet_session_time_limit_seconds", "LIVE: this session's wall-clock budget limit, when one is set.",
			func(_ sessionInventoryRow, d session.Descriptor) (float64, bool) {
				return float64(d.Time.LimitNanos) / 1e9, d.Time.LimitNanos > 0
			}},
		{"fak_fleet_session_time_elapsed_seconds", "LIVE: wall-clock budget consumed so far, as persisted by the durable registry.",
			func(_ sessionInventoryRow, d session.Descriptor) (float64, bool) {
				return float64(d.Time.ElapsedNanos) / 1e9, d.Time.LimitNanos > 0
			}},
	} {
		for _, r := range rows {
			if v, ok := fam.val(r, byID[r.ID]); ok {
				w.gauge(fam.name, fam.help, v, "session", r.ID)
			}
		}
	}
}

// fleetInfoLabels builds the descriptive label set for one session's _info series. The
// labels come from BOTH folds on purpose: the inventory row carries the classified tuple
// (state / liveness / posture, which the CLI also prints), and the raw durable descriptor
// carries the operational detail (pid, and the closed reason token a THROTTLED/STOPPED
// row stopped for) that the acceptance tuple deliberately leaves out.
func fleetInfoLabels(r sessionInventoryRow, d session.Descriptor) []string {
	return []string{
		"session", r.ID,
		"host", fleetHostLabel(r.Host),
		"state", r.PCBState,
		"liveness", r.LivenessClass,
		"cache_posture", r.CachePosture,
		"source", r.Source,
		"parent", fleetDashIfEmpty(r.ParentID),
		"pid", fmt.Sprint(d.PID),
		"reason", fleetDashIfEmpty(d.Reason),
	}
}

// fleetSessionRank orders sessions for the cardinality cap: live first (the fleet's
// working set), stalled next (the wedged ones an operator is hunting), idle last.
func fleetSessionRank(r sessionInventoryRow) int {
	switch r.LivenessClass {
	case string(taskmgr.LivenessLive):
		return 0
	case string(taskmgr.LivenessStalled):
		return 1
	default:
		return 2
	}
}

// ---- historical tier -----------------------------------------------------------

// fleetUsageAgg is one summed slice of the gateway-usage ledger — the whole fleet, one
// session_type, or one session. Only the token/cache/turn/verdict fields a fleet rollup
// charts are summed; the arithmetic is spelled out (rather than reusing the ledger's own
// unexported summer) so every number on a panel is auditable from this one function.
type fleetUsageAgg struct {
	Rows                 int
	Sessions             int
	Seconds              float64
	SessionType          string
	InputTokens          uint64
	OutputTokens         uint64
	CachedPromptTokens   uint64
	CacheCreationTokens  uint64
	KVPrefixPromptTokens uint64
	KVPrefixReusedTokens uint64
	ObservedTurns        uint64
	CachedTurns          uint64
	Adjudications        uint64
	Submits              uint64
	Allowed              uint64
	Denied               uint64
	Transformed          uint64
	Quarantined          uint64
	Deferred             uint64
	Escalated            uint64
	Errored              uint64
}

func (a *fleetUsageAgg) add(r gatewayusageledger.Row) {
	a.Rows++
	c := r.Counters
	// A carryforward row is a CUT's fold witness, not one session: it stands for
	// FoldedRows real sessions whose individual rows were replaced. Counting it as one
	// session would silently shrink every historical session count the moment a cut ran.
	switch {
	case r.Kind == gatewayusageledger.KindCarryforward && r.Carryforward != nil:
		a.Sessions += r.Carryforward.FoldedRows
	case r.Kind == gatewayusageledger.KindCarryforward:
		// A carryforward row with no witness: its counters are real, but how many
		// sessions they came from is unknown, so claim none rather than guess.
	default:
		a.Sessions++
	}
	a.Seconds += r.UptimeSecs
	a.InputTokens += c.InputTokens
	a.OutputTokens += c.OutputTokens
	a.CachedPromptTokens += c.CachedPromptTokens
	a.CacheCreationTokens += c.CacheCreationTokens
	a.KVPrefixPromptTokens += c.KVPrefixPromptTokens
	a.KVPrefixReusedTokens += c.KVPrefixReusedTokens
	a.ObservedTurns += c.ObservedTurns
	a.CachedTurns += c.CachedTurns
	a.Adjudications += c.Total
	if c.Submits > 0 {
		a.Submits += uint64(c.Submits)
	}
	a.Allowed += c.Allowed
	a.Denied += c.Denied
	a.Transformed += c.Transformed
	a.Quarantined += c.Quarantined
	a.Deferred += c.Deferred
	a.Escalated += c.Escalated
	a.Errored += c.Errored
}

// CacheReadRatio is the share of prompt tokens the provider served from its cache:
// cache_read / (input + cache_read + cache_creation). The denominator is every prompt
// token the turn accounted for, so a session that paid to CREATE a large cache entry is
// not credited with a high hit rate for it. Returns -1 (skipped, honest absence) when the
// denominator is zero — a ratio over no tokens is not 0%, it is unmeasured.
func (a *fleetUsageAgg) CacheReadRatio() float64 {
	den := a.InputTokens + a.CachedPromptTokens + a.CacheCreationTokens
	if den == 0 {
		return -1
	}
	return float64(a.CachedPromptTokens) / float64(den)
}

// fleetUsageFold is the whole historical fold: fleet total, the by-session_type split,
// and the per-session index — plus the identification census that says how much of the
// corpus the per-session tier can actually speak for.
func (a *fleetUsageAgg) merge(b fleetUsageAgg) {
	a.Rows += b.Rows
	a.Sessions += b.Sessions
	a.Seconds += b.Seconds
	a.InputTokens += b.InputTokens
	a.OutputTokens += b.OutputTokens
	a.CachedPromptTokens += b.CachedPromptTokens
	a.CacheCreationTokens += b.CacheCreationTokens
	a.KVPrefixPromptTokens += b.KVPrefixPromptTokens
	a.KVPrefixReusedTokens += b.KVPrefixReusedTokens
	a.ObservedTurns += b.ObservedTurns
	a.CachedTurns += b.CachedTurns
	a.Adjudications += b.Adjudications
	a.Submits += b.Submits
	a.Allowed += b.Allowed
	a.Denied += b.Denied
	a.Transformed += b.Transformed
	a.Quarantined += b.Quarantined
	a.Deferred += b.Deferred
	a.Escalated += b.Escalated
	a.Errored += b.Errored
}
func (a fleetUsageAgg) subtract(b fleetUsageAgg) fleetUsageAgg {
	return fleetUsageAgg{Rows: a.Rows - b.Rows, Sessions: a.Sessions - b.Sessions, Seconds: a.Seconds - b.Seconds, InputTokens: a.InputTokens - b.InputTokens, OutputTokens: a.OutputTokens - b.OutputTokens, CachedPromptTokens: a.CachedPromptTokens - b.CachedPromptTokens, CacheCreationTokens: a.CacheCreationTokens - b.CacheCreationTokens, KVPrefixPromptTokens: a.KVPrefixPromptTokens - b.KVPrefixPromptTokens, KVPrefixReusedTokens: a.KVPrefixReusedTokens - b.KVPrefixReusedTokens, ObservedTurns: a.ObservedTurns - b.ObservedTurns, CachedTurns: a.CachedTurns - b.CachedTurns, Adjudications: a.Adjudications - b.Adjudications, Submits: a.Submits - b.Submits, Allowed: a.Allowed - b.Allowed, Denied: a.Denied - b.Denied, Transformed: a.Transformed - b.Transformed, Quarantined: a.Quarantined - b.Quarantined, Deferred: a.Deferred - b.Deferred, Escalated: a.Escalated - b.Escalated, Errored: a.Errored - b.Errored}
}

type fleetUsageFold struct {
	Rows            int
	Total           fleetUsageAgg
	ByType          map[string]*fleetUsageAgg
	BySession       map[string]*fleetUsageAgg
	Identified      int
	Unidentified    int
	FirstUnixMillis int64
	LastUnixMillis  int64
}

// foldFleetUsage sums the ledger rows into the fleet total, the session_type split, and
// the per-session index. PURE over the rows.
//
// THE IDENTIFICATION CENSUS IS LOAD-BEARING. gatewayusageledger.Row carries an optional
// session_id, and a row written without one can never be joined to the live inventory. So
// the fold counts identified vs unidentified rows and the exposition publishes both: a
// per-session historical panel that silently covered 3 of 4000 sessions would read as a
// complete drill-down. Publishing the census makes the coverage of the drill-down itself
// a charted number.
func foldFleetUsage(rows []gatewayusageledger.Row) fleetUsageFold {
	f := fleetUsageFold{
		ByType:    map[string]*fleetUsageAgg{},
		BySession: map[string]*fleetUsageAgg{},
	}
	for _, r := range rows {
		f.Rows++
		f.Total.add(r)

		typ := strings.TrimSpace(r.SessionType)
		if typ == "" {
			typ = "unknown"
		}
		if f.ByType[typ] == nil {
			f.ByType[typ] = &fleetUsageAgg{SessionType: typ}
		}
		f.ByType[typ].add(r)

		if id := strings.TrimSpace(r.SessionID); id != "" {
			f.Identified++
			if f.BySession[id] == nil {
				f.BySession[id] = &fleetUsageAgg{SessionType: typ}
			}
			f.BySession[id].add(r)
		} else {
			f.Unidentified++
		}

		if r.UnixMillis > 0 {
			if f.FirstUnixMillis == 0 || r.UnixMillis < f.FirstUnixMillis {
				f.FirstUnixMillis = r.UnixMillis
			}
			if r.UnixMillis > f.LastUnixMillis {
				f.LastUnixMillis = r.UnixMillis
			}
		}
	}
	return f
}

// renderFleetUsageExposition projects the historical fold into the fak_fleet_usage_*
// families. PURE over its inputs.
func renderFleetUsageExposition(w *promWriter, f fleetUsageFold, maxSessions, duplicateRowsDropped int) {
	w.gauge("fak_fleet_usage_rows", "HISTORICAL: gateway-usage ledger rows folded into this exposition (after dedupe by row key).", float64(f.Rows))
	w.gauge("fak_fleet_usage_duplicate_rows_dropped", "HISTORICAL: ledger rows dropped as byte-identical duplicates of an already-folded row key (a retried flush). Ledger health, not fleet activity.", float64(duplicateRowsDropped))
	w.gauge("fak_fleet_usage_sessions_identified", "HISTORICAL: folded rows carrying a session_id — the share of the corpus the per-session drill-down can speak for.", float64(f.Identified))
	w.gauge("fak_fleet_usage_sessions_unidentified", "HISTORICAL: folded rows with NO session_id. These count in every fleet total but can never appear in a per-session panel; a large value means the drill-down is blind to most of the corpus.", float64(f.Unidentified))
	if f.FirstUnixMillis > 0 {
		w.gauge("fak_fleet_usage_window_start_unixtime", "HISTORICAL: earliest folded row's timestamp.", float64(f.FirstUnixMillis)/1000)
	}
	if f.LastUnixMillis > 0 {
		w.gauge("fak_fleet_usage_window_end_unixtime", "HISTORICAL: latest folded row's timestamp. A value that stops advancing means sessions stopped writing, not that the fleet went quiet.", float64(f.LastUnixMillis)/1000)
	}

	emitFleetUsageFamilies(w, "fak_fleet_usage_", "HISTORICAL (fleet total)", []fleetUsageSeries{{agg: &f.Total}})

	byType := make([]fleetUsageSeries, 0, len(f.ByType))
	for _, typ := range fleetSortedKeys(f.ByType) {
		byType = append(byType, fleetUsageSeries{labels: []string{"session_type", typ}, agg: f.ByType[typ]})
	}
	emitFleetUsageFamilies(w, "fak_fleet_usage_by_type_", "HISTORICAL (per session_type)", byType)

	if maxSessions <= 0 || len(f.BySession) == 0 {
		w.gauge("fak_fleet_usage_sessions_truncated", "HISTORICAL: identified sessions the --max-sessions bound kept OUT of the per-session usage families.", float64(boundedDrop(len(f.BySession), maxSessions)))
		return
	}
	// Keep the heaviest sessions: a cardinality cap that dropped the biggest spender
	// would defeat the reason an operator opens a per-session cost panel.
	ids := fleetSortedKeys(f.BySession)
	sort.SliceStable(ids, func(i, j int) bool {
		ti := f.BySession[ids[i]].InputTokens + f.BySession[ids[i]].OutputTokens + f.BySession[ids[i]].CachedPromptTokens
		tj := f.BySession[ids[j]].InputTokens + f.BySession[ids[j]].OutputTokens + f.BySession[ids[j]].CachedPromptTokens
		if ti != tj {
			return ti > tj
		}
		return ids[i] < ids[j]
	})
	dropped := 0
	if len(ids) > maxSessions {
		dropped = len(ids) - maxSessions
		ids = ids[:maxSessions]
	}
	w.gauge("fak_fleet_usage_sessions_truncated", "HISTORICAL: identified sessions the --max-sessions bound kept OUT of the per-session usage families.", float64(dropped))

	sort.Strings(ids)
	perSession := make([]fleetUsageSeries, 0, len(ids))
	for _, id := range ids {
		perSession = append(perSession, fleetUsageSeries{labels: []string{"session", id}, agg: f.BySession[id]})
	}
	emitFleetUsageFamilies(w, "fak_fleet_usage_session_", "HISTORICAL (per session)", perSession)
}

// fleetUsageSeries binds one aggregate to the labels that identify it (none for the fleet
// total, session_type=… for the split, session=… for the drill-down).
type fleetUsageSeries struct {
	labels []string
	agg    *fleetUsageAgg
}

// emitFleetUsageFamilies writes the historical families for a whole TIER under one name
// prefix, FAMILY-MAJOR: one pass per family with every series inside it, so all samples
// of a family stay contiguous as the text exposition format requires.
//
// The prefix, not a label, is what separates the fleet total from the per-type and
// per-session tiers. A single family carrying both a total series and its own breakdown
// would double count under any naive sum() in PromQL — the classic exposition trap — so
// the three tiers can never be summed together by accident.
func emitFleetUsageFamilies(w *promWriter, prefix, tier string, series []fleetUsageSeries) {
	if len(series) == 0 {
		return
	}
	for _, fam := range []struct {
		suffix, help string
		val          func(*fleetUsageAgg) float64
	}{
		{"sessions", "sessions folded (a carryforward row expands to the row count it stands for).", func(a *fleetUsageAgg) float64 { return float64(a.Sessions) }},
		{"seconds", "summed session wall-clock uptime, in seconds.", func(a *fleetUsageAgg) float64 { return a.Seconds }},
		{"turns", "OBSERVED served turns.", func(a *fleetUsageAgg) float64 { return float64(a.ObservedTurns) }},
		{"cached_turns", "OBSERVED turns that got a provider prompt-cache read.", func(a *fleetUsageAgg) float64 { return float64(a.CachedTurns) }},
		{"input_tokens", "OBSERVED (provider-relayed) uncached prompt tokens.", func(a *fleetUsageAgg) float64 { return float64(a.InputTokens) }},
		{"output_tokens", "OBSERVED (provider-relayed) completion tokens.", func(a *fleetUsageAgg) float64 { return float64(a.OutputTokens) }},
		{"cache_read_tokens", "OBSERVED prompt tokens served from the provider's cache.", func(a *fleetUsageAgg) float64 { return float64(a.CachedPromptTokens) }},
		{"cache_creation_tokens", "OBSERVED prompt tokens written INTO the provider's cache (a cost, not a saving).", func(a *fleetUsageAgg) float64 { return float64(a.CacheCreationTokens) }},
		{"kv_prefix_prompt_tokens", "WITNESSED in-kernel prompt tokens eligible for KV-prefix reuse (fak-authored, not provider-relayed).", func(a *fleetUsageAgg) float64 { return float64(a.KVPrefixPromptTokens) }},
		{"kv_prefix_reused_tokens", "WITNESSED in-kernel prompt tokens actually reused from the kernel-owned KV cache (fak-authored).", func(a *fleetUsageAgg) float64 { return float64(a.KVPrefixReusedTokens) }},
		{"adjudications", "total kernel adjudication decisions.", func(a *fleetUsageAgg) float64 { return float64(a.Adjudications) }},
	} {
		for _, s := range series {
			if s.agg != nil {
				w.gauge(prefix+fam.suffix, tier+": "+fam.help, fam.val(s.agg), s.labels...)
			}
		}
	}
	// The ratio is skipped per-series when its denominator is zero, so a session that
	// folded no prompt tokens is ABSENT from the family rather than reading 0% — a
	// distinction a "lowest cache hit rate" panel would otherwise get exactly backwards.
	for _, s := range series {
		if s.agg == nil {
			continue
		}
		if ratio := s.agg.CacheReadRatio(); ratio >= 0 {
			w.gauge(prefix+"cache_read_ratio", tier+": cache_read / (input + cache_read + cache_creation). Absent (no series) when no prompt tokens were folded — unmeasured, not 0%.", ratio, s.labels...)
		}
	}
	for _, v := range []struct {
		verdict string
		val     func(*fleetUsageAgg) uint64
	}{
		{"allowed", func(a *fleetUsageAgg) uint64 { return a.Allowed }},
		{"denied", func(a *fleetUsageAgg) uint64 { return a.Denied }},
		{"transformed", func(a *fleetUsageAgg) uint64 { return a.Transformed }},
		{"quarantined", func(a *fleetUsageAgg) uint64 { return a.Quarantined }},
		{"deferred", func(a *fleetUsageAgg) uint64 { return a.Deferred }},
		{"escalated", func(a *fleetUsageAgg) uint64 { return a.Escalated }},
		{"errored", func(a *fleetUsageAgg) uint64 { return a.Errored }},
	} {
		for _, s := range series {
			if s.agg == nil {
				continue
			}
			w.gauge(prefix+"adjudications_by_verdict", tier+": kernel adjudication decisions per verdict.", float64(v.val(s.agg)), append(append([]string(nil), s.labels...), "verdict", v.verdict)...)
		}
	}
}

// ---- serve -----------------------------------------------------------------------

// fleetMetricsMux builds the HTTP surface: /metrics re-folds per scrape (so a session
// appearing or dying between scrapes is visible), and the root points a browser at it.
// Split from serveFleetMetrics so a test can exercise the handler without binding a port.
func fleetMetricsMux(src fleetMetricsSources) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		io.WriteString(w, src.render(time.Now()))
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.WriteString(w, "fak fleet metrics — the fleet-first session exposition.\nScrape /metrics for the LIVE fak_fleet_session_* / fak_fleet_sessions* families and the HISTORICAL fak_fleet_usage_* roll-ups.\n")
	})
	return mux
}

func serveFleetMetrics(stdout, stderr io.Writer, src fleetMetricsSources, addr string) int {
	fmt.Fprintf(stdout, "fak fleet metrics: serving /metrics on http://%s/metrics\n", addr)
	srv := &http.Server{Addr: addr, Handler: fleetMetricsMux(src)}
	if err := srv.ListenAndServe(); err != nil {
		fmt.Fprintf(stderr, "fak fleet metrics: serve: %v\n", err)
		return 1
	}
	return 0
}

// ---- small helpers ---------------------------------------------------------------

func boolGauge(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// fleetHostLabel keeps an unset host addressable on a panel. A C1 descriptor written by a
// host that did not stamp itself carries "", which would render as an unclickable blank
// legend entry; "unknown" is the honest, selectable label for it.
func fleetHostLabel(h string) string {
	if h = strings.TrimSpace(h); h != "" {
		return h
	}
	return "unknown"
}

func fleetDashIfEmpty(s string) string {
	if s = strings.TrimSpace(s); s != "" {
		return s
	}
	return "-"
}

// boundedDrop reports how many entries a cap of max would exclude from n.
func boundedDrop(n, max int) int {
	if max <= 0 {
		return n
	}
	if n <= max {
		return 0
	}
	return n - max
}

// fleetSortedKeys is the generic key sorter this file needs (cmd/fak already has a
// non-generic sortedKeys over map[string]int in audit_diagnose.go; the fold indexes
// aggregate pointers, so it needs its own).
func fleetSortedKeys[V any](m map[string]V) []string {
	out := maputil.SortedKeys(m)
	return out
}

// sortedKeysByPCBOrder orders states the way the CLI headline does (RUNNING, THROTTLED,
// PAUSED, DRAINING, STALLED, STOPPED, then unknown states alphabetically), so a stacked
// panel's series order matches what an operator reads in the terminal.
func sortedKeysByPCBOrder(m map[string]int) []string {
	out := fleetSortedKeys(m)
	sort.SliceStable(out, func(i, j int) bool {
		oi, iok := pcbStateOrder[out[i]]
		oj, jok := pcbStateOrder[out[j]]
		if iok && jok {
			return oi < oj
		}
		if iok != jok {
			return iok
		}
		return out[i] < out[j]
	})
	return out
}
