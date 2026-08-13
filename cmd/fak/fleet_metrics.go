package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
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
	since := fs.String("since", "", "fold only usage rows on or after this date (YYYY-MM-DD)")
	maxSessions := fs.Int("max-sessions", defaultFleetMetricsMaxSessions, "cardinality bound on the per-session label (0 disables the per-session tier entirely)")
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

	src := fleetMetricsSources{
		registryPath: *registry,
		fleet:        *fleet,
		remote:       *remote,
		staleWindow:  *stale,
		usageLedger:  *usageLedger,
		since:        *since,
		maxSessions:  *maxSessions,
		stderr:       stderr,
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
	registryPath       string
	fleet              bool
	remote             string
	staleWindow        time.Duration
	usageLedger        string
	registrationLedger string
	since              string
	maxSessions        int
	stderr             io.Writer
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

	// Dedupe BEFORE folding: a retried exit flush (or a periodic and an exit flush of one
	// snapshot landing in the same millisecond) writes the same row key twice, and a
	// double-counted session would inflate every historical panel by a silent amount.
	usageRows, dupDropped := gatewayusageledger.DedupeByKey(filterGatewayUsageSince(gatewayusageledger.ReadLedgerFile(s.usagePath()), s.since))
	usageFold := foldFleetUsage(usageRows)
	renderFleetUsageExposition(w, usageFold, s.maxSessions, dupDropped)

	registrations, registrationReadable := s.registrationInventory()
	renderFleetGoalExposition(w, registrations, usageFold)
	w.gauge("fak_fleet_registration_registry_readable", "1 when the child-registration lineage ledger was read successfully; 0 means goal-level attribution is unavailable, not that the fleet has no goals.", boolGauge(registrationReadable))

	w.gauge("fak_fleet_registry_readable", "1 when the durable session registry was read successfully; 0 when it could not be read (every live family then reads an honest zero, which is NOT the same as an empty fleet).", boolGauge(readable))
	return w.String()
}

// registrationInventory reads the execution lineage graph written before every guard /
// dispatchworker child starts. Missing is an honest empty graph; malformed or unreadable
// data flips the readability gauge so absence is never mistaken for "no goals".
func defaultChildRegistrationLedgerPath() string {
	// FAK_SESSION_REGISTRY historically names the durable PCB registry consumed by
	// --registry. Reusing it here would make the two unrelated schemas collide. The
	// child-lineage writer uses its own documented default; an operator sharing a custom
	// lineage store names it explicitly with --registration-ledger.
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "fak", "child-registrations.jsonl")
	}
	return filepath.Join(".fak", "child-registrations.jsonl")
}

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

type fleetGoalAgg struct {
	rootID    string
	rootIssue string
	taskID    string
	states    map[sessionregistry.State]int
	sessions  map[string]struct{}
	usage     fleetUsageAgg
}

// renderFleetGoalExposition is the root-goal join: every descendant registration is
// folded under its durable root_registration_id, and historical gateway usage joins by
// the registered session id. A parent, a headless child, and a micro-context grandchild
// therefore contribute to one bounded, queryable goal series without relying on process
// nesting or an agent's self-report.
func renderFleetGoalExposition(w *promWriter, rows []sessionregistry.Record, usage fleetUsageFold) {
	goals := map[string]*fleetGoalAgg{}
	for _, r := range rows {
		root := strings.TrimSpace(r.RootRegistrationID)
		if root == "" {
			continue
		}
		g := goals[root]
		if g == nil {
			g = &fleetGoalAgg{rootID: root, states: map[sessionregistry.State]int{}, sessions: map[string]struct{}{}}
			goals[root] = g
		}
		if g.rootIssue == "" {
			g.rootIssue = strings.TrimSpace(r.RootIssue)
		}
		if g.taskID == "" {
			g.taskID = strings.TrimSpace(r.TaskID)
		}
		g.states[r.State]++
		sid := strings.TrimSpace(r.Identity.SessionID)
		if sid != "" {
			g.sessions[sid] = struct{}{}
			if a, ok := usage.BySession[sid]; ok {
				g.usage.merge(*a)
			}
		}
	}
	for _, root := range fleetSortedKeys(goals) {
		g := goals[root]
		labels := []string{"root_registration", g.rootID, "root_issue", fleetDashIfEmpty(g.rootIssue), "task", fleetDashIfEmpty(g.taskID)}
		w.gauge("fak_fleet_goal_info", "Durable root-goal identity. Descendant registrations and joined usage share these labels.", 1, labels...)
		w.gauge("fak_fleet_goal_registrations", "Latest child registrations attributed to this root goal across all descendant depths.", float64(sumGoalStates(g.states)), labels...)
		w.gauge("fak_fleet_goal_sessions", "Distinct non-empty session ids registered under this root goal.", float64(len(g.sessions)), labels...)
		for _, state := range sortedGoalStates(g.states) {
			w.gauge("fak_fleet_goal_registration_state", "Registrations under a root goal by latest lifecycle state.", float64(g.states[state]), append(labels, "state", string(state))...)
		}
		w.gauge("fak_fleet_goal_observed_turns_total", "Historical observed turns joined from every registered session under this root goal.", float64(g.usage.ObservedTurns), labels...)
		w.gauge("fak_fleet_goal_input_tokens_total", "Historical input tokens joined from every registered session under this root goal.", float64(g.usage.InputTokens), labels...)
		w.gauge("fak_fleet_goal_output_tokens_total", "Historical output tokens joined from every registered session under this root goal.", float64(g.usage.OutputTokens), labels...)
		w.gauge("fak_fleet_goal_adjudications_total", "Historical policy adjudications joined from every registered session under this root goal.", float64(g.usage.Adjudications), labels...)
	}
}

func (a *fleetUsageAgg) merge(b fleetUsageAgg) {
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
	a.Allowed += b.Allowed
	a.Denied += b.Denied
	a.Transformed += b.Transformed
	a.Quarantined += b.Quarantined
	a.Deferred += b.Deferred
	a.Escalated += b.Escalated
	a.Errored += b.Errored
}

func sumGoalStates(states map[sessionregistry.State]int) int {
	n := 0
	for _, v := range states {
		n += v
	}
	return n
}
func sortedGoalStates(states map[sessionregistry.State]int) []sessionregistry.State {
	out := make([]sessionregistry.State, 0, len(states))
	for state := range states {
		out = append(out, state)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
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
		id := strings.TrimSpace(d.ID)
		if id == "" {
			id = strings.TrimSpace(d.Trace)
		}
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
	if s.stderr == nil {
		return
	}
	fmt.Fprintf(s.stderr, "fak fleet metrics: "+format+"\n", args...)
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
	Allowed              uint64
	Denied               uint64
	Transformed          uint64
	Quarantined          uint64
	Deferred             uint64
	Escalated            uint64
	Errored              uint64
}

func (a *fleetUsageAgg) add(r gatewayusageledger.Row) {
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
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
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
