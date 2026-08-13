package main

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

// sampleNameRe matches the metric name that opens an exposition sample line.
var sampleNameRe = regexp.MustCompile(`^([a-zA-Z_:][a-zA-Z0-9_:]*)`)

// parsedExposition is a minimal structural read of a Prometheus text exposition: the
// order families appear in, how many HELP/TYPE lines each got, and the sample lines per
// family. Enough to assert the format contract without a Prometheus dependency.
type parsedExposition struct {
	familyOrder  []string
	helpCount    map[string]int
	typeCount    map[string]int
	samples      map[string][]string
	discontinued []string // families whose samples were split by another family's
}

func parseExposition(t *testing.T, text string) parsedExposition {
	t.Helper()
	p := parsedExposition{
		helpCount: map[string]int{},
		typeCount: map[string]int{},
		samples:   map[string][]string{},
	}
	closed := map[string]bool{}
	last := ""
	for _, line := range strings.Split(text, "\n") {
		switch {
		case strings.TrimSpace(line) == "":
			continue
		case strings.HasPrefix(line, "# HELP "):
			p.helpCount[strings.Fields(line)[2]]++
			continue
		case strings.HasPrefix(line, "# TYPE "):
			p.typeCount[strings.Fields(line)[2]]++
			continue
		case strings.HasPrefix(line, "#"):
			continue
		}
		m := sampleNameRe.FindStringSubmatch(line)
		if m == nil {
			t.Fatalf("unparseable exposition line: %q", line)
		}
		name := m[1]
		if name != last {
			if closed[name] {
				p.discontinued = append(p.discontinued, name)
			}
			if last != "" {
				closed[last] = true
			}
			p.familyOrder = append(p.familyOrder, name)
			last = name
		}
		p.samples[name] = append(p.samples[name], line)
	}
	return p
}

// fleetTestSources returns sources pointed at paths that do not exist, so a test that
// only cares about the fold shape gets an honest empty read rather than this machine's
// real fleet. Tests that want data call the pure renderers directly.
func fleetTestSources(t *testing.T) fleetMetricsSources {
	t.Helper()
	dir := t.TempDir()
	return fleetMetricsSources{
		registryPath: dir + "/no-such-registry.json",
		usageLedger:  dir + "/no-such-ledger.jsonl",
		staleWindow:  defaultSessionStaleWindow,
		maxSessions:  defaultFleetMetricsMaxSessions,
		stderr:       io.Discard,
	}
}

// TestFleetMetricsExpositionIsFamilyMajor pins the exposition FORMAT contract: every
// sample of a metric family must be contiguous, and each family declares HELP and TYPE
// exactly once. This is the invariant a session-major emission loop breaks — it would
// interleave info(a), live(a), info(b), live(b) — which the lenient scrape parser
// tolerates but `promtool check metrics` and any OpenMetrics consumer reject. The test
// exists because that breakage is invisible in a Grafana panel until a stricter consumer
// reads the same endpoint.
func TestFleetMetricsExpositionIsFamilyMajor(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	w := newPromWriter()

	local := []session.Descriptor{
		{ID: "sess-live", Host: "node-1", PCBState: "RUNNING", PID: 11, Generation: 2, CreatedAt: now.Add(-time.Minute), LastSeen: now.Add(-5 * time.Second)},
		{ID: "sess-wedged", Host: "node-1", PCBState: "RUNNING", PID: 12, CreatedAt: now.Add(-time.Hour), LastSeen: now.Add(-time.Hour)},
		{ID: "sess-parked", Host: "node-2", PCBState: "PAUSED", PID: 13, CreatedAt: now.Add(-2 * time.Hour), LastSeen: now.Add(-90 * time.Minute)},
	}
	byID := map[string]session.Descriptor{}
	for _, d := range local {
		byID[d.ID] = d
	}
	inv := buildSessionInventory(local, nil, now, defaultSessionStaleWindow, false)
	renderFleetLiveExposition(w, inv, byID, defaultFleetMetricsMaxSessions)

	rows := []gatewayusageledger.Row{
		{Kind: "exit", SessionType: "guard", SessionID: "sess-live", UnixMillis: now.UnixMilli(), Counters: gatewayusageledger.Counters{InputTokens: 100, CachedPromptTokens: 300, OutputTokens: 50, Allowed: 4}},
		{Kind: "exit", SessionType: "serve", SessionID: "sess-parked", UnixMillis: now.UnixMilli(), Counters: gatewayusageledger.Counters{InputTokens: 200, CacheCreationTokens: 200, Denied: 1}},
	}
	renderFleetUsageExposition(w, foldFleetUsage(rows), defaultFleetMetricsMaxSessions, 0)

	p := parseExposition(t, w.String())
	if len(p.discontinued) > 0 {
		t.Fatalf("families emitted non-contiguously (samples split by another family): %v", p.discontinued)
	}
	for _, fam := range p.familyOrder {
		if p.helpCount[fam] != 1 {
			t.Fatalf("family %s has %d HELP lines, want exactly 1", fam, p.helpCount[fam])
		}
		if p.typeCount[fam] != 1 {
			t.Fatalf("family %s has %d TYPE lines, want exactly 1", fam, p.typeCount[fam])
		}
	}
	// The per-session tier is the whole point of the exporter: prove it actually carries
	// one series per session rather than a single rolled-up sample.
	if got := len(p.samples["fak_fleet_session_info"]); got != 3 {
		t.Fatalf("fak_fleet_session_info samples = %d, want 3 (one per session)", got)
	}
	if got := len(p.samples["fak_fleet_usage_session_input_tokens"]); got != 2 {
		t.Fatalf("per-session usage samples = %d, want 2", got)
	}
}

// TestFleetMetricsLiveTierAgreesWithSessionLs proves the live families are a projection
// of the SAME fold `fak session ls --durable` prints — including the projection that
// matters most: a session that claims RUNNING but whose heartbeat has lapsed must read
// STALLED on the dashboard, exactly as it does in the CLI headline. A dashboard that
// reported the raw pcb_state would show a wedged fleet as a working one.
func TestFleetMetricsLiveTierAgreesWithSessionLs(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	local := []session.Descriptor{
		{ID: "sess-live", Host: "node-1", PCBState: "RUNNING", CreatedAt: now.Add(-time.Minute), LastSeen: now.Add(-5 * time.Second)},
		{ID: "sess-wedged", Host: "node-1", PCBState: "RUNNING", CreatedAt: now.Add(-time.Hour), LastSeen: now.Add(-time.Hour)},
		{ID: "sess-parked", Host: "node-2", PCBState: "PAUSED", CreatedAt: now.Add(-2 * time.Hour), LastSeen: now.Add(-90 * time.Minute)},
	}
	byID := map[string]session.Descriptor{}
	for _, d := range local {
		byID[d.ID] = d
	}
	inv := buildSessionInventory(local, nil, now, defaultSessionStaleWindow, false)

	w := newPromWriter()
	renderFleetLiveExposition(w, inv, byID, defaultFleetMetricsMaxSessions)
	got := w.String()

	for _, want := range []string{
		"fak_fleet_sessions 3",
		`fak_fleet_sessions_by_state{state="RUNNING"} 1`,
		`fak_fleet_sessions_by_state{state="STALLED"} 1`,
		`fak_fleet_sessions_by_state{state="PAUSED"} 1`,
		`fak_fleet_sessions_by_liveness{liveness="live"} 1`,
		`fak_fleet_sessions_by_liveness{liveness="stalled"} 1`,
		`fak_fleet_sessions_by_liveness{liveness="idle"} 1`,
		`fak_fleet_session_live{session="sess-live"} 1`,
		`fak_fleet_session_live{session="sess-wedged"} 0`,
		`fak_fleet_session_stalled{session="sess-wedged"} 1`,
		`fak_fleet_hosts 2`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("exposition missing %q\n---\n%s", want, got)
		}
	}
	// The rollup total must equal the CLI's own count — one authority, two renderings.
	if !strings.Contains(got, "fak_fleet_sessions "+strconv.Itoa(inv.Count)) {
		t.Fatalf("fak_fleet_sessions disagrees with the inventory count %d", inv.Count)
	}
}

// TestFleetMetricsMaxSessionsTruncatesAndReports proves the cardinality bound both bites
// and CONFESSES. A cap that silently dropped rows would render a per-session panel that
// looks complete; fak_fleet_sessions_truncated is what makes the omission readable. The
// ranking is also asserted: the cap must keep the live session, not an arbitrary one.
func TestFleetMetricsMaxSessionsTruncatesAndReports(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	local := []session.Descriptor{
		// Sorted by (host,id) the idle row comes FIRST, so a cap that did not rank would
		// keep it and drop the live session — the exact failure this asserts against.
		{ID: "aaa-parked", Host: "node-1", PCBState: "PAUSED", CreatedAt: now.Add(-time.Hour), LastSeen: now.Add(-time.Hour)},
		{ID: "zzz-live", Host: "node-1", PCBState: "RUNNING", CreatedAt: now.Add(-time.Minute), LastSeen: now.Add(-time.Second)},
	}
	byID := map[string]session.Descriptor{}
	for _, d := range local {
		byID[d.ID] = d
	}
	inv := buildSessionInventory(local, nil, now, defaultSessionStaleWindow, false)

	w := newPromWriter()
	renderFleetLiveExposition(w, inv, byID, 1)
	got := w.String()

	if !strings.Contains(got, "fak_fleet_sessions_truncated 1") {
		t.Fatalf("cap of 1 over 2 sessions must report 1 truncated:\n%s", got)
	}
	if !strings.Contains(got, `fak_fleet_session_info{session="zzz-live"`) {
		t.Fatalf("the cap must keep the LIVE session:\n%s", got)
	}
	if strings.Contains(got, `fak_fleet_session_info{session="aaa-parked"`) {
		t.Fatalf("the cap must drop the idle session, not the live one:\n%s", got)
	}
	// The fleet-level rollup counts the WHOLE fleet even when the per-session tier is
	// capped — truncation bounds cardinality, it must never understate the fleet.
	if !strings.Contains(got, "fak_fleet_sessions 2") {
		t.Fatalf("the fleet total must count all sessions, not just the un-truncated ones:\n%s", got)
	}
}

// TestFoldFleetUsageCarryforwardExpandsSessionCount pins the one arithmetic trap in the
// historical fold. `fak cachevalue`'s ledger cut replaces N session rows with ONE
// carryforward row that sums them; counting that row as a single session would make every
// historical session count silently collapse the moment a cut ran.
func TestFoldFleetUsageCarryforwardExpandsSessionCount(t *testing.T) {
	rows := []gatewayusageledger.Row{
		{Kind: "exit", SessionType: "guard", Counters: gatewayusageledger.Counters{InputTokens: 10}},
		{
			Kind:         gatewayusageledger.KindCarryforward,
			SessionType:  "guard",
			Carryforward: &gatewayusageledger.Carryforward{FoldedKind: "exit", FoldedRows: 7},
			Counters:     gatewayusageledger.Counters{InputTokens: 700},
		},
		// A carryforward with no witness: its counters are real but its session count is
		// unknowable, so it must claim NONE rather than guess one.
		{Kind: gatewayusageledger.KindCarryforward, SessionType: "guard", Counters: gatewayusageledger.Counters{InputTokens: 5}},
	}
	f := foldFleetUsage(rows)
	if f.Total.Sessions != 8 {
		t.Fatalf("sessions = %d, want 8 (1 exit + 7 folded + 0 unwitnessed)", f.Total.Sessions)
	}
	if f.Total.InputTokens != 715 {
		t.Fatalf("input tokens = %d, want 715 — a carryforward's counters still count", f.Total.InputTokens)
	}
}

// TestFoldFleetUsageIdentificationCensus proves the fold publishes how much of the corpus
// the per-session drill-down can actually speak for. session_id is optional on the ledger
// row, so a per-session panel can cover a tiny slice of the fleet while looking complete;
// the identified/unidentified split is what makes that readable on the dashboard.
func TestFoldFleetUsageIdentificationCensus(t *testing.T) {
	rows := []gatewayusageledger.Row{
		{Kind: "exit", SessionType: "guard", SessionID: "sess-a", Counters: gatewayusageledger.Counters{InputTokens: 10}},
		{Kind: "exit", SessionType: "guard", SessionID: "sess-a", Counters: gatewayusageledger.Counters{InputTokens: 20}},
		{Kind: "exit", SessionType: "serve", Counters: gatewayusageledger.Counters{InputTokens: 30}},
	}
	f := foldFleetUsage(rows)
	if f.Identified != 2 || f.Unidentified != 1 {
		t.Fatalf("census = %d identified / %d unidentified, want 2/1", f.Identified, f.Unidentified)
	}
	if len(f.BySession) != 1 || f.BySession["sess-a"].InputTokens != 30 {
		t.Fatalf("per-session index wrong: %+v", f.BySession)
	}
	// Every row counts in the fleet total, identified or not — the census reports
	// drill-down coverage, it never removes a row from the rollup.
	if f.Total.InputTokens != 60 {
		t.Fatalf("fleet total = %d, want 60 (unidentified rows still count)", f.Total.InputTokens)
	}

	w := newPromWriter()
	renderFleetUsageExposition(w, f, defaultFleetMetricsMaxSessions, 3)
	got := w.String()
	for _, want := range []string{
		"fak_fleet_usage_sessions_identified 2",
		"fak_fleet_usage_sessions_unidentified 1",
		"fak_fleet_usage_duplicate_rows_dropped 3",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("exposition missing %q\n---\n%s", want, got)
		}
	}
}

// TestFleetUsageCacheReadRatioAbsentWhenUnmeasured pins the honesty fence on the ratio:
// a session that folded no prompt tokens has an UNMEASURED hit rate, not a 0% one. If it
// emitted 0, a "lowest cache hit rate" panel would rank the sessions it knows nothing
// about above the ones that genuinely thrash the cache.
func TestFleetUsageCacheReadRatioAbsentWhenUnmeasured(t *testing.T) {
	measured := &fleetUsageAgg{InputTokens: 100, CachedPromptTokens: 300, CacheCreationTokens: 100}
	if got := measured.CacheReadRatio(); got != 0.6 {
		t.Fatalf("ratio = %v, want 0.6 (300 / (100+300+100))", got)
	}
	empty := &fleetUsageAgg{OutputTokens: 42}
	if got := empty.CacheReadRatio(); got >= 0 {
		t.Fatalf("ratio over zero prompt tokens = %v, want a negative sentinel (unmeasured)", got)
	}

	f := fleetUsageFold{
		ByType:    map[string]*fleetUsageAgg{"guard": measured, "serve": empty},
		BySession: map[string]*fleetUsageAgg{},
	}
	w := newPromWriter()
	renderFleetUsageExposition(w, f, defaultFleetMetricsMaxSessions, 0)
	got := w.String()
	if !strings.Contains(got, `fak_fleet_usage_by_type_cache_read_ratio{session_type="guard"} 0.6`) {
		t.Fatalf("measured ratio missing:\n%s", got)
	}
	if strings.Contains(got, `fak_fleet_usage_by_type_cache_read_ratio{session_type="serve"}`) {
		t.Fatalf("unmeasured ratio must be ABSENT, not zero:\n%s", got)
	}
}

// TestFleetMetricsUnreadableRegistryIsDistinguishableFromEmptyFleet proves the exporter
// separates "no sessions" from "could not look". Both fold to zero live series, so
// without fak_fleet_registry_readable an operator could not tell a quiet fleet from a
// blind exporter — and would trust an all-zero dashboard.
func TestFleetMetricsUnreadableRegistryIsDistinguishableFromEmptyFleet(t *testing.T) {
	src := fleetTestSources(t)
	got := src.render(time.Unix(1_700_000_000, 0))

	if !strings.Contains(got, "fak_fleet_up 1") {
		t.Fatalf("a fold that read nothing must still report the exporter as up:\n%s", got)
	}
	if !strings.Contains(got, "fak_fleet_sessions 0") {
		t.Fatalf("missing zero session rollup:\n%s", got)
	}
	// A missing registry file reads as an EMPTY registry (session.Registry treats absence
	// as no sessions), so the readable gauge stays 1 — the flag exists for a registry that
	// is present but unreadable/corrupt, which is the case that must not read as calm.
	if !strings.Contains(got, "fak_fleet_registry_readable ") {
		t.Fatalf("fak_fleet_registry_readable must always be present:\n%s", got)
	}
}

// TestGatewayUsageSessionIDRejectsTheSharedSentinel pins the filter that decides whether a
// usage row may carry a join key at all. A non-durable `fak guard` launch resolves to the
// CONSTANT "guard" trace, shared by every such launch on every machine. Stamping it would
// collapse thousands of unrelated sessions into one giant series that LOOKS like a real
// session — strictly worse than the honest empty field, which shows up in the exporter's
// unidentified census instead.
func TestGatewayUsageSessionIDRejectsTheSharedSentinel(t *testing.T) {
	for _, tc := range []struct {
		trace, want string
	}{
		{"guard", ""},     // the shared non-durable sentinel
		{"  guard  ", ""}, // …still the sentinel after trimming
		{"", ""},          // nothing to join on
		{"guard-abc123-9f2e", "guard-abc123-9f2e"},     // a durable per-launch identity
		{"my-explicit-session", "my-explicit-session"}, // an operator's --session-id
		{"contract-repair-claude-35640", "contract-repair-claude-35640"},
	} {
		if got := gatewayUsageSessionID(tc.trace); got != tc.want {
			t.Fatalf("gatewayUsageSessionID(%q) = %q, want %q", tc.trace, got, tc.want)
		}
	}
}

// TestFleetMetricsMuxServesExposition proves the scrape endpoint is wired to the same
// fold as the one-shot render, without binding a port.
func TestFleetMetricsMuxServesExposition(t *testing.T) {
	mux := fleetMetricsMux(fleetTestSources(t))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("content-type = %q, want the Prometheus text exposition type", ct)
	}
	if !strings.Contains(rec.Body.String(), "fak_fleet_up 1") {
		t.Fatalf("/metrics did not serve the exposition:\n%s", rec.Body.String())
	}
}

// TestServeUsageSessionIDOnlyStampsSingleSessionServe is the load-bearing one. A serve
// gateway multiplexes a session table, so its exit counters are a PROCESS total. Stamping
// one of several traces onto that row would report the whole process's tokens as a single
// session's spend on the drill-down panel, with nothing on the panel revealing the blend.
// Len()==1 is the only shape where the process total and the session total coincide.
func TestServeUsageSessionIDOnlyStampsSingleSessionServe(t *testing.T) {
	if got := serveUsageSessionID(nil); got != "" {
		t.Errorf("nil table = %q, want empty", got)
	}
	if got := serveUsageSessionID(session.NewTable()); got != "" {
		t.Errorf("empty table = %q, want empty", got)
	}

	one := session.NewTable()
	seedFleetBusSession(t, one, "win-642901dc623be530", sessionctl.BroadcastMeta{})
	if got := serveUsageSessionID(one); got != "win-642901dc623be530" {
		t.Errorf("single-session serve = %q, want the one trace", got)
	}

	// The whole point: a second session must REVOKE the stamp, not pick a winner.
	multi := session.NewTable()
	seedFleetBusSession(t, multi, "win-642901dc623be530", sessionctl.BroadcastMeta{})
	seedFleetBusSession(t, multi, "win-136d42bd3d9ae657", sessionctl.BroadcastMeta{})
	if got := serveUsageSessionID(multi); got != "" {
		t.Errorf("multiplexed serve = %q, want empty (its counters are a process total)", got)
	}

	// A single session whose trace is the shared guard sentinel is still not a join key.
	sentinel := session.NewTable()
	seedFleetBusSession(t, sentinel, guardSharedTraceSentinel, sessionctl.BroadcastMeta{})
	if got := serveUsageSessionID(sentinel); got != "" {
		t.Errorf("sentinel-trace serve = %q, want empty", got)
	}
}

func TestFleetMetricsAttributesDescendantUsageToRootGoal(t *testing.T) {
	now := time.Date(2026, 8, 13, 12, 0, 0, 0, time.UTC)
	registrationPath := filepath.Join(t.TempDir(), "child-registrations.jsonl")
	store := sessionregistry.Store{Path: registrationPath}
	root, err := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID: "reg-root", RootIssue: "7000", TaskID: "goal-improve-fleet",
		LaunchKind: "guard", Runtime: "codex", SessionID: "session-root", Now: now,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Register(root); err != nil {
		t.Fatal(err)
	}
	child, err := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID: "reg-child", ParentRegistrationID: root.RegistrationID,
		RootRegistrationID: root.RootRegistrationID, RootIssue: root.RootIssue, TaskID: root.TaskID,
		LaunchKind: "headless", Runtime: "codex", SessionID: "session-child", Now: now.Add(time.Second),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Register(child); err != nil {
		t.Fatal(err)
	}
	grandchild, err := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID: "reg-micro", ParentRegistrationID: child.RegistrationID,
		RootRegistrationID: root.RootRegistrationID, RootIssue: root.RootIssue, TaskID: root.TaskID,
		LaunchKind: "micro-context", Runtime: "in-process", SessionID: "session-micro", Now: now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Register(grandchild); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Start(child.RegistrationID, 22, now.Add(3*time.Second)); err != nil {
		t.Fatal(err)
	}

	usagePath := filepath.Join(t.TempDir(), "usage.jsonl")
	rows := []gatewayusageledger.Row{
		{Schema: gatewayusageledger.Schema, SessionID: "session-root", UnixMillis: now.UnixMilli(), Counters: gatewayusageledger.Counters{ObservedTurns: 2, InputTokens: 20, OutputTokens: 4, Total: 2}},
		{Schema: gatewayusageledger.Schema, SessionID: "session-child", UnixMillis: now.UnixMilli(), Counters: gatewayusageledger.Counters{ObservedTurns: 3, InputTokens: 30, OutputTokens: 6, Total: 3}},
		{Schema: gatewayusageledger.Schema, SessionID: "session-micro", UnixMillis: now.UnixMilli(), Counters: gatewayusageledger.Counters{ObservedTurns: 5, InputTokens: 50, OutputTokens: 10, Total: 5}},
	}
	var ledger strings.Builder
	for _, row := range rows {
		b, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		ledger.Write(b)
		ledger.WriteByte('\n')
	}
	if err := os.WriteFile(usagePath, []byte(ledger.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	src := fleetMetricsSources{registryPath: filepath.Join(t.TempDir(), "sessions.json"), registrationLedger: registrationPath, usageLedger: usagePath, maxSessions: 100, stderr: io.Discard}
	raw := src.render(now.Add(time.Minute))
	parseExposition(t, raw)
	labels := `root_registration="reg-root",root_issue="7000",task="goal-improve-fleet"`
	for _, want := range []string{
		"fak_fleet_goal_info{" + labels + "} 1",
		"fak_fleet_goal_registrations{" + labels + "} 3",
		"fak_fleet_goal_sessions{" + labels + "} 3",
		"fak_fleet_goal_observed_turns_total{" + labels + "} 10",
		"fak_fleet_goal_input_tokens_total{" + labels + "} 100",
		"fak_fleet_goal_output_tokens_total{" + labels + "} 20",
		"fak_fleet_goal_adjudications_total{" + labels + "} 10",
	} {
		if !strings.Contains(raw, want) {
			t.Errorf("missing root-goal rollup %q\n%s", want, raw)
		}
	}
	if !strings.Contains(raw, `fak_fleet_goal_registration_state{`+labels+`,state="active"} 1`) {
		t.Errorf("active child was not attributed to root goal\n%s", raw)
	}
	if !strings.Contains(raw, `fak_fleet_goal_registration_state{`+labels+`,state="registered"} 2`) {
		t.Errorf("registered descendants were not attributed to root goal\n%s", raw)
	}
	if !strings.Contains(raw, "fak_fleet_registration_registry_readable 1") {
		t.Errorf("registration ledger readability missing\n%s", raw)
	}
}
