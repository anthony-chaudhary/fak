package gateway

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// writeDenyAllMetrics renders the deny-all stop family: the cumulative count of turns the
// floor refused entirely (forcing an unchosen end_turn) and the live consecutive-run gauge
// the guard Stop-hook polls. Both are 0 on a healthy session, so the surface stays quiet
// until the floor actually refuses a whole turn.
func (m *gatewayMetrics) writeDenyAllMetrics(b *strings.Builder) {
	stops, consec := m.denyAllSnapshot()
	sameConsec := m.denyAllSameSnapshot()
	feedbackTurns, feedbackConsec := m.toolFeedbackSnapshot()
	writeCounter(b, "fak_guard_deny_all_stops_total",
		"Served turns whose EVERY proposed tool call the capability floor refused, forcing the wire to report end_turn (a stop the agent did not choose; the v0.15.0 contract that keeps the client from hanging on a dropped tool_use block). The guard --deny-all-continue Stop-hook reads the consecutive gauge below to auto-continue the agent past these.",
		int64(stops))
	writeCounter(b, "fak_gateway_policy_canary_rollbacks_total",
		"Capability-floor reloads automatically rolled back after a deny-all anomaly.",
		int64(m.policyCanaryRollbacks.Load()))
	writeHelpType(b, "fak_guard_deny_all_consecutive",
		"Consecutive deny-all turns ending the most recent served turn (reset to 0 by any turn with a surviving or no tool call). Blind to WHICH call was refused — kept for observability; the guard Stop hook now keys its give-up on the same-issue gauge below instead.",
		"gauge")
	fmt.Fprintf(b, "fak_guard_deny_all_consecutive %d\n", consec)
	writeHelpType(b, "fak_guard_deny_all_same_consecutive",
		"Consecutive deny-all turns proposing the IDENTICAL refused action (same tool + same reason). Reset to 0 by any non-deny-all turn and re-seeded to 1 whenever the refused tool/reason changes, so a session hitting a fresh block each turn pins this at 1. The same-issue signal the guard --deny-all-continue Stop-hook keys its bounded give-up on: only a true repeated same issue (this gauge climbing to the give-up depth) stops the session; a varied session is never stopped.",
		"gauge")
	fmt.Fprintf(b, "fak_guard_deny_all_same_consecutive %d\n", sameConsec)
	writeCounter(b, "fak_guard_tool_feedback_turns_total",
		"Served turns whose EVERY proposed tool call was rejected as retryable model feedback (for example MALFORMED JSON/args). These turns may need guard auto-continue, but they are NOT hard deny-all stops and do not drive the bounded give-up policy.",
		int64(feedbackTurns))
	writeHelpType(b, "fak_guard_tool_feedback_consecutive",
		"Consecutive retryable tool-feedback turns ending the most recent served turn. The guard Stop-hook may continue the turn from this signal, but a session stop must come from a separate declared stop policy, not from malformed tool calls accidentally accumulating as hard deny-all.",
		"gauge")
	fmt.Fprintf(b, "fak_guard_tool_feedback_consecutive %d\n", feedbackConsec)
	writeCounter(b, "fak_mcp_verb_calls_total",
		"Admitted MCP fak-verb (tools/call) invocations since process start — the signal that fak was USED as a substrate, not merely present as a passive guard. The guard Stop-hook reads this to warn when a long run ends having called ZERO fak verbs (the unused-substrate pathology: fak available but inert).",
		int64(m.fakVerbCallsSnapshot()))
}

// writeSessionMetrics renders the fak_sessions{state} gauge (#1204): a read-time fold
// of the live session registry (C1 — the same listSessions snapshot /v1/fak/sessions
// and /debug/vars project) into a per-state count, so "how many sessions are running,
// by state" is scrapeable, not narratable. The fold is read-time off the live snapshot
// (never a per-turn increment), so a registry that GCs a session naturally decrements
// its bucket at the next scrape — no leaked counts, no stale descriptor.
//
// A nil listSessions injection (the default serve path) suppresses the family entirely
// — the same fail-closed posture as GET /v1/fak/sessions — rather than emitting a
// phantom all-zero surface. Every state in the closed vocabulary is always emitted
// (0 when absent) so a dashboard series does not flap as sessions transition; a session
// whose Run token is empty or outside the closed set is bucketed as "unknown" and
// emitted only when nonzero, so registry drift is legible without leaking a count past
// the closed surface.
//
// fak_session_liveness_class (#1204 acceptance) is intentionally NOT emitted here. That
// gauge is fed from #750's two-heartbeat witness, which ships in internal/taskmgr as a
// per-TASK LivenessClass (live|idle|stalled) and is not yet wired onto the gateway's
// per-SESSION plane (#750 P6.1 remains open). Projecting a session-level
// alive/stalled/degraded breakdown ahead of that wiring would fabricate a number; it is
// deferred until #750's witness reaches the session registry.
func (s *Server) writeSessionMetrics(b *strings.Builder) {
	if s.listSessions == nil {
		return
	}
	sessions := s.listSessions(context.Background())
	counts := make(map[string]int, len(sessionRunStates)+1)
	refusals := make(map[string]int, len(sessionEnvelopeRefusalReasonOrder))
	for _, st := range sessions {
		state := strings.ToLower(strings.TrimSpace(st.Run))
		if state == "" {
			state = "unknown"
		}
		counts[state]++
		if sessionEnvelopeRefusalReasons[st.Reason] {
			refusals[st.Reason]++
		}
	}
	writeHelpType(b, "fak_sessions", "Live served sessions by DRIVE run-state token (read-time fold of the session registry; a GC'd session decrements its bucket next scrape).", "gauge")
	known := make(map[string]bool, len(sessionRunStates))
	for _, state := range sessionRunStates {
		fmt.Fprintf(b, "fak_sessions{state=\"%s\"} %d\n", promQuote(state), counts[state])
		known[state] = true
	}
	for state, n := range counts {
		if known[state] || n == 0 {
			continue
		}
		fmt.Fprintf(b, "fak_sessions{state=\"%s\"} %d\n", promQuote(state), n)
	}
	writeHelpType(b, "fak_session_refusals", "Live sessions refused by structured envelope ceiling reason.", "gauge")
	for _, reason := range sessionEnvelopeRefusalReasonOrder {
		fmt.Fprintf(b, "fak_session_refusals{reason=\"%s\"} %d\n", reason, refusals[reason])
	}
}

// writeSpendGovernorMetrics renders the control-plane SPEND-CAP breach counter (#3273):
// each time a scope (tenant/team/agent/session) crossed its versioned budget and the
// kernel hard-paused/killed it, keyed by scope and action. A nil governor (the default
// serve path) suppresses the family entirely; once attached it declares HELP/TYPE and
// emits a row per (scope, action) that has fired, so the panel stays quiet until a real
// breach and never fabricates a phantom zero.
func (s *Server) writeSpendGovernorMetrics(b *strings.Builder) {
	s.admissionMu.RLock()
	g := s.spendGovernor
	s.admissionMu.RUnlock()
	if g == nil {
		return
	}
	rows := g.Snapshot()
	writeHelpType(b, "fak_gateway_spend_breaches_total",
		"Control-plane spend-cap breaches (#3273): each time a scope's cumulative provider spend crossed its versioned budget and the kernel hard-paused/killed the session, by scope (tenant/team/agent/session) and action (pause/kill). Absent until a governor is attached; a scope/action row appears only once it has fired.",
		"counter")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_spend_breaches_total{scope=\"%s\",action=\"%s\"} %d\n",
			promQuote(string(row.Scope)), promQuote(string(row.Action)), row.Count)
	}
}

func (s *Server) renderMetrics() string {
	m := s.metrics
	if m == nil {
		m = newGatewayMetrics(time.Now())
	}
	httpRows, opRows := m.snapshot()
	var b strings.Builder

	writeHelpType(&b, "fak_gateway_up", "Whether the fak gateway process is scrapeable.", "gauge")
	fmt.Fprintln(&b, "fak_gateway_up 1")
	writeHelpType(&b, "fak_gateway_start_time_seconds", "Unix start time of this fak gateway process.", "gauge")
	fmt.Fprintf(&b, "fak_gateway_start_time_seconds %d\n", m.start.Unix())
	s.writeStartupMetrics(&b)
	writeHelpType(&b, "fak_gateway_inflight_requests", "HTTP requests currently executing in the fak gateway.", "gauge")
	fmt.Fprintf(&b, "fak_gateway_inflight_requests %d\n", atomic.LoadInt64(&m.inflight))

	// Live-request visibility: derived from the in-flight registry at scrape time.
	// max_age is the oldest currently-running request's age (0 when idle); it
	// surfaces a slow or wedged request at the next scrape, where the completion-
	// time histograms would show nothing until the request finally returned.
	byRoute, maxAge := m.inflightSnapshot(time.Now())
	writeHelpType(&b, "fak_gateway_inflight_max_age_seconds", "Age of the oldest HTTP request currently in flight (0 when idle).", "gauge")
	fmt.Fprintf(&b, "fak_gateway_inflight_max_age_seconds %s\n", promFloat(maxAge))
	writeHelpType(&b, "fak_gateway_inflight_requests_by_route", "HTTP requests currently executing, by route.", "gauge")
	inflightRoutes := make([]string, 0, len(byRoute))
	for route := range byRoute {
		inflightRoutes = append(inflightRoutes, route)
	}
	sort.Strings(inflightRoutes)
	for _, route := range inflightRoutes {
		fmt.Fprintf(&b, "fak_gateway_inflight_requests_by_route{route=\"%s\"} %d\n", promQuote(route), byRoute[route])
	}

	writeHelpType(&b, "fak_gateway_build_info", "Static fak gateway build and runtime labels.", "gauge")
	fmt.Fprintf(&b, "fak_gateway_build_info{version=\"%s\",engine=\"%s\",model=\"%s\",vdso=\"%s\"} 1\n",
		promQuote(s.version), promQuote(s.engineID), promQuote(s.model), promQuote(strconv.FormatBool(s.k.VDSOEnabled())))

	writeHTTPMetrics(&b, httpRows)

	// Upstream-error visibility (the metric twin of the per-turn FAILED debug line): WHY turns
	// failed this session, by coarse kind.
	m.writeUpstreamErrorMetrics(&b)

	// In-kernel background-loop runtime: per-loop tick/error/panic/restart counters,
	// last-tick gauge, and liveness — the proof the kernel's loops keep progressing.
	s.writeBgloopMetrics(&b)

	writeOperationMetrics(&b, opRows)

	c := s.k.Counters()
	writeCounter(&b, "fak_kernel_submits_total", "Kernel submissions since process start.", c.Submits)
	writeCounter(&b, "fak_kernel_vdso_hits_total", "Kernel submissions served by the vDSO fast path.", c.VDSOHits)
	writeCounter(&b, "fak_kernel_engine_calls_total", "Kernel submissions that reached the engine.", c.EngineCalls)
	writeCounter(&b, "fak_kernel_denies_total", "Kernel submissions denied before execution.", c.Denies)
	writeCounter(&b, "fak_kernel_transforms_total", "Kernel submissions transformed by adjudication.", c.Transforms)
	writeCounter(&b, "fak_kernel_quarantines_total", "Kernel result admissions quarantined by the result-side stack.", c.Quarantines)
	writeCounter(&b, "fak_kernel_result_denies_total", "Kernel result admissions hard-refused by the result-side stack.", c.ResultDenies)
	writeCounter(&b, "fak_kernel_admitted_total", "Kernel result admissions that were accepted or transformed.", c.Admitted)
	// Per-rung decision distribution (issue #693): which adjudication rung actually
	// decided each call, bucketed by (rung, kind, reason). Passive — re-derived off the
	// hot path; a vDSO-served call (no adjudication) lands in rung="vdso". Drill down on
	// one call with `fak preflight --explain`. nil (older construction) suppresses it.
	if s.rungObs != nil {
		writeHelpType(&b, "fak_kernel_decisions_total", "Kernel decisions by winning adjudication rung, verdict kind, and reason (passive; re-derived off the hot path).", "counter")
		for _, row := range s.rungObs.Snapshot() {
			fmt.Fprintf(&b, "fak_kernel_decisions_total{rung=\"%s\",kind=\"%s\",reason=\"%s\"} %d\n",
				promQuote(row.Rung), promQuote(row.Kind), promQuote(row.Reason), row.Count)
		}
		fused := s.rungObs.FusedSnapshot()
		writeCounter(&b, "fak_fused_turns_total", "Turns that emitted at least one classical and one weight-classified operation.", fused.FusedTurns)
		writeCounter(&b, "fak_turns_total", "Turns that emitted at least one classified classical or weight operation.", fused.Turns)
		writeHelpType(&b, "fak_turn_ops_total", "Kernel operations by fused-turn concept family (unknown ops do not enter the fused-turn denominator).", "counter")
		for _, row := range fused.Ops {
			fmt.Fprintf(&b, "fak_turn_ops_total{family=\"%s\"} %d\n", promQuote(row.Family), row.Count)
		}
		writeHelpType(&b, "fak_fused_turn_rate", "Cumulative fused-turn rate: fak_fused_turns_total / fak_turns_total.", "gauge")
		fmt.Fprintf(&b, "fak_fused_turn_rate %s\n", promFloat(fused.Rate))
	}
	writeHelpType(&b, "fak_gateway_vdso_hit_ratio", "Current cumulative vDSO hit ratio over kernel submissions.", "gauge")
	ratio := 0.0
	if c.Submits > 0 {
		ratio = float64(c.VDSOHits) / float64(c.Submits)
	}
	fmt.Fprintf(&b, "fak_gateway_vdso_hit_ratio %s\n", promFloat(ratio))
	writeVDSOMetrics(&b)
	// Unified cache-stream family (fak_cache_*): the per-(plane,tier,kind) fold over
	// the cachemeta.Entry stream, fed live by the vDSO tier-2 cache-event sink wired
	// in New. It sits beside the per-cache fak_vdso_* family above; the snapshot
	// carries its own HELP/TYPE lines, so it concatenates cleanly. A Server without a
	// stream (older construction paths) emits nothing rather than a phantom series.
	if s.cacheStream != nil {
		b.WriteString(s.cacheStream.Snapshot().Prometheus())
	}
	writeBlobMetrics(&b)
	writeKVPrefixMetrics(&b)
	s.writeKVMemoryMetrics(&b)
	s.writeRequestMemoryMetrics(&b)
	m.writeRequestMemoryAggregateMetrics(&b)
	inf := m.writeInferenceMetrics(&b)
	s.writeServingMetrics(&b, inf)
	m.writeHarnessMetrics(&b)  // fak_harness_* — the guard harness's own CPU/mem/IO (epic #2044)
	m.writeLogvaultMetrics(&b) // fak_logvault_* — vault last-capture age/footprint/verify mismatches (#2455)
	s.writeNativePDMetrics(&b) // #28: native prefill/decode role-split telemetry, when a cluster is wired
	if s.nativeReceiptMetrics != nil {
		b.WriteString(s.nativeReceiptMetrics.Prometheus(time.Now()))
	}
	m.writeVCacheMetrics(&b)
	m.writeVCacheWarmthMetrics(&b)
	m.writeVCacheWarmthDemotionMetrics(&b)
	m.writeVCacheGovernorMetrics(&b)
	m.writeInKernelOOMMetrics(&b)
	s.writeInKernelOOMRetryMetrics(&b)
	s.writeInKernelPressureTrimMetrics(&b)
	s.writeMoEResidencyMetrics(&b) // #5617: activated-expert residency for a serve that declared an expert budget
	m.writeCompactionMetrics(&b)
	s.writeToolPageMetrics(&b) // #2440: ctxmmu tool-schema page catalog residency + dedup witnesses
	m.writeResetShadowMetrics(&b)
	m.writeCacheBreakMetrics(&b) // #2916: per-session cache-break events + cold-rebuild token cost, by closed cause
	m.writeDenyAllMetrics(&b)
	s.writeSessionMetrics(&b)           // #1204: live session count by DRIVE run-state token
	s.writeSessionSaturationMetrics(&b) // #3425: deployment session saturation vs the configured ceiling (FAK_MAX_SESSIONS), when armed
	s.writeSpendGovernorMetrics(&b)     // #3273: control-plane spend-cap breaches by scope + action
	m.harnessCoherence.writeHarnessCoherenceMetrics(&b)
	m.writeRoutingMetrics(&b)            // #603: per-aspect model-routing decision distribution (rule/strategy/aspect)
	outputNegframeAudit.writeMetrics(&b) // #3567: negative-framing spans in model OUTPUT prose (sampled shadow, observe-only)
	s.resumeProj.writeMetrics(&b)        // #941: resume projected-vs-observed residual (self-contained family)
	s.observers.writeMetrics(&b)         // #2434: async result-observer lag + auto-disable (self-contained family)
	s.writeFleetMembershipMetrics(&b)    // #42: live fleet membership/health/drain/failover transitions, per worker
	s.writeAdmissionMetrics(&b)          // #35: native serving-scheduler admission family (fak_sched_*), when a controller is wired
	s.writePreemptionMetrics(&b)         // #31: native KV preemption/swap/recompute family, when a preemptor is wired

	// Fleet-value (hero-axis) KPIs, derived live from the kernel counters + the
	// inference accumulators above. fak's product axis is agent-fleet serving
	// efficiency (HERO-BENCHMARK), and these are the per-node ingredients of that
	// headline: turns the kernel saved (engine round-trips + retry turns avoided),
	// context-window pollutions it blocked, and the wall-clock agents were served.
	// agentSeconds is the time spent generating completions plus adjudicating/
	// dispatching kernel operations — the denominator for a live per-second view.
	var opSecs float64
	for _, row := range opRows {
		opSecs += row.val.sum
	}
	servedInline := m.servedInlineSnapshot()
	writeFleetValueMetrics(&b, c, servedInline, inf.decodeSecs+opSecs)
	cacheSavings := m.adjudicationSummary().MechanismSavings()
	if c.VDSOHits > 0 {
		cacheSavings.FakVDSOAvoidedCalls += uint64(c.VDSOHits)
	}
	cacheSavings.FakVDSOAvoidedCalls += servedInline
	writeCacheAttributionMetrics(&b, cacheSavings)

	s.writeModelLoadMetrics(&b)
	fmt.Fprintf(&b, "# HELP fak_traceparent_invalid_total Malformed inbound W3C traceparent headers.\n# TYPE fak_traceparent_invalid_total counter\nfak_traceparent_invalid_total %d\n", atomic.LoadUint64(&s.traceparentInvalid))
	otlp := s.otlp.stats()
	fmt.Fprintf(&b, "# HELP fak_otlp_spans_total OTLP spans by outcome.\n# TYPE fak_otlp_spans_total counter\n")
	fmt.Fprintf(&b, "fak_otlp_spans_total{outcome=\"accepted\"} %d\n", otlp.Accepted)
	fmt.Fprintf(&b, "fak_otlp_spans_total{outcome=\"exported\"} %d\n", otlp.Exported)
	fmt.Fprintf(&b, "fak_otlp_spans_total{outcome=\"dropped\"} %d\n", otlp.Dropped)
	fmt.Fprintf(&b, "fak_otlp_spans_total{outcome=\"failed\"} %d\n", otlp.Failed)
	fmt.Fprintf(&b, "# HELP fak_otlp_queue_depth Current OTLP span queue depth.\n# TYPE fak_otlp_queue_depth gauge\nfak_otlp_queue_depth %d\n", otlp.QueueDepth)
	audit := s.orgAudit.Stats()
	fmt.Fprintf(&b, "# HELP fak_org_audit_receipts_total Organization audit receipts by outcome.\n# TYPE fak_org_audit_receipts_total counter\n")
	fmt.Fprintf(&b, "fak_org_audit_receipts_total{outcome=\"accepted\"} %d\n", audit.Accepted)
	fmt.Fprintf(&b, "fak_org_audit_receipts_total{outcome=\"exported\"} %d\n", audit.Exported)
	fmt.Fprintf(&b, "fak_org_audit_receipts_total{outcome=\"buffered\"} %d\n", audit.Buffered)
	fmt.Fprintf(&b, "fak_org_audit_receipts_total{outcome=\"dropped\"} %d\n", audit.Dropped)
	fmt.Fprintf(&b, "fak_org_audit_receipts_total{outcome=\"failed\"} %d\n", audit.Failed)
	fmt.Fprintf(&b, "# HELP fak_org_audit_queue_depth Current organization audit receipt queue depth.\n# TYPE fak_org_audit_queue_depth gauge\nfak_org_audit_queue_depth %d\n", audit.QueueDepth)
	traj := s.trajctlMetricsSnapshot()
	fmt.Fprintf(&b, "# HELP fak_trajctl_objectives Trajectory objectives by lifecycle status.\n# TYPE fak_trajctl_objectives gauge\n")
	for _, status := range []string{"abandoned", "active", "met", "paused"} {
		fmt.Fprintf(&b, "fak_trajctl_objectives{status=%q} %d\n", status, traj.Objectives[status])
	}
	fmt.Fprintf(&b, "# HELP fak_trajctl_score Mean latest open-objective score by bounded objective kind.\n# TYPE fak_trajctl_score gauge\n")
	for _, kind := range []string{"child", "root", "scorer"} {
		fmt.Fprintf(&b, "fak_trajctl_score{objective_kind=%q} %g\n", kind, traj.Scores[kind])
	}
	fmt.Fprintf(&b, "# HELP fak_trajctl_signals Trajectory objectives by current health signal.\n# TYPE fak_trajctl_signals gauge\n")
	for _, signal := range []string{"DRIFT", "HEALTHY", "STALL"} {
		fmt.Fprintf(&b, "fak_trajctl_signals{signal=%q} %d\n", signal, traj.Signals[signal])
	}
	fmt.Fprintf(&b, "# HELP fak_trajctl_nudges_total Trajectory re-anchor nudges by delivery outcome.\n# TYPE fak_trajctl_nudges_total counter\n")
	for _, outcome := range []string{"delivered", "failed"} {
		fmt.Fprintf(&b, "fak_trajctl_nudges_total{outcome=%q} %d\n", outcome, traj.Nudges[outcome])
	}
	return b.String()
}

func writeHTTPMetrics(b *strings.Builder, httpRows []httpMetricSnapshot) {
	writeHelpType(b, "fak_gateway_http_requests_total", "HTTP requests served by route, method, and status.", "counter")
	for _, row := range httpRows {
		fmt.Fprintf(b, "fak_gateway_http_requests_total{route=\"%s\",method=\"%s\",status=\"%s\"} %d\n",
			promQuote(row.key.route), promQuote(row.key.method), promQuote(row.key.status), row.val.count)
	}
	writeHelpType(b, "fak_gateway_http_request_duration_seconds", "HTTP request latency by route, method, and status.", "histogram")
	for _, row := range httpRows {
		baseLabels := fmt.Sprintf("route=\"%s\",method=\"%s\",status=\"%s\"",
			promQuote(row.key.route), promQuote(row.key.method), promQuote(row.key.status))
		writeHistogram(b, "fak_gateway_http_request_duration_seconds", baseLabels, row.val)
	}
}

func writeOperationMetrics(b *strings.Builder, opRows []operationMetricSnapshot) {
	writeHelpType(b, "fak_gateway_operations_total", "Gateway kernel operations by operation, verdict, and deciding adjudicator (by).", "counter")
	for _, row := range opRows {
		fmt.Fprintf(b, "fak_gateway_operations_total{operation=\"%s\",verdict=\"%s\",reason=\"%s\",disposition=\"%s\",by=\"%s\"} %d\n",
			promQuote(row.key.operation), promQuote(row.key.verdict), promQuote(row.key.reason),
			promQuote(row.key.disposition), promQuote(row.key.by), row.val.count)
	}
	writeHelpType(b, "fak_gateway_operation_duration_seconds", "Gateway kernel operation latency by operation, verdict, and deciding adjudicator (by).", "histogram")
	for _, row := range opRows {
		baseLabels := fmt.Sprintf("operation=\"%s\",verdict=\"%s\",reason=\"%s\",disposition=\"%s\",by=\"%s\"",
			promQuote(row.key.operation), promQuote(row.key.verdict), promQuote(row.key.reason),
			promQuote(row.key.disposition), promQuote(row.key.by))
		writeHistogram(b, "fak_gateway_operation_duration_seconds", baseLabels, row.val)
	}
}

func (s *Server) writeRequestMemoryMetrics(b *strings.Builder) {
	reporter, ok := s.planner.(agent.RequestMemoryReporter)
	if !ok {
		return
	}
	st := reporter.RequestMemoryStats()
	if !st.Observed {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_plan_bytes", "Most recent served in-kernel backend request memory plan, by class/scope/dtype. This is a last-request gauge, not a cumulative counter.", "gauge")
	for _, row := range requestMemoryPlanByClassScopeDType(st.MemoryPlan) {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_plan_bytes{backend=\"%s\",class=\"%s\",scope=\"%s\",dtype=\"%s\"} %d\n",
			promQuote(backend), promQuote(row.Class), promQuote(row.Scope), promQuote(row.DType), row.Bytes)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_tokens", "Token window used by the most recent served in-kernel backend request memory plan.", "gauge")
	fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens{backend=\"%s\",kind=\"prompt\"} %d\n", promQuote(backend), st.PromptTokens)
	fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens{backend=\"%s\",kind=\"max_new\"} %d\n", promQuote(backend), st.MaxNewTokens)
	fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens{backend=\"%s\",kind=\"planned\"} %d\n", promQuote(backend), st.PlannedTokens)
	if st.HeadroomRatio > 0 {
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_headroom_ratio", "Fraction of reported capacity reserved by the most recent in-kernel request fit check for runtime headroom.", "gauge")
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_headroom_ratio{backend=\"%s\"} %s\n", promQuote(backend), promFloat(st.HeadroomRatio))
	}
	if len(st.Capacities) > 0 {
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_capacity_known", "Whether the backend reported capacity for a memory scope used by the most recent in-kernel request fit check.", "gauge")
		for _, cap := range sortedRequestMemoryCapacities(st.Capacities) {
			known := 0
			if cap.Known {
				known = 1
			}
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_known{backend=\"%s\",scope=\"%s\"} %d\n",
				promQuote(backend), promQuote(modelLoadScope(cap.Scope)), known)
		}
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_capacity_free_known", "Whether the backend reported current free bytes for a memory scope used by the most recent in-kernel request fit check.", "gauge")
		for _, cap := range sortedRequestMemoryCapacities(st.Capacities) {
			known := 0
			if cap.Known && cap.FreeKnown {
				known = 1
			}
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_free_known{backend=\"%s\",scope=\"%s\"} %d\n",
				promQuote(backend), promQuote(modelLoadScope(cap.Scope)), known)
		}
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_capacity_bytes", "Reported backend capacity bytes used by the most recent in-kernel request fit check. The free row is omitted when current free bytes are unknown.", "gauge")
		for _, cap := range sortedRequestMemoryCapacities(st.Capacities) {
			if !cap.Known {
				continue
			}
			scope := modelLoadScope(cap.Scope)
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_bytes{backend=\"%s\",scope=\"%s\",kind=\"total\"} %d\n",
				promQuote(backend), promQuote(scope), cap.TotalBytes)
			if cap.FreeKnown {
				fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_bytes{backend=\"%s\",scope=\"%s\",kind=\"free\"} %d\n",
					promQuote(backend), promQuote(scope), cap.FreeBytes)
			}
		}
	}
	if rows := requestMemoryFitRows(st.MemoryPlan, st.Capacities, st.HeadroomRatio); len(rows) > 0 {
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_fit_bytes", "Headroom-adjusted fit summary for the most recent in-kernel request by backend and scope. kind=want is planned bytes; kind=budget and kind=margin are omitted when capacity is unknown.", "gauge")
		for _, row := range rows {
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_bytes{backend=\"%s\",scope=\"%s\",kind=\"want\"} %d\n",
				promQuote(backend), promQuote(row.Scope), row.WantBytes)
			if !row.CapacityKnown {
				continue
			}
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_bytes{backend=\"%s\",scope=\"%s\",kind=\"budget\"} %d\n",
				promQuote(backend), promQuote(row.Scope), row.BudgetBytes)
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_bytes{backend=\"%s\",scope=\"%s\",kind=\"margin\"} %d\n",
				promQuote(backend), promQuote(row.Scope), row.MarginBytes)
		}
	}
}

func (m *gatewayMetrics) writeRequestMemoryAggregateMetrics(b *strings.Builder) {
	snap := m.requestMemoryAggregateSnapshotData()
	if len(snap.observed) == 0 {
		return
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_observations_total", "In-kernel backend request memory plans observed after served planner turns, by backend. Includes successful turns and local OOM/capacity refusals that produced a plan.", "counter")
	backends := make([]string, 0, len(snap.observed))
	for backend := range snap.observed {
		backends = append(backends, backend)
	}
	sort.Strings(backends)
	for _, backend := range backends {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_observations_total{backend=\"%s\"} %d\n",
			promQuote(backend), snap.observed[backend])
	}
	// The plan / token / fit families each publish several metrics over the SAME rows,
	// so each family renders its label set once here. Building the labels in one place
	// is what keeps the members of a family label-identical — a Prometheus consumer that
	// joins plan_bytes_total against plan_observations_total needs exactly that.
	planLabels := func(row requestMemoryPlanSnapshot) string {
		k := row.key
		return fmt.Sprintf("backend=\"%s\",class=\"%s\",scope=\"%s\",dtype=\"%s\"",
			promQuote(k.backend), promQuote(k.class), promQuote(k.scope), promQuote(k.dtype))
	}
	tokenLabels := func(row requestMemoryTokenSnapshot) string {
		k := row.key
		return fmt.Sprintf("backend=\"%s\",kind=\"%s\"", promQuote(k.backend), promQuote(k.kind))
	}
	fitLabels := func(row requestMemoryFitSnapshot) string {
		k := row.key
		return fmt.Sprintf("backend=\"%s\",scope=\"%s\"", promQuote(k.backend), promQuote(k.scope))
	}

	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_plan_observations_total", "Observed in-kernel request memory plan rows by backend, class, scope, and dtype.", "counter",
		snap.plans, planLabels, func(r requestMemoryPlanSnapshot) any { return r.observations })
	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_plan_bytes_total", "Cumulative planned bytes observed for served in-kernel backend requests, by backend, class, scope, and dtype.", "counter",
		snap.plans, planLabels, func(r requestMemoryPlanSnapshot) any { return r.totalBytes })
	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_plan_high_water_bytes", "Largest single observed in-kernel request memory plan row, by backend, class, scope, and dtype.", "gauge",
		snap.plans, planLabels, func(r requestMemoryPlanSnapshot) any { return r.highWaterBytes })
	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_tokens_total", "Cumulative prompt/max_new/planned token windows from observed in-kernel request memory plans.", "counter",
		snap.tokens, tokenLabels, func(r requestMemoryTokenSnapshot) any { return r.total })
	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_tokens_high_water", "Largest prompt/max_new/planned token window from an observed in-kernel request memory plan.", "gauge",
		snap.tokens, tokenLabels, func(r requestMemoryTokenSnapshot) any { return r.highWater })
	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_fit_observations_total", "Observed in-kernel request memory fit rows by backend and scope.", "counter",
		snap.fits, fitLabels, func(r requestMemoryFitSnapshot) any { return r.observations })
	writeKeyedFamily(b, "fak_gateway_in_kernel_request_memory_fit_want_high_water_bytes", "Largest observed planned in-kernel request memory bytes by backend and scope.", "gauge",
		snap.fits, fitLabels, func(r requestMemoryFitSnapshot) any { return r.wantHighWater })
	// The margin family is NOT a writeKeyedFamily call: a scope whose capacity is unknown
	// has no margin at all, and publishing 0 for it would read as a measured zero headroom.
	// Skipping the row is the distinction, so it keeps its own loop.
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_fit_margin_low_water_bytes", "Smallest observed headroom-adjusted fit margin for known-capacity in-kernel requests, by backend and scope. Omitted for scopes whose capacity was unknown.", "gauge")
	for _, row := range snap.fits {
		if !row.marginKnown {
			continue
		}
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_margin_low_water_bytes{%s} %d\n", fitLabels(row), row.marginLowWater)
	}
}

func (s *Server) writeInKernelOOMRetryMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.InKernelOOMRetryReporter)
	if !ok {
		return
	}
	st := reporter.InKernelOOMRetryStats()
	if len(st.Rows) == 0 {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	writeHelpType(b, "fak_gateway_in_kernel_oom_retry_total", "Idle-pool trim retries attempted after local in-kernel device allocation OOMs, bucketed by backend, memory class, and outcome. These are decode retries only; capacity precheck refusals do not retry.", "counter")
	for _, row := range st.Rows {
		class := oomClassLabel(row.Class)
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_total{backend=\"%s\",class=\"%s\",outcome=\"attempted\"} %d\n",
			promQuote(backend), promQuote(class), row.Attempts)
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_total{backend=\"%s\",class=\"%s\",outcome=\"succeeded\"} %d\n",
			promQuote(backend), promQuote(class), row.Successes)
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_total{backend=\"%s\",class=\"%s\",outcome=\"failed\"} %d\n",
			promQuote(backend), promQuote(class), row.Failures)
	}
	writeHelpType(b, "fak_gateway_in_kernel_oom_retry_last_failed_bytes", "Most recent allocation size that triggered an idle-pool trim retry for each backend and memory class.", "gauge")
	for _, row := range st.Rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_last_failed_bytes{backend=\"%s\",class=\"%s\"} %d\n",
			promQuote(backend), promQuote(oomClassLabel(row.Class)), row.LastFailedBytes)
	}
}

// writeMoEResidencyMetrics emits the activated-expert residency family (R6, #5617): what a serve
// that declared an expert budget actually paid to keep the top-k experts resident.
//
// Two silences are deliberate. A proxy planner does not implement the reporter, and a local planner
// whose operator declared no budget never builds a ring — both emit nothing at all, because a
// scrape of `fak_gateway_moe_expert_staging_total 0` is indistinguishable from a ring that ran and
// missed everything, and the whole point of this family is telling those apart.
//
// The series are the raw counters plus the gauges that are NOT recoverable from them. Hit rate,
// refusal rate and bytes-per-token are deliberately absent: each is a ratio of two counters here,
// so PromQL derives them at the resolution the operator asks for rather than at whatever window
// this process happened to accumulate. Budget and peak cannot be derived, and the placement/shape
// gauges describe the most recent request only — the same last-request framing the request-memory
// plan above uses, because drift against a pin set does not sum across requests.
func (s *Server) writeMoEResidencyMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.MoEResidencyReporter)
	if !ok {
		return
	}
	l := reporter.MoEResidencyStats()
	if l.Requests == 0 {
		return // no request ever engaged a routed-expert ring: not measured, not "measured as zero"
	}
	writeHelpType(b, "fak_gateway_moe_expert_staging_total", "Routed-expert weight stagings served by the local activated-expert ring, split by outcome. A refusal is a staging no budget could admit, which falls back to permanent residency and means the declared budget stopped bounding anything.", "counter")
	fmt.Fprintf(b, "fak_gateway_moe_expert_staging_total{outcome=\"hit\"} %d\n", l.Hits)
	fmt.Fprintf(b, "fak_gateway_moe_expert_staging_total{outcome=\"page_in\"} %d\n", l.PageIns)
	fmt.Fprintf(b, "fak_gateway_moe_expert_staging_total{outcome=\"refused\"} %d\n", l.Refusals)
	writeHelpType(b, "fak_gateway_moe_expert_evictions_total", "Routed-expert weights evicted from the local activated-expert ring to stay inside the declared budget.", "counter")
	fmt.Fprintf(b, "fak_gateway_moe_expert_evictions_total %d\n", l.Evictions)
	writeHelpType(b, "fak_gateway_moe_expert_page_in_bytes_total", "Device bytes moved by cold routed-expert page-ins. Divided by fak_gateway_moe_residency_tokens_total this is the expert bytes each forwarded token cost, which is the number an expert budget is sized against.", "counter")
	fmt.Fprintf(b, "fak_gateway_moe_expert_page_in_bytes_total %d\n", l.PageInBytes)
	writeHelpType(b, "fak_gateway_moe_residency_requests_total", "Completed local requests that engaged a routed-expert ring.", "counter")
	fmt.Fprintf(b, "fak_gateway_moe_residency_requests_total %d\n", l.Requests)
	writeHelpType(b, "fak_gateway_moe_residency_tokens_total", "Tokens those requests actually forwarded through the model. Prompt tokens served from the prefix cache are excluded: they activated no expert, and counting them would make the byte rates read cheaper than they are.", "counter")
	fmt.Fprintf(b, "fak_gateway_moe_residency_tokens_total %d\n", l.Tokens)
	writeHelpType(b, "fak_gateway_moe_residency_reconciliation_failures_total", "Requests whose own residency report failed its internal identity checks. Should be 0 forever; any increase means the ring's accounting disagreed with itself and every series in this family is suspect.", "counter")
	fmt.Fprintf(b, "fak_gateway_moe_residency_reconciliation_failures_total %d\n", l.ReconciliationFailures)
	writeHelpType(b, "fak_gateway_moe_expert_budget_bytes", "Declared device-byte ceiling for resident routed experts, as most recently observed.", "gauge")
	fmt.Fprintf(b, "fak_gateway_moe_expert_budget_bytes %d\n", l.BudgetBytes)
	writeHelpType(b, "fak_gateway_moe_expert_peak_resident_bytes", "High-water resident routed-expert footprint across every request. Staying well under the budget means the surplus could be given back to KV.", "gauge")
	fmt.Fprintf(b, "fak_gateway_moe_expert_peak_resident_bytes %d\n", l.PeakBytes)

	last := l.Last
	if last.Shape.Experts > 0 {
		writeHelpType(b, "fak_gateway_moe_activated_fraction", "Fraction of the model's experts a single token routes to (k/E) — the gap between stored and activated bytes this ladder exists to exploit.", "gauge")
		fmt.Fprintf(b, "fak_gateway_moe_activated_fraction %s\n", promFloat(last.Shape.ActivatedFraction))
	}
	if basis := strings.TrimSpace(last.Placement.Basis); basis != "" && basis != "none" {
		writeHelpType(b, "fak_gateway_moe_placement_drift", "Share of the most recent request's resident-expert plan that the request never routed to. This is a last-request gauge, not a cumulative counter: drift against a pin set does not sum across requests.", "gauge")
		fmt.Fprintf(b, "fak_gateway_moe_placement_drift{basis=\"%s\"} %s\n", promQuote(basis), promFloat(last.Placement.Drift))
		writeHelpType(b, "fak_gateway_moe_placement_served_share", "Share of the most recent request's expert touches that the resident plan actually served — the complement of drift. Last-request gauge.", "gauge")
		fmt.Fprintf(b, "fak_gateway_moe_placement_served_share{basis=\"%s\"} %s\n", promQuote(basis), promFloat(last.Placement.Coverage))
	}
	if last.Shared != nil && last.Shared.Agents > 0 {
		writeHelpType(b, "fak_gateway_moe_shared_ring_agents", "Sessions attached to the shared activated-expert ring during the most recent request. Last-request gauge.", "gauge")
		fmt.Fprintf(b, "fak_gateway_moe_shared_ring_agents %d\n", last.Shared.Agents)
		writeHelpType(b, "fak_gateway_moe_agents_per_page_in", "Sessions served by the average cold page-in under the shared ring. At 1.0 every page-in served only the agent that paid for it, which is N private rings wearing one name. Last-request gauge.", "gauge")
		fmt.Fprintf(b, "fak_gateway_moe_agents_per_page_in %s\n", promFloat(last.Rates.AgentsPerPageIn))
	}
}

func (s *Server) writeInKernelPressureTrimMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.InKernelMemoryPressureTrimReporter)
	if !ok {
		return
	}
	st := reporter.InKernelMemoryPressureTrimStats()
	if len(st.Rows) == 0 {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	writeHelpType(b, "fak_gateway_in_kernel_memory_pressure_trim_total", "Idle-pool trims attempted before local in-kernel decode when a known request memory plan is refused or close to the headroom-adjusted budget. resolved means a capacity-precheck refusal fit after trimming.", "counter")
	for _, row := range st.Rows {
		scope := modelLoadScope(row.Scope)
		class := oomClassLabel(row.Class)
		reason := pressureTrimReasonLabel(row.Reason)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"attempted\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.Attempts)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"trimmed\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.Trimmed)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"no_hooks\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.NoHooks)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"resolved\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.Resolved)
	}
	writeHelpType(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes", "Most recent request memory pressure trim sizing by backend, scope, class, and reason. kind=margin may be negative for a refused precheck.", "gauge")
	for _, row := range st.Rows {
		scope := modelLoadScope(row.Scope)
		class := oomClassLabel(row.Class)
		reason := pressureTrimReasonLabel(row.Reason)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",kind=\"want\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.LastWantBytes)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",kind=\"budget\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.LastBudgetBytes)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",kind=\"margin\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.LastMarginBytes)
	}
}

func pressureTrimReasonLabel(reason string) string {
	reason = strings.TrimSpace(reason)
	switch reason {
	case "capacity_precheck", "low_margin":
		return reason
	case "":
		return "unknown"
	default:
		return "other"
	}
}

func requestMemoryPlanByClassScopeDType(plan []agent.RequestMemoryDemand) []agent.RequestMemoryDemand {
	type key struct {
		class string
		scope string
		dtype string
	}
	by := map[key]int64{}
	for _, row := range plan {
		if row.Bytes <= 0 {
			continue
		}
		k := key{class: modelLoadClass(row.Class), scope: modelLoadScope(row.Scope), dtype: modelLoadDType(row.DType)}
		by[k] += row.Bytes
	}
	out := make([]agent.RequestMemoryDemand, 0, len(by))
	for k, bytes := range by {
		out = append(out, agent.RequestMemoryDemand{Class: k.class, Scope: k.scope, DType: k.dtype, Bytes: bytes})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Scope != out[j].Scope {
			return out[i].Scope < out[j].Scope
		}
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return out[i].DType < out[j].DType
	})
	return out
}

func sortedRequestMemoryCapacities(in []agent.RequestMemoryCapacity) []agent.RequestMemoryCapacity {
	out := append([]agent.RequestMemoryCapacity(nil), in...)
	sort.SliceStable(out, func(i, j int) bool { return modelLoadScope(out[i].Scope) < modelLoadScope(out[j].Scope) })
	return out
}

// writeVDSOMetrics renders vDSO fast-path effectiveness from the live vdso.Default
// stats API — the SAME process-global instance the kernel consults on the fast path
// and the coherence feed subscribes to. The kernel's fak_kernel_vdso_hits_total above
// counts submissions the fast path served; these gauges instead report the cache's OWN
// view: how often a lookup hit, how many entries it has filled, and how often a write
// stranded cached reads.
//
// Two of the issue's three asks are rendered from a direct accessor; two are rendered
// from the nearest exported signal because the vDSO does not expose them:
//   - hit rate    -> Stats() hitRate, a direct accessor.
//   - entry count -> Stats() fills is the CUMULATIVE number of entries stored. The vDSO
//     does not export the live cache occupancy (len of the tier-2 map), so this is the
//     fills counter, not a current-size gauge.
//   - eviction rate -> the vDSO does not export a per-entry LRU-eviction counter. The
//     nearest exported invalidation signal is Mutations(): write-shaped completions
//     that strand cached reads by bumping the world/scope epoch. Reported as such.
func writeVDSOMetrics(b *strings.Builder) {
	lookups, hits, fills, hitRate := vdso.Default.Stats()
	writeHelpType(b, "fak_vdso_lookups_total", "vDSO fast-path lookups attempted (tier-1/2/3 consulted).", "counter")
	fmt.Fprintf(b, "fak_vdso_lookups_total %d\n", lookups)
	writeHelpType(b, "fak_vdso_hits_total", "vDSO fast-path lookups served locally (a hit at any tier).", "counter")
	fmt.Fprintf(b, "fak_vdso_hits_total %d\n", hits)
	writeHelpType(b, "fak_vdso_hit_rate", "vDSO lookup hit rate (hits/lookups) from the live vdso.Default stats API.", "gauge")
	fmt.Fprintf(b, "fak_vdso_hit_rate %s\n", promFloat(hitRate))
	writeHelpType(b, "fak_vdso_cache_fills_total", "vDSO tier-2 cache entries filled since start (cumulative; the vDSO exports no live occupancy).", "counter")
	fmt.Fprintf(b, "fak_vdso_cache_fills_total %d\n", fills)
	writeHelpType(b, "fak_vdso_invalidations_total", "Write-shaped completions that stranded cached reads (the vDSO exports no per-entry LRU-eviction counter; this is the nearest invalidation signal).", "counter")
	fmt.Fprintf(b, "fak_vdso_invalidations_total %d\n", vdso.Default.Mutations())

	// Miss attribution: every lookup that returned no local answer, by reason — so a
	// low hit rate is explainable (write-shaped tools vs missing readOnly/idempotent
	// hints vs genuine cache churn) instead of collapsing to a bare miss. This is the
	// aggregate complement to the per-call decision trace (fak preflight --explain).
	writeHelpType(b, "fak_vdso_misses_total", "vDSO fast-path lookups that returned no local answer, by reason (DESTRUCTIVE: write-shaped tool, never cacheable; MISSING_HINTS: no readOnly/idempotent hint; RESOURCE_MISNAMED: read cannot name its entity; WITNESS_REVOKED: entry's external witness was refuted; NOT_CACHED: cacheable but unfilled or epoch-stranded).", "counter")
	misses := vdso.Default.MissReasons()
	missReasons := make([]string, 0, len(misses))
	for r := range misses {
		missReasons = append(missReasons, r)
	}
	sort.Strings(missReasons)
	for _, r := range missReasons {
		fmt.Fprintf(b, "fak_vdso_misses_total{reason=\"%s\"} %d\n", promQuote(r), misses[r])
	}

	// Emission drops: a tier-2 key that fails FromVDSOKey parsing silently shrinks
	// the cache-event stream unless this is watched (#1939).
	writeHelpType(b, "fak_vdso_cachemeta_emit_dropped_total", "cachemeta cache-event emissions dropped because the tier-2 key failed to parse (FromVDSOKey error), by reason.", "counter")
	fmt.Fprintf(b, "fak_vdso_cachemeta_emit_dropped_total{reason=\"key_parse\"} %d\n", vdso.Default.EmitDropped())
}

// writeBlobMetrics renders the content-addressed blob store (internal/blob) — the ONE
// CAS the vDSO tier-2 cache AND the context-MMU page-out share, so it is the
// cross-cache footprint/dedup/eviction surface a level below the per-cache families
// above. The store kept concurrency-safe KPI taps (Stats/Resident/MaxBytes) but
// emitted no metrics; this lights them up so an operator can see resident footprint,
// content-dedup effectiveness, and whether the byte bound is actually evicting — a
// rising fak_blob_evicted_total while the resident gauges plateau is the
// leak-absorbed-by-the-bound signal the store's own doc comment calls out.
func writeBlobMetrics(b *strings.Builder) {
	puts, dedupHits, resolves := blob.Default.Stats()
	residentBlobs, residentBytes, evicted := blob.Default.Resident()

	writeCounter(b, "fak_blob_puts_total", "Payloads stored into the content-addressed blob store (CAS puts; small inline payloads never reach the store and are not counted).", puts)
	writeCounter(b, "fak_blob_dedup_hits_total", "CAS puts whose digest was already resident — content-addressed dedup, the byte stored once and shared by the vDSO cache and the context-MMU.", dedupHits)
	writeCounter(b, "fak_blob_resolves_total", "Blob materializations (Resolve) served from the CAS.", resolves)
	writeHelpType(b, "fak_blob_resident_blobs", "Distinct blobs currently resident in the shared CAS.", "gauge")
	fmt.Fprintf(b, "fak_blob_resident_blobs %d\n", residentBlobs)
	writeHelpType(b, "fak_blob_resident_bytes", "Total bytes currently resident in the shared CAS (the live footprint a leak/pressure alarm watches).", "gauge")
	fmt.Fprintf(b, "fak_blob_resident_bytes %d\n", residentBytes)
	writeCounter(b, "fak_blob_evicted_total", "Digests dropped by the CAS byte bound (only ever UNPINNED transient payloads; a rising count is real working pressure or a leak the bound is absorbing).", evicted)
	writeHelpType(b, "fak_blob_max_bytes", "Configured resident-footprint ceiling for the CAS in bytes (0 = unbounded).", "gauge")
	fmt.Fprintf(b, "fak_blob_max_bytes %d\n", blob.Default.MaxBytes())
	writeHelpType(b, "fak_blob_dedup_ratio", "Fraction of CAS puts served by content dedup (dedup_hits/puts; 0 when nothing has been put).", "gauge")
	ratio := 0.0
	if puts > 0 {
		ratio = float64(dedupHits) / float64(puts)
	}
	fmt.Fprintf(b, "fak_blob_dedup_ratio %s\n", promFloat(ratio))
}

// writeKVPrefixMetrics renders the in-kernel KV-prefix reuse family from the process-global
// cacheobs.Default tap (fed by the planner on every served in-kernel turn). It is the LIVE
// measurement of the frozen-trajectory cache cliff (docs/explainers/frozen-trajectory-cache-cliff.md):
// the reuse ratio is the realized cache-hit, and the per-regime turn buckets show when turns
// leave the frozen append-only regime. This is the LOCAL-KV analogue of the provider-side
// fak_gateway_inference_cached_prompt_* family — distinct signal (the in-kernel RadixAttention
// prefix match, not a remote provider's cache_read), so the two never double-count. On a pure
// proxy workload (no in-kernel model) the tap is never fed and these series stay 0.
func writeKVPrefixMetrics(b *strings.Builder) {
	s := cacheobs.Default.Snapshot()
	writeHelpType(b, "fak_gateway_kv_prefix_tier_accesses_rejected_total",
		"Invalid KV-prefix tier observations rejected before attribution because a tier, operation, outcome, or backend fell outside its closed vocabulary.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_accesses_rejected_total %d\n", s.RejectedTierAccesses)
	writeCounter(b, "fak_gateway_kv_prefix_turns_total",
		"In-kernel model turns observed for KV-prefix reuse (the planner reported a prompt-token count).", int64(s.Turns))
	writeCounter(b, "fak_gateway_kv_prefix_prompt_tokens_total",
		"Prompt (prefill) tokens summed across in-kernel turns — the denominator of the realized cache-hit.", int64(s.PromptTokens))
	writeCounter(b, "fak_gateway_kv_prefix_reused_tokens_total",
		"Prompt tokens served from the cached KV prefix (the RadixAttention match) across in-kernel turns — the prefill work the kernel did NOT redo. Distinct from the provider's cache_read (fak_gateway_inference_cached_prompt_tokens_total).", int64(s.ReusedTokens))
	writeHelpType(b, "fak_gateway_kv_prefix_turns_by_regime_total",
		"In-kernel turns by reuse regime — the live cliff distribution. frozen: reuse >= 0.90 (the append-only ceiling); partial: 0.10-0.90; cold: < 0.10 (a cold first prefill, or a head-mutated / fanned-out turn that left the frozen single-linear regime — see docs/explainers/frozen-trajectory-cache-cliff.md).", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_regime_total{regime=\"frozen\"} %d\n", s.FrozenTurns)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_regime_total{regime=\"partial\"} %d\n", s.PartialTurns)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_regime_total{regime=\"cold\"} %d\n", s.ColdTurns)
	writeHelpType(b, "fak_gateway_kv_prefix_reuse_ratio",
		"Realized in-kernel KV-prefix cache-hit: reused / prompt tokens across served turns (0 until the first turn). A single append-only agent climbs toward ~1 (the frozen ceiling); flexibility, cold fan-out, or a divergent prefix drives it down — the frozen-trajectory cache cliff, measured live.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_reuse_ratio %s\n", promFloat(s.ReuseRatio))
	// The provenance split (#3896, vLLM's by_source axis): the SAME prompt tokens decomposed by
	// WHERE each was served from, orthogonal to the reuse-depth family above. The three buckets
	// sum to the by-source prompt tokens (parts == total). external_kv_transfer stays 0 until a
	// remote / disaggregated KV tier feeds the planner — the live witness that disaggregation is
	// not yet wired, and the exact series that lights up the moment it is.
	src := cacheobs.Default.SourceSnapshot()
	writeHelpType(b, "fak_gateway_kv_prefix_prompt_tokens_by_source_total",
		"In-kernel prompt tokens by PROVENANCE (#3896). source=\"local_compute\": recomputed here (a miss); source=\"local_cache_hit\": served from a locally-resident KV prefix (the RadixAttention match); source=\"external_kv_transfer\": pulled across the fabric from the external / disaggregated KV tier. Sums to the by-source prompt tokens; external_kv_transfer is 0 until a remote tier feeds the planner. Orthogonal to fak_gateway_kv_prefix_reused_tokens_total (which splits by depth, not source).", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_prompt_tokens_by_source_total{source=\"local_compute\"} %d\n", src.LocalComputeTokens)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_prompt_tokens_by_source_total{source=\"local_cache_hit\"} %d\n", src.LocalHitTokens)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_prompt_tokens_by_source_total{source=\"external_kv_transfer\"} %d\n", src.ExternalTransferTokens)
}

// writeCacheAttributionMetrics renders the owner/mechanism split for cache-like value. The
// token-equivalent series are gauges because the provider write-premium mechanism can be
// negative until later reads repay it. VDSO is deliberately a separate avoided-call counter:
// the current witness is a skipped engine round-trip, not a prompt-token amount.
func writeCacheAttributionMetrics(b *strings.Builder, s MechanismSavings) {
	writeHelpType(b, "fak_cache_saved_by_owner", "Cache-like token-equivalent saving by owner. owner=\"provider\" is OBSERVED provider prompt-cache net (read rebate minus write premium); owner=\"fak\" is WITNESSED fak-authored token savings (compaction shed + in-kernel KV-prefix reuse).", "gauge")
	fmt.Fprintf(b, "fak_cache_saved_by_owner{owner=\"provider\"} %s\n", promFloat(s.ProviderTokenEquiv()))
	fmt.Fprintf(b, "fak_cache_saved_by_owner{owner=\"fak\"} %s\n", promFloat(s.FakTokenEquiv()))

	writeHelpType(b, "fak_cache_saved_by_mechanism", "Cache-like token-equivalent saving by owner/mechanism. provider_prompt_cache_write_premium is negative when cache writes have not yet been repaid; fak_vdso is not included here because its current witness is avoided calls, emitted separately.", "gauge")
	fmt.Fprintf(b, "fak_cache_saved_by_mechanism{owner=\"provider\",mechanism=\"provider_prompt_cache_read\"} %s\n", promFloat(s.ProviderPromptCacheReadTokenEquiv))
	fmt.Fprintf(b, "fak_cache_saved_by_mechanism{owner=\"provider\",mechanism=\"provider_prompt_cache_write_premium\"} %s\n", promFloat(s.ProviderPromptCacheWritePremiumTokenEquiv))
	fmt.Fprintf(b, "fak_cache_saved_by_mechanism{owner=\"fak\",mechanism=\"compaction_shed\"} %d\n", s.FakCompactionShedTokens)
	fmt.Fprintf(b, "fak_cache_saved_by_mechanism{owner=\"fak\",mechanism=\"kv_prefix_reuse\"} %d\n", s.FakKVPrefixReusedTokens)

	writeHelpType(b, "fak_cache_saved_token_equiv_by_owner", "Cache-like token-equivalent saving by owner. owner=\"provider\" is OBSERVED provider prompt-cache net (read rebate minus write premium); owner=\"fak\" is WITNESSED fak-authored token savings (compaction shed + in-kernel KV-prefix reuse).", "gauge")
	fmt.Fprintf(b, "fak_cache_saved_token_equiv_by_owner{owner=\"provider\"} %s\n", promFloat(s.ProviderTokenEquiv()))
	fmt.Fprintf(b, "fak_cache_saved_token_equiv_by_owner{owner=\"fak\"} %s\n", promFloat(s.FakTokenEquiv()))

	writeHelpType(b, "fak_cache_saved_token_equiv_by_mechanism", "Cache-like token-equivalent saving by mechanism. provider_prompt_cache_write_premium is negative when cache writes have not yet been repaid; fak_vdso is not included here because its current witness is avoided calls, emitted separately.", "gauge")
	fmt.Fprintf(b, "fak_cache_saved_token_equiv_by_mechanism{owner=\"provider\",mechanism=\"provider_prompt_cache_read\"} %s\n", promFloat(s.ProviderPromptCacheReadTokenEquiv))
	fmt.Fprintf(b, "fak_cache_saved_token_equiv_by_mechanism{owner=\"provider\",mechanism=\"provider_prompt_cache_write_premium\"} %s\n", promFloat(s.ProviderPromptCacheWritePremiumTokenEquiv))
	fmt.Fprintf(b, "fak_cache_saved_token_equiv_by_mechanism{owner=\"fak\",mechanism=\"compaction_shed\"} %d\n", s.FakCompactionShedTokens)
	fmt.Fprintf(b, "fak_cache_saved_token_equiv_by_mechanism{owner=\"fak\",mechanism=\"kv_prefix_reuse\"} %d\n", s.FakKVPrefixReusedTokens)

	writeHelpType(b, "fak_cache_avoided_calls_by_mechanism_total", "Cache-like avoided engine calls by fak-authored mechanism. VDSO is counted here, not in token-equivalent gauges, because the live witness is skipped calls rather than prompt tokens.", "counter")
	fmt.Fprintf(b, "fak_cache_avoided_calls_by_mechanism_total{owner=\"fak\",mechanism=\"vdso\"} %d\n", s.FakVDSOAvoidedCalls)
}

func (s *Server) writeKVMemoryMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.KVMemoryReporter)
	if !ok {
		return
	}
	st := reporter.KVMemoryStats()
	class := strings.TrimSpace(st.MemoryClass)
	if class == "" {
		class = "kv_cache"
	}
	scope := strings.TrimSpace(st.Scope)
	if scope == "" {
		scope = "host"
	}
	backend := defaultBackendLabel(st.Backend)
	labels := fmt.Sprintf("class=\"%s\",scope=\"%s\",backend=\"%s\"", promQuote(class), promQuote(scope), promQuote(backend))
	dtype := modelLoadDType(st.DType)
	enabled := 0
	if st.Enabled {
		enabled = 1
	}
	writeHelpType(b, "fak_gateway_kv_memory_enabled", "Whether a local reusable KV prefix cache is active for this planner. Proxy/mock planners emit no resident-KV series.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_enabled{%s} %d\n", labels, enabled)
	writeHelpType(b, "fak_gateway_kv_memory_dtype_info", "Storage dtype for local KV-cache rows under this planner/backend. Current HAL KV rows are f32; proxy/mock planners emit no resident-KV series.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_dtype_info{%s,dtype=\"%s\"} 1\n", labels, promQuote(dtype))
	writeHelpType(b, "fak_gateway_kv_memory_bytes_per_token", "Estimated bytes for one resident KV position under this model layout (classed as kv_cache).", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_bytes_per_token{%s} %d\n", labels, st.BytesPerToken)
	if st.HeadroomRatio > 0 {
		writeHelpType(b, "fak_gateway_kv_memory_headroom_ratio", "Fraction of reported capacity reserved when calculating resident KV-cache fit budget.", "gauge")
		fmt.Fprintf(b, "fak_gateway_kv_memory_headroom_ratio{%s} %s\n", labels, promFloat(st.HeadroomRatio))
	}
	known := 0
	if st.CapacityKnown {
		known = 1
	}
	writeHelpType(b, "fak_gateway_kv_memory_capacity_known", "Whether the planner reported capacity for the resident KV-cache memory scope.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_known{%s} %d\n", labels, known)
	freeKnown := 0
	if st.CapacityKnown && st.CapacityFreeKnown {
		freeKnown = 1
	}
	writeHelpType(b, "fak_gateway_kv_memory_capacity_free_known", "Whether the planner reported current free bytes for the resident KV-cache memory scope.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_free_known{%s} %d\n", labels, freeKnown)
	if st.CapacityKnown {
		writeHelpType(b, "fak_gateway_kv_memory_capacity_bytes", "Reported capacity bytes for the resident KV-cache memory scope. The free row is omitted when current free bytes are unknown.", "gauge")
		fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_bytes{%s,kind=\"total\"} %d\n", labels, st.CapacityTotalBytes)
		if st.CapacityFreeKnown {
			fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_bytes{%s,kind=\"free\"} %d\n", labels, st.CapacityFreeBytes)
		}
	}
	writeHelpType(b, "fak_gateway_kv_memory_fit_bytes", "Headroom-adjusted fit summary for resident KV-cache memory. kind=want is current resident bytes; kind=budget and kind=margin are omitted when capacity is unknown.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_fit_bytes{%s,kind=\"want\"} %d\n", labels, st.ResidentBytes)
	if st.CapacityKnown {
		fmt.Fprintf(b, "fak_gateway_kv_memory_fit_bytes{%s,kind=\"budget\"} %d\n", labels, st.FitBudgetBytes)
		fmt.Fprintf(b, "fak_gateway_kv_memory_fit_bytes{%s,kind=\"margin\"} %d\n", labels, st.FitMarginBytes)
	}
	if !st.Enabled {
		return
	}
	writeHelpType(b, "fak_gateway_kv_memory_resident_tokens", "True resident KV prefix positions held by the local prefix cache. This can exceed the LRU edge-token budget on deep radix chains.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_resident_tokens{%s} %d\n", labels, st.ResidentTokens)
	writeHelpType(b, "fak_gateway_kv_memory_resident_bytes", "Estimated resident local KV-cache bytes, derived from resident prefix positions and model KV geometry.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_resident_bytes{%s} %d\n", labels, st.ResidentBytes)
	writeHelpType(b, "fak_gateway_kv_memory_lru_tokens", "Radix KV edge-token count enforced by the LRU budget. This is not the same as true resident prefix positions.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_lru_tokens{%s} %d\n", labels, st.LRUTokens)
	writeHelpType(b, "fak_gateway_kv_memory_budget_tokens", "Configured radix KV LRU edge-token budget (0 means unbounded).", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_budget_tokens{%s} %d\n", labels, st.BudgetTokens)
	writeHelpType(b, "fak_gateway_kv_memory_max_depth_tokens", "Longest cached KV prefix depth in tokens.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_max_depth_tokens{%s} %d\n", labels, st.MaxDepthTokens)
	writeHelpType(b, "fak_gateway_kv_memory_nodes", "Radix KV prefix-cache nodes currently resident.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_nodes{%s} %d\n", labels, st.Nodes)
	writeHelpType(b, "fak_gateway_kv_memory_leaves", "Radix KV prefix-cache leaves currently resident.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_leaves{%s} %d\n", labels, st.Leaves)
	writeHelpType(b, "fak_gateway_kv_memory_evictions_total", "Radix KV prefix-cache evictions by cause: lru pressure or policy/quarantine.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_memory_evictions_total{%s,kind=\"lru\"} %d\n", labels, st.Evictions)
	fmt.Fprintf(b, "fak_gateway_kv_memory_evictions_total{%s,kind=\"policy\"} %d\n", labels, st.PolicyEvictions)
	writeHelpType(b, "fak_gateway_kv_memory_splits_total", "Radix KV prefix-cache edge splits performed to expose reusable mid-edge prefixes.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_memory_splits_total{%s} %d\n", labels, st.Splits)
	if st.L2HostCapacityBytes > 0 || st.L3Enabled {
		tierLabels := fmt.Sprintf("backend=\"%s\"", promQuote(backend))
		writeHelpType(b, "fak_gateway_kv_prefix_tier_resident_bytes", "Physical complete-prefix payload bytes owned by the native in-kernel cache, split by source tier and memory scope. Provider/proxy counters never enter this family.", "gauge")
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_resident_bytes{%s,tier=\"device_l1\",scope=\"device\"} %d\n", tierLabels, st.L1DeviceResidentBytes)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_resident_bytes{%s,tier=\"device_l1\",scope=\"host_metadata\"} %d\n", tierLabels, st.L1HostResidentBytes)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_resident_bytes{%s,tier=\"host_dram_l2\",scope=\"host\"} %d\n", tierLabels, st.L2HostResidentBytes)
		if st.L3Enabled {
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_resident_bytes{%s,tier=\"remote_http_l3\",scope=\"remote_referenced\"} %d\n", tierLabels, st.L3ReferencedBytes)
		}
		if st.L2HostCapacityBytes > 0 {
			writeHelpType(b, "fak_gateway_kv_prefix_tier_capacity_bytes", "Configured physical capacity for a native complete-prefix tier.", "gauge")
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_capacity_bytes{%s,tier=\"host_dram_l2\"} %d\n", tierLabels, st.L2HostCapacityBytes)
		}
		writeHelpType(b, "fak_gateway_kv_prefix_tier_lookups_total", "Native complete-prefix lookup outcomes by physical source tier, consulted in device L1, host DRAM L2, remote HTTP L3 order.", "counter")
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"device_l1\",outcome=\"hit\"} %d\n", tierLabels, st.L1Hits)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"device_l1\",outcome=\"miss\"} %d\n", tierLabels, st.L1Misses)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"device_l1\",outcome=\"fault\"} %d\n", tierLabels, st.L1Faults)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"host_dram_l2\",outcome=\"hit\"} %d\n", tierLabels, st.L2Hits)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"host_dram_l2\",outcome=\"miss\"} %d\n", tierLabels, st.L2Misses)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"host_dram_l2\",outcome=\"fault\"} %d\n", tierLabels, st.L2Faults)
		if st.L3Enabled {
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"remote_http_l3\",outcome=\"hit\"} %d\n", tierLabels, st.L3Hits)
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"remote_http_l3\",outcome=\"miss\"} %d\n", tierLabels, st.L3Misses)
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_lookups_total{%s,tier=\"remote_http_l3\",outcome=\"fault\"} %d\n", tierLabels, st.L3Faults)
		}
		writeHelpType(b, "fak_gateway_kv_prefix_tier_hit_tokens_total", "Prefix tokens restored from each native physical source tier.", "counter")
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_hit_tokens_total{%s,tier=\"device_l1\"} %d\n", tierLabels, st.L1HitTokens)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_hit_tokens_total{%s,tier=\"host_dram_l2\"} %d\n", tierLabels, st.L2HitTokens)
		if st.L3Enabled {
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_hit_tokens_total{%s,tier=\"remote_http_l3\"} %d\n", tierLabels, st.L3HitTokens)
		}
		writeHelpType(b, "fak_gateway_kv_prefix_tier_transfer_bytes_total", "Payload bytes transferred while staging or restoring native complete-prefix snapshots, split by physical tier and direction.", "counter")
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_bytes_total{%s,tier=\"host_dram_l2\",direction=\"stage\"} %d\n", tierLabels, st.L2StageBytes)
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_bytes_total{%s,tier=\"host_dram_l2\",direction=\"restore\"} %d\n", tierLabels, st.L2RestoreBytes)
		if st.L3Enabled {
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_bytes_total{%s,tier=\"remote_http_l3\",direction=\"stage\"} %d\n", tierLabels, st.L3StageBytes)
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_bytes_total{%s,tier=\"remote_http_l3\",direction=\"restore\"} %d\n", tierLabels, st.L3RestoreBytes)
			writeHelpType(b, "fak_gateway_kv_prefix_tier_transfer_latency_seconds_total", "Cumulative native complete-prefix remote transfer latency by direction, including failed attempts.", "counter")
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_latency_seconds_total{%s,tier=\"remote_http_l3\",direction=\"stage\"} %s\n", tierLabels, promFloat(float64(st.L3StageNanos)/float64(time.Second)))
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_latency_seconds_total{%s,tier=\"remote_http_l3\",direction=\"restore\"} %s\n", tierLabels, promFloat(float64(st.L3RestoreNanos)/float64(time.Second)))
			writeHelpType(b, "fak_gateway_kv_prefix_tier_transfer_faults_total", "Native complete-prefix transfer faults by physical tier and direction.", "counter")
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_faults_total{%s,tier=\"remote_http_l3\",direction=\"stage\"} %d\n", tierLabels, st.L3StageFaults)
			fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_transfer_faults_total{%s,tier=\"remote_http_l3\",direction=\"restore\"} %d\n", tierLabels, st.L3RestoreFaults)
		}
		writeHelpType(b, "fak_gateway_kv_prefix_tier_evictions_total", "Complete host-DRAM prefix images evicted from the bounded native L2.", "counter")
		fmt.Fprintf(b, "fak_gateway_kv_prefix_tier_evictions_total{%s,tier=\"host_dram_l2\"} %d\n", tierLabels, st.L2Evictions)
	}
}

func (m *gatewayMetrics) writeInKernelOOMMetrics(b *strings.Builder) {
	rows := m.inKernelOOMSnapshotData()
	writeHelpType(b, "fak_gateway_in_kernel_oom_total", "LOCAL in-kernel device OOMs and capacity precheck refusals, bucketed by memory class. These are WITNESSED fak-owned resource faults, distinct from provider-side errors.", "counter")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_total{class=\"%s\"} %d\n", promQuote(row.class), row.count)
	}
	writeHelpType(b, "fak_gateway_in_kernel_oom_failed_bytes_total", "Cumulative bytes requested by local in-kernel allocation OOMs and capacity refusals, bucketed by memory class.", "counter")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_failed_bytes_total{class=\"%s\"} %d\n", promQuote(row.class), row.failedBytes)
	}
	writeHelpType(b, "fak_gateway_in_kernel_oom_last_failed_bytes", "Most recent failed allocation or refused plan size for each memory class (0 until that class has faulted).", "gauge")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_last_failed_bytes{class=\"%s\"} %d\n", promQuote(row.class), row.lastFailedBytes)
	}
}

// writeUpstreamErrorMetrics renders the upstream/planner turn-failure family: a count of failed
// turns by kind (stalled / unreachable / oom / rate_limited / auth / forbidden / status_4xx /
// status_5xx / other). It is the cumulative metric twin of the glanceable per-turn `fak-turn …
// FAILED` debug line, so a scrape answers WHY turns failed — including a rate-limit storm vs an
// auth-failure storm — not just that the route returned a 502/504. Snapshot under the lock, then
// render in sorted key order so the scrape is byte-stable.
func (m *gatewayMetrics) writeUpstreamErrorMetrics(b *strings.Builder) {
	m.upstreamErrMu.Lock()
	snap := make(map[string]uint64, len(m.upstreamErrors))
	for k, v := range m.upstreamErrors {
		snap[k] = v
	}
	authSnap := make(map[string]uint64, len(m.upstreamAuthRefreshes))
	for k, v := range m.upstreamAuthRefreshes {
		authSnap[k] = v
	}
	fbSnap := make(map[string]uint64, len(m.upstreamForbiddenRetries))
	for k, v := range m.upstreamForbiddenRetries {
		fbSnap[k] = v
	}
	afSnap := make(map[string]uint64, len(m.upstreamAccountFailovers))
	for k, v := range m.upstreamAccountFailovers {
		afSnap[k] = v
	}
	m.upstreamErrMu.Unlock()
	writeHelpType(b, "fak_gateway_upstream_errors_total", "Upstream/planner turn failures, OBSERVED from the provider and relayed by fak (not a fak fault), by kind (stalled, unreachable, oom, rate_limited, auth, forbidden, status_4xx, status_5xx, other).", "counter")
	kinds := make([]string, 0, len(snap))
	for k := range snap {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, kind := range kinds {
		fmt.Fprintf(b, "fak_gateway_upstream_errors_total{kind=\"%s\"} %d\n", promQuote(kind), snap[kind])
	}
	writeCounter(b, "fak_gateway_upstream_retries_total", "Upstream retry attempts — fak's exponential backoff in response to OBSERVED, provider-reported 429/5xx from the planner — since process start.", int64(atomic.LoadUint64(&m.upstreamRetries)))
	// The time twin of the retry counter: how much wall-clock the backoff loop slept between
	// attempts. Always emitted (0 on a pushback-free session) so the panel exists from the
	// first scrape — this is the answer to "how much of my slow session was the provider's
	// rate limit, absorbed by fak, rather than fak itself".
	writeHelpType(b, "fak_gateway_upstream_retry_wait_seconds_total", "Wall-clock the backoff loop SLEPT between upstream retry attempts (fak-authored waits in response to OBSERVED, provider-reported 429/5xx) since process start — the time cost of the retries fak absorbed on the session's behalf.", "counter")
	fmt.Fprintf(b, "fak_gateway_upstream_retry_wait_seconds_total %s\n", promFloat(time.Duration(atomic.LoadUint64(&m.upstreamRetryWaitNS)).Seconds()))
	// The 401 token-rotation self-heal family: recovered = a fresh subscription OAuth token was
	// adopted mid-session (the live guarded session healed across a re-login), exhausted = no
	// fresher token landed within the grace window (the 401 surfaced and the agent dropped into
	// its own /login). Always emit BOTH series (even at 0) so a dashboard panel exists from the
	// first scrape — a missing "exhausted" series must not read as "no failures", and a healthy
	// session's "recovered 0 / exhausted 0" is itself the signal that no token expired.
	writeHelpType(b, "fak_gateway_upstream_auth_refresh_total", "401 token-rotation self-heals on the rotating Claude subscription path, by outcome: recovered (a fresh OAuth token was adopted mid-session and the call re-sent in place) or exhausted (no fresher token landed within the grace window, so the 401 surfaced).", "counter")
	for _, outcome := range []string{"recovered", "exhausted"} {
		fmt.Fprintf(b, "fak_gateway_upstream_auth_refresh_total{outcome=\"%s\"} %d\n", outcome, authSnap[outcome])
	}
	// The 403 transient-recovery family, the permission-flap twin of the 401 auth-refresh family:
	// recovered = a retry within the short bounded window returned 200 (a transient abuse/capacity
	// gate cleared and the live session healed in place instead of dropping into a spurious
	// /login), exhausted = the window/attempts elapsed still 403ing (the denial is the permanent
	// entitlement kind, now surfaced with the actionable answer). Both series always emitted (even
	// at 0) so a "recovered N / exhausted 0" reads as "N transient flaps, all self-healed" from the
	// first scrape. This is the metric the 2026-07-03 gem8 transient-403 storm proved was missing.
	writeHelpType(b, "fak_gateway_upstream_forbidden_retry_total", "403 transient-permission recoveries, by outcome: recovered (a retry within the bounded window returned 200, so a transient abuse/capacity gate cleared and the session healed in place) or exhausted (the window/attempts elapsed still 403ing, so a permanent entitlement denial surfaced). OBSERVED from the provider, absorbed by fak.", "counter")
	for _, outcome := range []string{"recovered", "exhausted"} {
		fmt.Fprintf(b, "fak_gateway_upstream_forbidden_retry_total{outcome=\"%s\"} %d\n", outcome, fbSnap[outcome])
	}
	// The account-scoped failover family: recovered = a 403/402 named this credential's org/region/
	// billing as walled and fak adopted a permitted SIBLING account so the turn completed in place;
	// exhausted = no permitted sibling existed and the account-scoped denial surfaced. This is the
	// org-OAuth-disabled signal — a wall that no retry or re-login clears — so a "recovered N" here
	// means N sessions that would otherwise have died on a futile /login were auto-switched onto a
	// working account. Both series always emitted (even at 0) for panel stability from the first
	// scrape. OBSERVED denial from the provider; the failover action is WITNESSED (fak authored it).
	writeHelpType(b, "fak_gateway_upstream_account_failover_total", "Account-failover-arm outcomes over the sibling-seat swap, by outcome: an org/region/billing-walled 403/402 reports recovered (a permitted sibling account was adopted and the walled turn completed in place, healing a session re-login could not) or exhausted (no permitted sibling existed, so the account-scoped denial surfaced); a 429 ACCOUNT CAP (session/weekly/usage, whose reset can be hours away) reports rehomed_seat (the session swapped to a free sibling seat and completed the turn now instead of sleeping toward the cap reset) or rehome_seat_unavailable (every sibling seat was capped/walled, so the cap-aware backoff rode it out). OBSERVED denial from the provider; the failover/rehome action is WITNESSED (fak authored it).", "counter")
	for _, outcome := range []string{"recovered", "exhausted", "rehomed_seat", "rehome_seat_unavailable"} {
		fmt.Fprintf(b, "fak_gateway_upstream_account_failover_total{outcome=\"%s\"} %d\n", outcome, afSnap[outcome])
	}
	writeCounter(b, "fak_gateway_served_inline_total", "Read-only tool calls the vDSO served LOCALLY on a served turn (vDSO live in the hot path): a re-proposed read whose fresh cached answer fak folded into the assistant turn and dropped before the client could re-run it — each one a saved engine round-trip. WITNESSED (fak authored the serve), distinct from the kernel fak_kernel_vdso_hits_total which only the explicit k.Syscall path bumps.", int64(m.servedInlineSnapshot()))
}

// writeInferenceMetrics renders the model-generation family from the live
// accumulators. fak_kernel_*/fak_vdso_* count the ADJUDICATION + dedup fast path,
// which a plain chat/messages turn never touches — so on a box serving real chat
// they read 0 while the model is busy decoding. This family is the missing signal:
// turns served (by finish reason), prompt/completion/cached token totals, the
// cumulative decode wall-clock, and the mean output tokens/sec derived from the two.
// Until the first served turn the counters carry no series and the derived rate is 0,
// so an idle gateway never publishes a phantom throughput. The snapshot is returned
// so the fleet-value block can reuse the same accumulator read (cached tokens, decode
// wall-clock) without re-locking.
func (m *gatewayMetrics) writeInferenceMetrics(b *strings.Builder) inferenceSnapshot {
	snap := m.inferenceSnapshotData()

	writeHelpType(b, "fak_gateway_inference_requests_total", "Model completion turns served by the gateway planner, by finish reason.", "counter")
	reasons := make([]string, 0, len(snap.reqs))
	for r := range snap.reqs {
		reasons = append(reasons, r)
	}
	sort.Strings(reasons)
	for _, r := range reasons {
		fmt.Fprintf(b, "fak_gateway_inference_requests_total{finish_reason=\"%s\"} %d\n", promQuote(r), snap.reqs[r])
	}

	writeCounter(b, "fak_gateway_inference_prompt_tokens_total", "Prompt (input) tokens summed across served model turns.", int64(snap.promptTok))
	writeCounter(b, "fak_gateway_inference_completion_tokens_total", "Completion (generated) tokens summed across served model turns.", int64(snap.complTok))
	writeCounter(b, "fak_gateway_inference_cached_prompt_tokens_total", "Prompt (input) tokens the upstream PROVIDER served from its own prompt cache (cache_read) across served turns, normalized across Anthropic/OpenAI/Gemini. This is provider-side reuse — distinct from the local fak_vdso_*/fak_cache_* caches — and reads 0 on the in-kernel path (no provider).", int64(snap.cachedTok))

	// fak_gateway_provider_cache_local_trust — the #432 acceptance-3 invariant,
	// exported live. The cached-prompt-tokens counter above is PERFORMANCE evidence
	// (cost/latency); it is NEVER local trust. The value is DERIVED from the cachemeta
	// provider_prefix materialization gate (the same gate the kernel uses), so it is
	// structurally 0 — and would flip to 1 only if that proven gate ever regressed to
	// treat a provider cache_read as a serveable local-trust hit.
	providerLocalTrust := 0
	if providerCacheEvidence(snap.cachedTok).CanServe() {
		providerLocalTrust = 1
	}
	writeHelpType(b, "fak_gateway_provider_cache_local_trust", "Whether the upstream PROVIDER's prompt-cache reuse counts as LOCAL TRUST (#432 acceptance 3). Structurally 0: provider cache is performance evidence (cost/latency) only — derived live from the cachemeta provider_prefix materialization gate, not a prose promise. A 1 would mean the trust/performance separation regressed.", "gauge")
	fmt.Fprintf(b, "fak_gateway_provider_cache_local_trust %d\n", providerLocalTrust)

	// Provider prompt-cache HIT rate: the token total above carried no denominator,
	// so a dashboard could see tokens-cached but not how OFTEN a turn hit the provider
	// cache. cached_prompt_hits_total counts served turns with a provider cache read
	// (>0 cached tokens); the ratio is hits/turns, mirroring fak_gateway_vdso_hit_ratio.
	var inferTurns uint64
	for _, n := range snap.reqs {
		inferTurns += n
	}
	writeCounter(b, "fak_gateway_inference_cached_prompt_hits_total", "Served model turns whose prompt got a provider prompt-cache READ (cached tokens > 0). The hit COUNT behind the cached-prompt-tokens total above.", int64(snap.cachedHits))
	writeHelpType(b, "fak_gateway_inference_cached_prompt_hit_ratio", "Fraction of served model turns that hit the provider prompt cache (cached_prompt_hits / turns; 0 until the first turn). The provider-cache analogue of fak_gateway_vdso_hit_ratio.", "gauge")
	cacheHitRatio := 0.0
	if inferTurns > 0 {
		cacheHitRatio = float64(snap.cachedHits) / float64(inferTurns)
	}
	fmt.Fprintf(b, "fak_gateway_inference_cached_prompt_hit_ratio %s\n", promFloat(cacheHitRatio))

	writeHelpType(b, "fak_gateway_inference_duration_seconds_total", "Cumulative wall-clock spent inside the planner generating completions (prefill+decode on the in-kernel path; round-trip on a proxy).", "counter")
	fmt.Fprintf(b, "fak_gateway_inference_duration_seconds_total %s\n", promFloat(snap.decodeSecs))

	writeHelpType(b, "fak_gateway_inference_output_tokens_per_second", "Mean output tokens/sec across served turns (completion tokens / total inference wall-clock; 0 until the first turn).", "gauge")
	tps := 0.0
	if snap.decodeSecs > 0 {
		tps = float64(snap.complTok) / snap.decodeSecs
	}
	fmt.Fprintf(b, "fak_gateway_inference_output_tokens_per_second %s\n", promFloat(tps))

	// Prefill vs decode split (time-to-first-token). The output_tokens_per_second
	// above blends prompt ingest (prefill) with generation (decode) into one mean,
	// which hides the first-request-slow story: a cold prefill dominates the first
	// turn's wall-clock while decode is fast. These two rates separate them, measured
	// ONLY over the turns whose TTFT was observable (the streaming passthrough path);
	// a buffered turn reports no boundary and is excluded so the rates never blend a
	// measured turn with an unmeasured one. Until the first measured turn the
	// denominators are 0 and the rates stay 0 (no phantom throughput).
	writeHelpType(b, "fak_gateway_inference_ttft_turns_total", "Served turns whose time-to-first-token (prefill boundary) was observable — the streaming passthrough path only. The denominator behind the prefill seconds/rate below; buffered turns are excluded.", "counter")
	fmt.Fprintf(b, "fak_gateway_inference_ttft_turns_total %d\n", snap.ttftTurns)
	writeHelpType(b, "fak_gateway_inference_prefill_seconds_total", "Cumulative time-to-first-token wall-clock (prompt ingest + first token) across turns that measured it. Decode wall-clock is fak_gateway_inference_duration_seconds_total minus this over the same turns.", "counter")
	fmt.Fprintf(b, "fak_gateway_inference_prefill_seconds_total %s\n", promFloat(snap.prefillSecs))
	writeHelpType(b, "fak_gateway_inference_prefill_tokens_per_second", "Prefill throughput: prompt (input) tokens ingested per second of TTFT wall-clock, over the turns that measured TTFT (prefill_prompt_tokens / prefill_seconds; 0 until the first measured turn). This is the cold-prefill rate that dominates a slow first request.", "gauge")
	prefillTPS := 0.0
	if snap.prefillSecs > 0 {
		prefillTPS = float64(snap.prefillPromptTok) / snap.prefillSecs
	}
	fmt.Fprintf(b, "fak_gateway_inference_prefill_tokens_per_second %s\n", promFloat(prefillTPS))
	// Decode-only rate: completion tokens over the decode phase (total minus prefill)
	// across the measured turns. This is the steady-state generation speed an operator
	// expects — distinct from output_tokens_per_second, which is diluted by prefill.
	// Falls back to 0 (not the blended rate) when no turn measured TTFT, so the two
	// rates never silently coincide.
	writeHelpType(b, "fak_gateway_inference_decode_tokens_per_second", "Decode throughput: completion (generated) tokens per second of DECODE wall-clock (total inference time minus prefill) over the turns that measured TTFT (0 until the first measured turn). The steady-state generation speed, undiluted by cold prefill — contrast fak_gateway_inference_output_tokens_per_second, which blends both.", "gauge")
	decodeTPS := 0.0
	if snap.measuredDecodeSecs > 0 {
		decodeTPS = float64(snap.measuredComplTok) / snap.measuredDecodeSecs
	}
	fmt.Fprintf(b, "fak_gateway_inference_decode_tokens_per_second %s\n", promFloat(decodeTPS))

	// #3176 Q1/Q2 on /metrics: the decode tok/s gauge above reports HOW FAST decode runs,
	// but not WHY. These two host-static facts answer the issue's other two levers on the
	// SAME surface an operator scrapes ("measurable without wall-clock guessing"), so a slow
	// decode_tokens_per_second can be read next to whether the Q8 SIMD lane actually fired
	// and how many decode streams dispatch — without grepping the server's inkernel_chat log
	// line. Both resolve once at model-package init (CPUID/XGETBV tier + the many-core worker
	// cap of 40e0afd), so they are constant across scrapes, like fak_gateway_build_info.
	q8Kernel, q8Fused := model.Q8DecodeKernel()
	writeHelpType(b, "fak_gateway_inference_q8_decode_kernel_info", "Static Q8_0 decode inner-kernel tier resolved for this host (#3176 Q1). kernel is avx512/avx2/scalar on amd64, neon-amort/neon/scalar on arm64, scalar on a no-SIMD build; fused=true only where the fused fast decode GEMV (qMatRowsRangeFast) is the active path (amd64 + AVX-512). Value is always 1 — read the labels. Answers 'is the SIMD decode lane engaged, or did it fall back to the reference path?' at the /metrics surface.", "gauge")
	fmt.Fprintf(b, "fak_gateway_inference_q8_decode_kernel_info{kernel=\"%s\",fused=\"%t\"} 1\n", promQuote(q8Kernel), q8Fused)
	writeHelpType(b, "fak_gateway_inference_q8_decode_workers", "Effective Q8 batch-1 decode-worker count (#3176 Q2): the parallel decode streams the in-kernel GEMV dispatches after the many-core cap (40e0afd). On a 256-thread box the default budget reads a modest capped value here — not 256 (the oversubscription collapse) nor 1 (single-thread) — so an operator can SEE decode parallelizes across a sane stream count. Reflects FAK_WORKERS/FAK_BUDGET/-budget when set.", "gauge")
	fmt.Fprintf(b, "fak_gateway_inference_q8_decode_workers %d\n", model.Q8DecodeWorkers())

	// Latency DISTRIBUTIONS — the P50/P95/P99 tail the cumulative-mean rates above
	// structurally hide. Each is empty (count 0) until the first qualifying turn, so an
	// idle gateway publishes no phantom distribution. These are the fak analogues of
	// vLLM's time_to_first_token / inter_token_latency / e2e_request_latency histograms,
	// bringing the de facto serving-latency Prometheus SET to parity.
	writeHelpType(b, "fak_gateway_inference_ttft_seconds", "Time-to-first-token (prefill: prompt ingest + first token) distribution, over the streamed turns whose prefill boundary was observable. The percentile view behind fak_gateway_inference_prefill_seconds_total's mean. fak analogue of vLLM time_to_first_token_seconds.", "histogram")
	writeHistogram(b, "fak_gateway_inference_ttft_seconds", "", snap.ttftHist)
	writeHelpType(b, "fak_gateway_inference_tpot_seconds", "Per-output-token (inter-token) latency distribution = decode wall-clock / generated tokens, per measured turn. The percentile view behind fak_gateway_inference_decode_tokens_per_second. fak analogue of vLLM inter_token_latency_seconds.", "histogram")
	writeHistogram(b, "fak_gateway_inference_tpot_seconds", "", snap.tpotHist)
	writeHelpType(b, "fak_gateway_inference_e2e_seconds", "Whole model-turn wall-clock distribution, over EVERY served turn (buffered or streamed). fak analogue of vLLM e2e_request_latency_seconds.", "histogram")
	writeHistogram(b, "fak_gateway_inference_e2e_seconds", "", snap.e2eHist)
	return snap
}

// writeFleetValueMetrics renders the hero-axis KPIs the live gateway can derive from
// its own counters — the per-node ingredients of fak's product headline (agent-fleet
// serving efficiency, HERO-BENCHMARK-2026-06-21.md). They are deliberately the LIVE
// cumulative mechanism counts, not an A/B headline: the comparable "turns saved vs a
// no-kernel baseline" number is only honest when both arms ran the same workload
// (LIVE-RESULTS.md / TICKETS T2), which a single serving node cannot witness. So this
// reports what THIS process measurably did:
//
//   - turns_saved_total{mechanism} — vDSO dedup hits (an engine round-trip not taken)
//     and grammar repairs (a retry turn the baseline spends re-emitting a malformed
//     call). These are the two turn-saving levers LIVE-RESULTS attributes turns to.
//   - context_pollutions_blocked_total — untrusted/poisoned tool-result payloads the
//     context-MMU paged out before they entered the model's context window (the live
//     "context saved" KPI; on a weak model each is also a derailment averted).
//   - context_pollution_rate — quarantines per kernel submission, the same ratio the
//     A/B report computes as context_pollution_rate (internal/bench).
//   - agent_seconds_total — cumulative agent-serving wall-clock (generation + kernel
//     op adjudication/dispatch), the denominator for a per-agent-second view.
//
// The KV-reuse "context saved" KPI is fak_gateway_inference_cached_prompt_tokens_total
// in the inference family above; it is not re-emitted here to avoid a duplicate series.
func writeFleetValueMetrics(b *strings.Builder, c kernel.Counters, servedInline uint64, agentSeconds float64) {
	writeHelpType(b, "fak_gateway_turns_saved_total", "Agent turns the kernel saved the served fleet, by mechanism: vdso_dedup = a duplicate read served from the fast path with no engine round-trip; grammar_repair = a malformed tool call repaired in-syscall instead of costing the baseline a retry turn. The comparable 'turns saved' headline (LIVE-RESULTS.md) is this measured against a no-kernel baseline arm; this is the live cumulative count of each mechanism firing.", "counter")
	fmt.Fprintf(b, "fak_gateway_turns_saved_total{mechanism=\"vdso_dedup\"} %d\n", c.VDSOHits+int64(servedInline))
	fmt.Fprintf(b, "fak_gateway_turns_saved_total{mechanism=\"grammar_repair\"} %d\n", c.Transforms)

	writeCounter(b, "fak_gateway_context_pollutions_blocked_total", "Untrusted/poisoned tool-result payloads the context-MMU paged out before they reached the model's context window — each a context-window pollution (and on a weak model a derailment) prevented. The live 'context saved' KPI.", c.Quarantines)

	writeHelpType(b, "fak_gateway_context_pollution_rate", "Quarantines per kernel submission, computed live over this process's submissions (the A/B report's context_pollution_rate; 0 when nothing has been submitted).", "gauge")
	pollRate := 0.0
	if c.Submits > 0 {
		pollRate = float64(c.Quarantines) / float64(c.Submits)
	}
	fmt.Fprintf(b, "fak_gateway_context_pollution_rate %s\n", promFloat(pollRate))

	writeHelpType(b, "fak_gateway_agent_seconds_total", "Cumulative wall-clock the gateway spent doing agent work: model generation (inference) plus kernel operation adjudication/dispatch (syscall, adjudicate, admit). The denominator for a live tokens/turns-per-agent-second view.", "counter")
	fmt.Fprintf(b, "fak_gateway_agent_seconds_total %s\n", promFloat(agentSeconds))
}

// writeToolPageMetrics renders the ctxmmu tool-schema page catalog family (#2440): the outbound
// tool-def counterpart of the inbound prompt-MMU. Both rows are WITNESSED (fak owns the catalog):
//   - tool_schema_resident_bytes (gauge): bytes of tool schema currently RESIDENT in the page
//     table. An evicted page contributes 0 — its bytes live re-faultably in the CAS, not the
//     transcript — so this gauge falls as the resident set is paged down and never counts a
//     schema twice, no matter how many turns re-advertise it.
//   - tool_page_dedup_hits_total (counter): Register calls that deduped to an existing
//     content-hashed page. It climbs every turn a tool schema is re-advertised unchanged (the
//     catalog is the tool's home, so the identical bytes are stored once and shared), the direct
//     witness that paging the tool catalog out of the transcript stopped the per-turn re-injection
//     churn the inspiring harness suffered.
//
// A bare/toolPages-less Server renders both as 0 rather than omitting the family, so a scrape
// target never sees the row vanish.
func (s *Server) writeToolPageMetrics(b *strings.Builder) {
	var residentBytes, dedupHits int64
	if s != nil && s.toolPages != nil {
		residentBytes = s.toolPages.ResidentBytes()
		dedupHits = s.toolPages.DedupHits()
	}
	writeHelpType(b, "fak_gateway_tool_schema_resident_bytes", "WITNESSED (fak authored): tool-schema bytes currently RESIDENT in the ctxmmu tool-page catalog (#2440). An evicted schema contributes 0 — its bytes live re-faultably in the CAS, not the transcript — so this gauge is the live residency of the deterministic per-turn resident set, never the cumulative catalog size.", "gauge")
	fmt.Fprintf(b, "fak_gateway_tool_schema_resident_bytes %d\n", residentBytes)
	writeCounter(b, "fak_gateway_tool_page_dedup_hits_total", "WITNESSED (fak authored): tool-schema Register calls that deduped to an existing content-hashed page (#2440). Keyed by content hash, never by name, so it climbs each turn a tool schema is re-advertised unchanged — the witness that the ctxmmu owning the tool catalog stopped the per-turn schema re-injection churn instead of re-inflating identical bytes.", dedupHits)
}

// writeCompactionMetrics renders the history-compaction family, split along what fak CONTROLS
// versus what it can only OBSERVE — keep the two apart or a provider-side miss reads as a fak
// bug it cannot support:
//   - WITNESSED (fak authored): attempts{fired|bailed|off}, bail_reason, dropped_turns, shed.
//     A turn counts `fired` only when the protected-prefix bytes it shipped were byte-identical
//     to the input (else it bails to `prefix_mismatch`). These describe what fak SENT.
//   - OBSERVED (provider-reported, relayed verbatim): cache_read_tokens / post_fire_cache_read.
//     fak attributes nothing to itself here; it forwards the upstream's number.
//
// The single fak-fault signal is bail_reason{reason="prefix_mismatch"}>0. A cratered cache_read
// while fires climb is NOT that bug — the shipped prefix was byte-identical, so the provider
// missed for a reason fak does not control (cache TTL expiry, eviction, or the client moving its
// own breakpoint). Reading the crater as "the splice broke the cache" is the conflation this
// split exists to prevent.
func (m *gatewayMetrics) writeCompactionMetrics(b *strings.Builder) {
	snap := m.compactionSnapshotData()

	writeHelpType(b, "fak_gateway_compaction_attempts_total",
		"WITNESSED (fak authored): Anthropic history-compaction attempts by outcome: fired (body rewritten, protected prefix shipped byte-identical), bailed (returned identity), off (budget unset).", "counter")
	for _, o := range []string{"fired", "bailed", "off"} { // stable order; emit at 0 so the panel exists pre-first-fire
		fmt.Fprintf(b, "fak_gateway_compaction_attempts_total{outcome=%q} %d\n", o, snap.attempts[o])
	}

	// The closed-set claim is DERIVED from agent.CompactBailReasons(), never re-typed here.
	// Hand-spelling it drifted twice: the string declared 9 members while internal/agent
	// emitted 13, so decode_failed (#5387) and three others were invisible to anyone who
	// trusted the declaration and built an alert over the listed labels (#5441). The emitter
	// owns the vocabulary; this rendering reads it, and agent's own registration test fails if
	// a CompactReason* constant is added without joining it.
	writeHelpType(b, "fak_gateway_compaction_bail_reason_total",
		"WITNESSED (fak authored): why a compaction attempt bailed to identity (closed set, derived from the emitting package's registered vocabulary: "+
			strings.Join(agent.CompactBailReasons(), "|")+"). prefix_mismatch>0 is the ONLY fak-fault cache signal and must stay 0; splice_failed/redecode_failed are splice bugs and must stay 0 too.", "counter")
	rs := make([]string, 0, len(snap.bailReasons))
	for r := range snap.bailReasons {
		rs = append(rs, r)
	}
	sort.Strings(rs)
	for _, r := range rs {
		fmt.Fprintf(b, "fak_gateway_compaction_bail_reason_total{reason=%q} %d\n", promQuote(r), snap.bailReasons[r])
	}

	// The ALERTABLE compaction-health pair (#5443). attempts{bailed} above counts every
	// non-fire, including the requests that were never compaction candidates at all — a
	// two-message subagent ping cannot compact at any setting — and on mixed fleet traffic
	// those dominate, pinning bails/(fires+bails) near 1.0 whatever the compactor does. These
	// two series are added ALONGSIDE the unchanged counters rather than redefining them: the
	// held-out population is published as its own counter so it is visible, not silently
	// subtracted, and the rate that divides only over eligible attempts is published beside it.
	// The same partition drives internal/gatewayusageledger's NonCandidateBails /
	// CandidateBailRate, so the live scrape and the durable ledger fold agree.
	nonCandidateBails, candidateBails := compactBailPartition(snap.bailReasons)
	writeCounter(b, "fak_gateway_compaction_non_candidate_bails_total",
		"WITNESSED (fak authored): the SUBSET of compaction bails the compactor decided before any compactible span existed (non_json, no_messages_key, decode_failed, too_few_msgs — see the bail_reason vocabulary). These requests were never in the running, so they say nothing about compaction health; they are held out of fak_gateway_compaction_candidate_bail_rate and published here so the held-out population stays visible. A cell whose bails are almost all these is a normal short-request stream, not a sick compactor. Subtract it from attempts{outcome=\"bailed\"} to recover the eligible bails.",
		int64(nonCandidateBails))
	writeHelpType(b, "fak_gateway_compaction_candidate_bail_rate",
		"WITNESSED (fak authored): eligible bails / (fires + eligible bails) — compaction declines over attempts that actually HAD a compactible span. This is the alertable rate; bails/(fires+bails) from attempts{} is not, because it counts requests that were never compactible and therefore reads near 1.0 on healthy and broken traffic alike. 0 when no attempt was ever eligible (an honest zero, not a fabricated ratio). A reason the emitting package has not registered counts as ELIGIBLE, so this can only read conservatively high — it never silently understates a problem.",
		"gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_candidate_bail_rate %s\n", promFloat(compactCandidateBailRate(snap.attempts["fired"], candidateBails)))

	writeCounter(b, "fak_gateway_compaction_anchor_starved_total", "WITNESSED (fak authored): under_budget bails whose protected prefix ALREADY exceeded the budget — the cache_control anchor swallowed the conversation, so compaction structurally cannot fire no matter how long the session grows. A SUBSET of bail_reason{reason=\"under_budget\"}, broken out because the two are opposite: plain under_budget is a benign short session, anchor-starved is the dormant-on-real-Claude-Code-traffic pathology (issue #1407) that no budget tightening fixes.", int64(snap.anchorStarved))
	writeCounter(b, "fak_gateway_compaction_thrash_sessions_total", "WITNESSED (fak authored): sessions that hit the closed verdict COMPACTION_THRASH (#2424) — the context window refilled to the compaction limit on 3 consecutive turns, so the lever fired every turn and bought no lasting headroom. Counted ONCE per thrashing stretch, so this is sessions-that-thrashed, not turns-spent-thrashing. It is NOT a bail (compaction FIRED each time), which is why it is published here instead of under bail_reason_total, whose label set stays the compactor's own closed vocabulary. Nonzero means the binding constraint is the traffic, not the budget: tightening the budget compacts more often and changes nothing. Detection is always on; the session STOP that acts on it is opt-in behind FAK_COMPACTION_THRASH_STOP.", int64(snap.thrashSessions))
	writeCounter(b, "fak_gateway_compaction_dropped_turns_total", "WITNESSED (fak authored): whole messages stubbed out across all fires.", int64(snap.dropped))
	writeCounter(b, "fak_gateway_compaction_shed_tokens_total", "WITNESSED (fak authored): estimated tokens fak removed from the outbound body by history compaction fires plus uncached-tail trim (same ~4ch/token currency as the budget and provider input_tokens). What fak SENT — not a claim about what the provider billed.", int64(snap.shed))
	writeCounter(b, "fak_gateway_compaction_cache_read_tokens_total", "OBSERVED (provider-reported, relayed verbatim): cumulative cache_read_input_tokens on compacted turns. Pair with shed_tokens to see the net effect; attribute nothing to fak from it alone — fak only guarantees the prefix it shipped was byte-identical (see attempts{fired} with prefix_mismatch=0).", int64(snap.cacheReads))
	writeHelpType(b, "fak_gateway_compaction_post_fire_cache_read_tokens",
		"OBSERVED (provider-reported): cache_read_input_tokens on the MOST RECENT compacted turn. If this craters while fires climb, the prefix fak shipped was still byte-identical (witnessed by fired with prefix_mismatch=0), so the provider did not reuse it for a reason fak does NOT control: cache TTL expiry, eviction, or the client moving its own breakpoint. Only bail_reason{reason=\"prefix_mismatch\"}>0 is fak's bug.", "gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_post_fire_cache_read_tokens %s\n", promFloat(snap.lastCacheRd))
	writeCounter(b, "fak_gateway_uncached_trim_results_total", "WITNESSED (fak authored): oversized old tool_result bodies shrunk in the uncached post-breakpoint region. The transform keeps the protected cache prefix byte-identical and leaves recent/cache_control-bearing results intact.", int64(snap.uncachedTrimResults))
	writeCounter(b, "fak_gateway_uncached_trim_shed_tokens_total", "WITNESSED (fak authored): estimated tokens removed by uncached-tail oversized-result trim, also folded into fak_gateway_compaction_shed_tokens_total for the owner=\"fak\" cache-savings attribution.", int64(snap.uncachedTrimShed))

	// The managed-cache 1h TTL upgrade family (--managed-cache, epic #1844 C6). WITNESSED:
	// fak spliced ttl:"1h" into an existing stable-head cache_control (or bailed for the
	// named reason). Whether the provider then HONORS the longer TTL across an idle gap is
	// OBSERVED on the cache_read counters, never claimed here. The "upgraded" row is emitted
	// even at 0 so an active lever with zero eligible heads is visible, not silent.
	writeHelpType(b, "fak_gateway_cache_ttl_upgrade_total",
		"WITNESSED (fak authored): managed-cache 1h TTL upgrade attempts on the outbound Anthropic wire, by outcome: upgraded (ttl:\"1h\" spliced into an existing stable system/tools-head cache_control), placed_and_upgraded (no cache_control existed; fak placed the stable-head breakpoint AND upgraded it in one turn, #2175), or the bail reason (no_stable_breakpoint|already_1h|ttl_already_set|volatile_head|non_json|splice_failed|redecode_failed). Rows exist only while --managed-cache is ACTIVE. splice_failed/redecode_failed are fak bugs and must stay 0. The provider honoring the 1h tier across an idle gap is OBSERVED via cache_read, not claimed by this counter.", "counter")
	fmt.Fprintf(b, "fak_gateway_cache_ttl_upgrade_total{outcome=%q} %d\n", "upgraded", snap.ttlUpgrades["upgraded"])
	ttlReasons := make([]string, 0, len(snap.ttlUpgrades))
	for r := range snap.ttlUpgrades {
		if r != "upgraded" {
			ttlReasons = append(ttlReasons, r)
		}
	}
	sort.Strings(ttlReasons)
	for _, r := range ttlReasons {
		fmt.Fprintf(b, "fak_gateway_cache_ttl_upgrade_total{outcome=%q} %d\n", promQuote(r), snap.ttlUpgrades[r])
	}

	// WITNESSED (fak authored): the OFFENSIVE cache-breakpoint placement family (#806). "placed"
	// counts turns where fak spliced a cache_control breakpoint onto the stable head of a caller that
	// sent none — the fak-UNLOCKED slice, since a no-breakpoint caller earns 0 provider cache without
	// it. "already_set" counts the Claude-Code shape fak leaves to the client's own cache (NOT fak's).
	// The "placed" row is emitted even at 0 so a passthrough that never placed is visible, not silent.
	// splice_failed/redecode_failed are fak bugs and must stay 0.
	writeHelpType(b, "fak_gateway_cache_breakpoint_placement_total",
		"WITNESSED (fak authored): offensive cache-breakpoint placements on the outbound Anthropic wire, by outcome: placed (a cache_control breakpoint spliced onto the stable system/tools head of a caller that sent none) or the bail reason (already_set|no_stable_head|volatile_head|non_json|splice_failed|redecode_failed). placed is the fak-unlocked slice — the provider cache_read those turns earn would be 0 without it; already_set is the client's own cache, not fak's. The provider serving turns 2..N from that cache is OBSERVED via cache_read, not claimed by this counter.", "counter")
	fmt.Fprintf(b, "fak_gateway_cache_breakpoint_placement_total{outcome=%q} %d\n", "placed", snap.placementAttempts["placed"])
	placementReasons := make([]string, 0, len(snap.placementAttempts))
	for r := range snap.placementAttempts {
		if r != "placed" {
			placementReasons = append(placementReasons, r)
		}
	}
	sort.Strings(placementReasons)
	for _, r := range placementReasons {
		fmt.Fprintf(b, "fak_gateway_cache_breakpoint_placement_total{outcome=%q} %d\n", promQuote(r), snap.placementAttempts[r])
	}

	// The INBOUND tool-floor prune family (the tools[] twin of the compaction shed above).
	// WITNESSED: how many unreachable tool DEFINITIONS fak dropped from the advertised surface,
	// a pure uncached-token saving the pruner makes only after the cache_control breakpoint so it
	// never bursts the cache. Both stay 0 on the dominant Claude Code path (its single breakpoint
	// sits on the LAST tool, so nothing is droppable) — which, before these rows existed, was the
	// invisible fact: the prune result was discarded with no counter.
	pruneTurns, pruneCount := m.inboundToolPruneSnapshot()
	writeCounter(b, "fak_gateway_inbound_tools_pruned_total", "WITNESSED (fak authored): cumulative unreachable tool DEFINITIONS dropped from the outbound tools[] across the session. A pure uncached-token saving — the pruner drops only tools after the cache_control breakpoint and re-proves the protected prefix is byte-identical, so a counted prune never bursts the provider-side upstream cache.", int64(pruneCount))
	writeCounter(b, "fak_gateway_inbound_tools_prune_turns_total", "WITNESSED (fak authored): turns on which at least one unreachable tool def was pruned from tools[]. Zero on a harness (e.g. Claude Code) whose single cache_control breakpoint sits on the LAST tool, since nothing is then droppable.", int64(pruneTurns))
	writeCounter(b, "fak_gateway_inbound_tools_pruned_then_proposed_total", "WITNESSED (fak authored): pruned tool definition names that the model later proposed anyway, counted once per trace/tool. Nonzero means the advertised floor and observed model behavior drifted; call-time adjudication still default-denies the proposal.", int64(m.inboundPrunedToolProposalSnapshot()))

	// The OUTBOUND cold-tool DEFERRAL family (the 10x floor lever, --defer-cold-tools #3232 under
	// epic #3229) — the OUTBOUND twin of the inbound prune family above, and the render surface the
	// #3232 follow-up called for. WITNESSED: how many cold tool DEFINITIONS fak marked
	// defer_loading and handed to the provider's tool-search fault-in. Unlike the prune, this does
	// NOT shrink request bytes — the reduction is provider-side (only the hot core loads into
	// context), so these counters witness the deferral fak DROVE; the actual token drop is OBSERVED
	// via input_tokens/cache_read, not claimed here. Both stay 0 when the lever is off (its DEFAULT)
	// or when no turn had a cold tool tail to defer.
	deferTurns, deferCold := m.toolDeferSnapshot()
	writeCounter(b, "fak_gateway_tool_defer_cold_total", "WITNESSED (fak authored): cumulative cold tool DEFINITIONS marked defer_loading:true on the outbound Anthropic body across the session (the 10x floor lever, --defer-cold-tools #3232). Cache-safe by construction (deterministic, byte-stable tools[] turn-over-turn), so a counted deferral never bursts the upstream prompt cache. The provider-side context/token drop it drives is OBSERVED via input_tokens/cache_read, not claimed by this counter.", int64(deferCold))
	writeCounter(b, "fak_gateway_tool_defer_turns_total", "WITNESSED (fak authored): turns on which fak deferred the cold tool tail (marked >=1 cold def defer_loading and injected a tool_search_tool). Zero when --defer-cold-tools is off (its default) or when every advertised tool was hot; nonzero means the lever fired that turn.", int64(deferTurns))
	// The DENOMINATOR of the two counters above (#3621). Both of them are pure numerators, so a
	// flat zero reads identically whether the lever was never armed or was armed and bit on
	// nothing — the silent-identity failure mode. This counter accrues only PAST the eligibility
	// gate (lever on, Anthropic passthrough wire, ablation arm off), so `_cold_total == 0 AND
	// _standdown_turns_total >= 3` is the scrape-side form of the DEFER_ENABLED_BUT_INERT finding
	// the guard banner and /debug/vars cache_attribution.fak_defer_finding raise.
	standDownTurns, _ := m.toolDeferStandDownSnapshot()
	writeCounter(b, "fak_gateway_tool_defer_standdown_turns_total", "WITNESSED (fak authored): turns on which the cold-tool-defer transform RAN — lever on, Anthropic passthrough wire, ablation arm off — and stood down to byte-identity anyway (no cold tools, a client body already deferred, or a splice fak could not prove). The denominator for fak_gateway_tool_defer_cold_total: a session with cold_total==0 and this counter climbing is the lever armed-but-inert (#3621), NOT a lever that was left off.", int64(standDownTurns))

	// The tool_reference SANITIZE family (a CORRECTNESS transform, not a cache saving). WITNESSED:
	// how many Claude-Code-internal tool_reference blocks fak rewrote into wire-valid text blocks so
	// the body was not 400'd upstream as malformed. Nonzero means fak repaired a body the provider
	// would otherwise have rejected outright (witnessed defect: session b98cf818, killed by a
	// ToolSearch tool_result carrying tool_reference blocks).
	refTurns, refConverted := m.toolRefSanitizeSnapshot()
	writeCounter(b, "fak_gateway_tool_reference_converted_total", "WITNESSED (fak authored): cumulative Claude-Code-internal tool_reference content blocks rewritten into wire-valid text blocks across the session. tool_reference is not a valid Anthropic tool_result.content type; converting it in place keeps a ToolSearch-bearing body from being 400'd by the API as malformed.", int64(refConverted))
	writeCounter(b, "fak_gateway_tool_reference_sanitize_turns_total", "WITNESSED (fak authored): turns on which at least one tool_reference block was converted. Zero on a harness that never replays tool-discovery results into a tool_result; nonzero means fak repaired a body the provider would otherwise have 400'd as malformed.", int64(refTurns))

	// The general-form EMPTY-CONTENT GATE family (#3118): the residual backstop to the
	// tool_reference sanitizer above. WITNESSED: how many tool_result.content arrays fak found
	// EMPTY (for ANY reason) and backfilled with a placeholder text block, since an empty content
	// array is itself a 400. Nonzero means fak repaired a body a future client-internal block type
	// (or a genuinely empty source result) would otherwise have gotten rejected upstream.
	emptyTurns, emptyRepaired := m.emptyContentRepairSnapshot()
	writeCounter(b, "fak_gateway_empty_tool_result_repaired_total", "WITNESSED (fak authored): cumulative empty tool_result.content arrays backfilled with a placeholder text block across the session. The general form of the tool_reference sanitizer — it catches any content array that ended up empty for ANY reason, since an empty array itself 400s as malformed.", int64(emptyRepaired))
	writeCounter(b, "fak_gateway_empty_tool_result_repair_turns_total", "WITNESSED (fak authored): turns on which at least one empty tool_result.content array was repaired. Zero on the common turn; nonzero means fak repaired a body the provider would otherwise have 400'd as empty content.", int64(emptyTurns))
}

// writeResetShadowMetrics renders the per-session resetScore SHADOW family (#792). The reset
// policy recommends cut-vs-reset on the compaction crossover; in SHADOW mode it acts on NOTHING,
// so this surface is purely "what it WOULD recommend" — recommendations_total is the count of
// turns the verdict said reset, and it stays a recommendation until shadow evidence supports
// enabling reset. The verdict is WITNESSED (fak's policy); the cache ratios it scored are OBSERVED
// (provider-reported), the same split the compaction family keeps. Every reason bucket is emitted
// at 0 so the panel exists before the first compacted turn.
func (m *gatewayMetrics) writeResetShadowMetrics(b *strings.Builder) {
	snap := m.resetShadowSnapshotData()
	writeHelpType(b, "fak_gateway_compaction_reset_shadow_total",
		"WITNESSED (fak policy, recommend-only): compacted turns scored by the resetScore SHADOW policy, by reason (healthy_cache|stale_prefix|cache_decay|cooldown|unknown_provider). SHADOW mode acts on nothing — this is what the cut-vs-reset crossover WOULD recommend, not a reset that happened. The cache ratios scored are OBSERVED (provider-reported).", "counter")
	for _, reason := range []string{
		string(ResetReasonHealthy), string(ResetReasonStalePrefix), string(ResetReasonDecay),
		string(ResetReasonCooldown), string(ResetReasonUnknown),
	} { // stable order; emit at 0 so the panel exists pre-first-turn
		fmt.Fprintf(b, "fak_gateway_compaction_reset_shadow_total{reason=%q} %d\n", reason, snap.reasons[reason])
	}
	writeCounter(b, "fak_gateway_compaction_reset_recommendations_total",
		"WITNESSED (fak policy, recommend-only): compacted turns whose resetScore SHADOW verdict was ShouldReset. In SHADOW mode NONE were acted on — the count is the reset pressure the policy held back, cut-by-default.", int64(snap.recommend))
	writeHelpType(b, "fak_gateway_compaction_reset_score",
		"The most recent compacted turn's 0..1 resetScore reset-pressure magnitude (0 = clearly keep cutting, 1 = clearly reset). Reported even when the cooldown holds the recommendation, so the building pressure is visible. Advisory: nothing acts on it in SHADOW mode.", "gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_reset_score %s\n", promFloat(snap.lastScore))
}

// writeCacheBreakMetrics renders the per-session cache-break family (#2916): two counter
// families — event count and cold-rebuild token cost — each labeled by the closed cause
// vocabulary (toolset_change/altered_turn/rebuilt_prompt/provider_quirk/unknown) and emitted
// in canonical cause order. Both families ALWAYS declare their HELP/TYPE so a regression gate
// and a dashboard panel exist from the first scrape; a session with no witnessed break renders
// the declarations with NO per-cause sample (a clean zero), which is exactly the empty state a
// gate reads as "no regression". The scrape and the guard exit summary fold the SAME
// cacheBreakReport witnesses, so the two views can never disagree.
func (m *gatewayMetrics) writeCacheBreakMetrics(b *strings.Builder) {
	report := m.cacheBreakReport()
	writeHelpType(b, metrics.CacheBreakEventsMetric,
		"Cache-break events this session, labeled by closed cause (toolset_change/altered_turn/rebuilt_prompt/provider_quirk/unknown). A rise means the warm prompt prefix broke more often mid-conversation.", "counter")
	for _, t := range report.ByCause {
		fmt.Fprintf(b, "%s{cause=\"%s\"} %d\n", metrics.CacheBreakEventsMetric, promQuote(string(t.Cause)), t.Events)
	}
	writeHelpType(b, metrics.CacheBreakCostMetric,
		"Cold-rebuild token cost of cache breaks this session, labeled by closed cause. Each break's cost is the warm prompt prefix that had to be re-prefilled.", "counter")
	for _, t := range report.ByCause {
		fmt.Fprintf(b, "%s{cause=\"%s\"} %d\n", metrics.CacheBreakCostMetric, promQuote(string(t.Cause)), t.CostTokens)
	}
}
