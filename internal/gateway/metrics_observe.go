package gateway

import (
	"sort"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// observeUpstreamError increments the upstream-error counter for the error's KIND. It is the
// single fold point for every proxy/planner error path (called from plannerErrorStatus), so a
// turn failure is counted exactly once. A nil or unclassifiable error is a no-op.
func (m *gatewayMetrics) observeUpstreamError(err error) {
	if m == nil {
		return
	}
	kind := upstreamErrorKind(err)
	if kind == "" {
		return
	}
	m.upstreamErrMu.Lock()
	if m.upstreamErrors == nil {
		m.upstreamErrors = map[string]uint64{}
	}
	m.upstreamErrors[kind]++
	m.upstreamErrMu.Unlock()
}

// observeUpstreamRetry counts one upstream retry attempt (the planner's 429/5xx backoff) and
// accumulates the wait it slept before re-hitting upstream. Atomic and off the request path,
// called from the RetryNotify hook. A non-positive wait still counts the attempt.
func (m *gatewayMetrics) observeUpstreamRetry(wait time.Duration) {
	if m == nil {
		return
	}
	atomic.AddUint64(&m.upstreamRetries, 1)
	if wait > 0 {
		atomic.AddUint64(&m.upstreamRetryWaitNS, uint64(wait))
	}
}

// observeFakVerbCall records one admitted MCP fak-verb (tools/call) invocation. Called
// from the single tools/call choke point after the --expose allow gate, so it counts only
// verbs the operator actually exposed and the client actually called. Atomic and off the
// request-critical path; a nil metrics receiver is a no-op.
func (m *gatewayMetrics) observeFakVerbCall() {
	if m == nil {
		return
	}
	atomic.AddUint64(&m.fakVerbCalls, 1)
}

// fakVerbCallsSnapshot reads the cumulative fak-verb call counter. Used by the metrics
// renderer and available to any read-time fold.
func (m *gatewayMetrics) fakVerbCallsSnapshot() uint64 {
	if m == nil {
		return 0
	}
	return atomic.LoadUint64(&m.fakVerbCalls)
}

// observeUpstreamAuthRefresh counts one 401 token-rotation self-heal by outcome ("recovered" /
// "exhausted"), called from the AuthRefreshNotify hook. Off the request path, guarded by the
// shared upstreamErrMu. An unknown outcome is ignored so a future caller typo cannot create a
// junk series.
func (m *gatewayMetrics) observeUpstreamAuthRefresh(outcome string) {
	if m == nil {
		return
	}
	if outcome != "recovered" && outcome != "exhausted" {
		return
	}
	m.upstreamErrMu.Lock()
	if m.upstreamAuthRefreshes == nil {
		m.upstreamAuthRefreshes = map[string]uint64{}
	}
	m.upstreamAuthRefreshes[outcome]++
	m.upstreamErrMu.Unlock()
}

// observeUpstreamForbiddenRetry counts one 403 transient-recovery outcome ("recovered" /
// "exhausted"), called from the ForbiddenRetryNotify hook. Off the request path, guarded by the
// shared upstreamErrMu. An unknown outcome is ignored so a future caller typo cannot create a
// junk series. Mirrors observeUpstreamAuthRefresh — the two stay parallel so an operator reads
// the 401 and 403 self-heals the same way.
func (m *gatewayMetrics) observeUpstreamForbiddenRetry(outcome string) {
	if m == nil {
		return
	}
	if outcome != "recovered" && outcome != "exhausted" {
		return
	}
	m.upstreamErrMu.Lock()
	if m.upstreamForbiddenRetries == nil {
		m.upstreamForbiddenRetries = map[string]uint64{}
	}
	m.upstreamForbiddenRetries[outcome]++
	m.upstreamErrMu.Unlock()
}

// observeUpstreamAccountFailover counts one account-failover-arm outcome, called from the
// AccountFailoverNotify hook. The arm fires for two causes over the SAME swap mechanism, and the
// outcome label keeps them apart: a 403 ORG WALL reports "recovered" / "exhausted", and a 429
// ACCOUNT CAP (session/weekly/usage) seat rehome reports "rehomed_seat" / "rehome_seat_unavailable"
// — the cap can hold for a hours-away reset, so the session swaps to a free sibling seat instead of
// sleeping on the capped one. Off the request path, guarded by the shared upstreamErrMu. An unknown
// outcome is ignored so a future caller typo cannot create a junk series. Mirrors
// observeUpstreamForbiddenRetry — the four upstream self-heal signals (retry, auth-refresh,
// forbidden-retry, account-failover) stay parallel so an operator reads them alike.
func (m *gatewayMetrics) observeUpstreamAccountFailover(outcome string) {
	if m == nil {
		return
	}
	switch outcome {
	case "recovered", "exhausted", "rehomed_seat", "rehome_seat_unavailable":
	default:
		return
	}
	m.upstreamErrMu.Lock()
	if m.upstreamAccountFailovers == nil {
		m.upstreamAccountFailovers = map[string]uint64{}
	}
	m.upstreamAccountFailovers[outcome]++
	m.upstreamErrMu.Unlock()
}

// recordForbiddenDetail stores a scrubbed, bounded snapshot of a PERSISTENT 403's upstream body
// for the operator-only /debug/vars drilldown. Called from the proxy/planner error path with the
// raw truncated body; scrubForbiddenDetail strips secrets and bounds it before it is stored. A
// body that scrubs to empty leaves the previous detail intact (a blank 403 body should not erase a
// useful earlier one). Off the request path, guarded by the shared upstreamErrMu.
func (m *gatewayMetrics) recordForbiddenDetail(body string) {
	if m == nil {
		return
	}
	scrubbed := scrubForbiddenDetail(body)
	if scrubbed == "" {
		return
	}
	m.upstreamErrMu.Lock()
	m.lastForbiddenDetail = scrubbed
	m.upstreamErrMu.Unlock()
}

// observeInKernelOOM folds a planner error into the local device-OOM visibility family when
// it is either an in-kernel allocation fault or the request-time capacity precheck refusal.
// It returns true only for that local OOM class, so callers can record without re-doing
// errors.As.
func (m *gatewayMetrics) observeInKernelOOM(err error) bool {
	if m == nil || err == nil {
		return false
	}
	class, bytes, site, ok := inKernelOOMObservation(err)
	if !ok {
		return false
	}
	m.oomMu.Lock()
	if m.inKernelOOM == nil {
		m.inKernelOOM = map[string]*inKernelOOMClassStats{}
	}
	st := m.inKernelOOM[class]
	if st == nil {
		st = &inKernelOOMClassStats{}
		m.inKernelOOM[class] = st
	}
	st.count++
	st.failedBytes += bytes
	st.lastFailedBytes = bytes
	st.lastSite = site
	m.oomMu.Unlock()
	return true
}

// observeCompaction records the outcome of one history-compaction attempt. off=true means the
// budget was unset (the lever is configured off); otherwise the outcome's Reason decides fired
// vs bailed and which bail-reason bucket increments.
func (m *gatewayMetrics) observeCompaction(out agent.CompactOutcome, off bool) {
	if m == nil {
		return
	}
	m.compactMu.Lock()
	defer m.compactMu.Unlock()
	switch {
	case off:
		m.compactAttempts["off"]++
	case out.Reason == agent.CompactReasonNone:
		m.compactAttempts["fired"]++
		m.compactDropped += uint64(out.Dropped)
		m.compactShed += uint64(out.ShedTokens)
		if out.SolvencyForced {
			// A subset of "fired": the burst economics refused and the context-solvency floor
			// overrode them. Counted apart so an unprofitable-by-design burst is never booked as
			// a cache win (see AdjudicationSummary.CompactionSolvencyForced).
			m.compactSolvencyForced++
		}
	default:
		// Every non-fire still books "bailed" and its reason bucket, unchanged: those two are
		// live dashboard series and redefining them underneath a running panel is not a fix.
		// The pre-eligibility half is held out of the ALERTABLE rate instead, and it is held out
		// at read time by compactBailPartition below — derived from this same reason tally rather
		// than from a second counter, so the two numbers cannot drift apart (#5443).
		m.compactAttempts["bailed"]++
		m.compactBailReasons[out.Reason]++
		// Headroom witness: the compactible span this bail measured. Recorded for every bail that
		// resolved one (under_budget and burst_unprofitable both carry it; the pre-eligibility
		// bails resolve no span and leave it 0, so they never pull the peak down).
		if out.SuffixTokens > 0 {
			m.compactLastSuffixTokens = uint64(out.SuffixTokens)
			if uint64(out.SuffixTokens) > m.compactPeakSuffixTokens {
				m.compactPeakSuffixTokens = uint64(out.SuffixTokens)
			}
		}
		if out.AnchorStarved {
			// A subset of under_budget: the anchor protected a prefix larger than the budget, so
			// the lever could not fire. Counted apart so "idle" can be proven NOT a short session.
			m.compactAnchorStarved++
		}
	}
}

// recordCompactionThrash books one COMPACTION_THRASH verdict (#2424): a session whose window
// refilled to the compaction limit ctxThrashConsecutiveRefills turns running. Called from
// observeCtxValue the turn the run reaches the line, so the count is sessions-that-thrashed,
// not turns-spent-thrashing. Always on — the STOP that acts on it is opt-in, but a signal an
// operator cannot see is the gap #2424 exists to close. Nil-safe like every sibling recorder.
func (m *gatewayMetrics) recordCompactionThrash() {
	if m == nil {
		return
	}
	m.compactMu.Lock()
	m.compactThrashSessions++
	m.compactMu.Unlock()
}

// compactBailPartition splits a bail-reason tally into the half the compactor decided BEFORE
// any compactible span existed and the half it decided after — agent.CompactBailPreEligible
// owns the classification, so this package never re-types the vocabulary it does not emit.
//
// Both halves are returned. The pre-eligibility count is rendered as its own series rather
// than quietly subtracted: a cell whose bails are almost all non-candidates is a normal
// short-request stream, not a sick compactor, and an operator can only tell those apart if
// the held-out population is visible. Nothing is dropped or capped — every reason in the map
// lands in exactly one of the two returned totals.
//
// An UNRECOGNISED reason counts as a CANDIDATE (agent.CompactBailPreEligible fails open), so
// a reason added upstream without being registered leaves the rate conservatively high
// instead of silently shrinking the measured population.
func compactBailPartition(bailReasons map[string]uint64) (nonCandidateBails, candidateBails uint64) {
	for reason, n := range bailReasons {
		if agent.CompactBailPreEligible(reason) {
			nonCandidateBails += n
			continue
		}
		candidateBails += n
	}
	return nonCandidateBails, candidateBails
}

// compactCandidateBailRate is candidateBails / (fires + candidateBails): declines over
// attempts that were actually ELIGIBLE to compact. It returns 0 when that population is
// empty, so a gateway that never reached the compactor reads an honest zero rather than a
// fabricated ratio.
//
// This is the health read the raw bail rate — bails/(fires+bails) — structurally cannot
// carry (#5443/#5388). The gateway attempts compaction on EVERY Anthropic passthrough once a
// budget is set, so short auxiliary pings and non-JSON bodies flood the raw denominator and
// pin it near 1.0 on healthy and broken traffic alike. The raw rate stays recoverable from
// the unchanged attempts{bailed} counter beside it.
func compactCandidateBailRate(fires, candidateBails uint64) float64 {
	total := fires + candidateBails
	if total == 0 {
		return 0
	}
	return float64(candidateBails) / float64(total)
}

// observeCtxViewRewrite records one successful ctxplan planned-view rewrite. It is
// intentionally separate from observeCompaction: both transforms can shed request-body
// residency, but only compact-history should increment compaction attempts, reset-score
// health, or debug compaction banners.
func (m *gatewayMetrics) observeCtxViewRewrite(out agent.CompactOutcome) {
	if m == nil || out.Reason != agent.CompactReasonNone {
		return
	}
	m.ctxViewMu.Lock()
	m.ctxViewEvents++
	m.ctxViewDropped += uint64(clampNonNeg(out.Dropped))
	m.ctxViewShed += uint64(clampNonNeg(out.ShedTokens))
	m.ctxViewMu.Unlock()
}

// observeCacheTTLUpgrade records one managed-cache 1h TTL upgrade attempt on the outbound
// Anthropic wire, bucketed by the closed agent.TTLUpgradeReason* outcome ("" — an actual
// upgrade — counts as "upgraded"). WITNESSED: fak authored the splice, and the upgrader
// re-proves the body redecodes before returning changed bytes. Called only while the lever
// (--managed-cache / Config.CacheTTL1H) is on, so the family doubles as the lever's
// default-state witness: absent rows mean OFF, a zero "upgraded" row with nonzero reason
// rows means ON-but-ineligible.
func (m *gatewayMetrics) observeCacheTTLUpgrade(reason string) {
	if m == nil {
		return
	}
	outcome := reason
	if outcome == "" {
		outcome = "upgraded"
	}
	m.compactMu.Lock()
	if m.ttlUpgrades == nil {
		m.ttlUpgrades = map[string]uint64{}
	}
	m.ttlUpgrades[outcome]++
	m.compactMu.Unlock()
}

// observePlacement records one offensive cache-breakpoint placement attempt on the outbound
// Anthropic wire, bucketed by the closed agent.BreakpointReason* outcome ("" — an actual placement
// — counts as "placed"). WITNESSED: fak authored the splice, and the placer re-proves the body
// redecodes and the cached prefix stays byte-identical before returning changed bytes. Called on
// every Anthropic passthrough turn while compaction is configured on, so the "placed" bucket is the
// always-on witness of fak-authored cache unlocking even with --debug-stats off — while "already_set"
// counts the Claude-Code shape fak deliberately leaves to the client's own cache.
func (m *gatewayMetrics) observePlacement(out agent.BreakpointOutcome) {
	if m == nil {
		return
	}
	outcome := out.Reason
	if outcome == "" {
		outcome = "placed"
	}
	m.compactMu.Lock()
	if m.placementAttempts == nil {
		m.placementAttempts = map[string]uint64{}
	}
	m.placementAttempts[outcome]++
	// #3622: fold the same outcome through the LIVE monitor. The counter above is cumulative and
	// unwatched — a head that turns volatile mid-session just stops incrementing "placed", which
	// reads exactly like a session that went quiet — so the rolling refused fraction is what turns
	// the tally into a signal. Lazily built for a directly-constructed gatewayMetrics (the same
	// posture inferE2EHist takes), so a Server assembled without newGatewayMetrics still monitors.
	if m.anchorMon == nil {
		m.anchorMon = metrics.NewAnchorRefusalMonitor(metrics.AnchorRefusalThresholds{})
	}
	m.anchorMon.Observe(outcome)
	m.compactMu.Unlock()
}

// anchorRefusalReport folds the session's placement-outcome mix into the operator readout the
// #3622 alarm is carried on. Returns nil — leaving the summary's JSON field absent — until the
// session has actually attempted a placement, so a cold or non-Anthropic session reports nothing
// rather than a fabricated 0% refusal it never measured. Taken under compactMu, the same lock
// observePlacement writes the monitor under, so the report can never tear mid-fold.
func (m *gatewayMetrics) anchorRefusalReport() *metrics.AnchorRefusalReport {
	if m == nil {
		return nil
	}
	m.compactMu.Lock()
	defer m.compactMu.Unlock()
	if m.anchorMon == nil {
		return nil
	}
	r := m.anchorMon.Report()
	if r.Turns == 0 {
		return nil
	}
	return &r
}

// observeUncachedTrim records oversized-result elision as a fak-authored uncached-token saving.
// It does not increment the history-compaction attempt counters: the transform is a sibling
// shrinker, not a whole-turn compaction fire. Its shed tokens are still folded into compactShed so
// the existing owner/mechanism attribution reports them under owner="fak"/compaction_shed.
func (m *gatewayMetrics) observeUncachedTrim(out agent.ElideOutcome) {
	if m == nil || out.Reason != agent.ElideReasonNone || out.ShedBytes <= 0 {
		return
	}
	tokens := estimatedTokensFromBytes(out.ShedBytes)
	if tokens == 0 {
		return
	}
	m.compactMu.Lock()
	m.uncachedTrimResults += uint64(out.Elided)
	m.uncachedTrimShed += tokens
	m.compactShed += tokens
	m.compactMu.Unlock()
}

// recordCompactionCacheRead records the provider's cache_read_input_tokens on a turn whose body
// WAS compacted. This is an OBSERVED value relayed verbatim from the upstream, NOT a fak claim:
// fak's own guarantee (the protected prefix it shipped was byte-identical) is already witnessed
// by the turn counting `fired` with no `prefix_mismatch` bail. A low cache_read here therefore
// does not mean fak broke anything — pair it with shed_tokens to see the net effect, and read a
// crater as a provider-side miss (TTL expiry / eviction / the client moving its breakpoint)
// unless bail_reason{prefix_mismatch} is nonzero.
func (m *gatewayMetrics) recordCompactionCacheRead(cacheRead int) {
	if m == nil {
		return
	}
	m.compactMu.Lock()
	m.compactCacheReads += uint64(cacheRead)
	m.compactLastCacheRd = float64(cacheRead)
	m.compactMu.Unlock()
}

// observeInboundToolPrune records that a turn pruned n unreachable tool definitions from the
// outbound tools[] (the INBOUND tool-floor prune lever). n<=0 is a no-op, so the common turn —
// where no advertised tool is floor-denied past the cache_control breakpoint — records nothing,
// exactly as a clean compaction turn does. WITNESSED: fak chose what to drop, and the pruner
// proved the cached prefix stayed byte-identical before returning Changed=true, so a counted
// prune never bursts the upstream cache.
func (m *gatewayMetrics) observeInboundToolPrune(n int) {
	if m == nil || n <= 0 {
		return
	}
	m.toolPruneMu.Lock()
	m.toolPruneTurns++
	m.toolPruneCount += uint64(n)
	m.toolPruneMu.Unlock()
}

// observeToolDefer records a tool-deferral turn (#3232, the 10x floor lever): cold is
// the number of cold defs marked defer_loading this turn, fired reports whether the
// transform changed the body, and names are the deferred custom tool names (#3647). A
// no-op turn (fired=false, or nothing cold) records nothing — the same posture as
// observeInboundToolPrune, so an operator can tell a turn that deferred the cold tail
// from one that stood down. names fold into a distinct set so the deterministic per-turn
// defer of the same tail does not multiply-list a tool.
func (m *gatewayMetrics) observeToolDefer(cold int, fired bool, names ...string) {
	if m == nil || !fired || cold <= 0 {
		return
	}
	m.deferMu.Lock()
	m.deferFiredTurns++
	m.deferColdCount += uint64(cold)
	for _, n := range names {
		if n == "" {
			continue
		}
		if m.deferColdNames == nil {
			m.deferColdNames = map[string]struct{}{}
		}
		m.deferColdNames[n] = struct{}{}
	}
	m.deferMu.Unlock()
}

// observeToolDeferStandDown records a defer-ELIGIBLE turn on which the transform ran and stood
// down to byte-identity (#3621), keyed by the deferResult reason. This is the DENOMINATOR
// observeToolDefer above deliberately never kept: it books only fired turns, so a session whose
// every turn stands down looks identical to one where the lever was never armed at all — the
// silent-identity blind spot (a wrong dated tool_search_tool type, an already-deferred client
// body, an all-hot surface). Callers must invoke it only PAST maybeDeferColdTools' eligibility
// gate (lever on, Anthropic passthrough wire, ablation arm off), which is what lets the
// DEFER_ENABLED_BUT_INERT watchdog raise from these counters alone with no posture flag threaded
// in. An empty reason books as "unknown" rather than an unnamed bucket.
func (m *gatewayMetrics) observeToolDeferStandDown(reason string) {
	if m == nil {
		return
	}
	if reason == "" {
		reason = "unknown"
	}
	m.deferMu.Lock()
	m.deferStandDownTurns++
	if m.deferStandDownReasons == nil {
		m.deferStandDownReasons = map[string]uint64{}
	}
	m.deferStandDownReasons[reason]++
	m.deferMu.Unlock()
}

// toolDeferStandDownSnapshot reads the stand-down denominator (#3621): eligible turns that stood
// down, plus a COPY of the per-reason breakdown so a caller can never race the live map. reasons
// is nil until something stands down, keeping the summary's JSON field absent on a clean session.
func (m *gatewayMetrics) toolDeferStandDownSnapshot() (turns uint64, reasons map[string]uint64) {
	if m == nil {
		return 0, nil
	}
	m.deferMu.Lock()
	defer m.deferMu.Unlock()
	if len(m.deferStandDownReasons) == 0 {
		return m.deferStandDownTurns, nil
	}
	out := make(map[string]uint64, len(m.deferStandDownReasons))
	for r, n := range m.deferStandDownReasons {
		out[r] = n
	}
	return m.deferStandDownTurns, out
}

// toolDeferSnapshot reads the deferral accumulators (turns, cold-def count).
func (m *gatewayMetrics) toolDeferSnapshot() (turns, cold uint64) {
	if m == nil {
		return 0, 0
	}
	m.deferMu.Lock()
	defer m.deferMu.Unlock()
	return m.deferFiredTurns, m.deferColdCount
}

// toolDeferNamesSnapshot returns the DISTINCT deferred custom tool names, sorted for a
// stable operator render (#3647). Nil when the lever never fired.
func (m *gatewayMetrics) toolDeferNamesSnapshot() []string {
	if m == nil {
		return nil
	}
	m.deferMu.Lock()
	defer m.deferMu.Unlock()
	if len(m.deferColdNames) == 0 {
		return nil
	}
	out := make([]string, 0, len(m.deferColdNames))
	for n := range m.deferColdNames {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// observeInboundPrunedToolProposal records that a model later proposed a tool name fak had
// removed from that trace's advertised tools[]. The caller de-duplicates per trace/tool; this
// counter is the process-wide witness count. It is names-only observability, not a verdict.
func (m *gatewayMetrics) observeInboundPrunedToolProposal(n int) {
	if m == nil || n <= 0 {
		return
	}
	m.toolPruneMu.Lock()
	m.toolPrunedPropose += uint64(n)
	m.toolPruneMu.Unlock()
}

// inboundToolPruneSnapshot reads the WITNESSED tool-prune accumulators under their lock. Pure
// read — the exit summary and the Prometheus surface both fold the same two numbers, so the line
// can never disagree with the scrape.
func (m *gatewayMetrics) inboundToolPruneSnapshot() (turns, count uint64) {
	if m == nil {
		return 0, 0
	}
	m.toolPruneMu.Lock()
	defer m.toolPruneMu.Unlock()
	return m.toolPruneTurns, m.toolPruneCount
}

func (m *gatewayMetrics) inboundPrunedToolProposalSnapshot() uint64 {
	if m == nil {
		return 0
	}
	m.toolPruneMu.Lock()
	defer m.toolPruneMu.Unlock()
	return m.toolPrunedPropose
}

// observeToolRefSanitize records that a turn converted one or more Claude-Code-internal
// tool_reference blocks into wire-valid text blocks. A body with no tool_reference records nothing
// (out.Reason != ToolRefReasonNone), so the common turn is silent — exactly as a clean prune is.
// WITNESSED: fak authored the rewrite that kept the body from being 400'd upstream.
func (m *gatewayMetrics) observeToolRefSanitize(out agent.ToolRefOutcome) {
	if m == nil || out.Reason != agent.ToolRefReasonNone || out.Converted <= 0 {
		return
	}
	m.toolRefMu.Lock()
	m.toolRefTurns++
	m.toolRefConverted += uint64(out.Converted)
	m.toolRefMu.Unlock()
}

// toolRefSanitizeSnapshot reads the WITNESSED tool_reference-sanitize accumulators under their
// lock. Pure read — the exit summary and the Prometheus surface fold the same two numbers.
func (m *gatewayMetrics) toolRefSanitizeSnapshot() (turns, converted uint64) {
	if m == nil {
		return 0, 0
	}
	m.toolRefMu.Lock()
	defer m.toolRefMu.Unlock()
	return m.toolRefTurns, m.toolRefConverted
}

// observeEmptyContentRepair records that a turn backfilled one or more empty tool_result.content
// arrays with a placeholder text block (#3118). A body with no empty content array records nothing
// (out.Reason != EmptyContentReasonNone), so the common turn is silent. WITNESSED: fak authored the
// repair that kept the body from being 400'd upstream as "empty content".
func (m *gatewayMetrics) observeEmptyContentRepair(out agent.EmptyContentOutcome) {
	if m == nil || out.Reason != agent.EmptyContentReasonNone || out.Repaired <= 0 {
		return
	}
	m.emptyContentMu.Lock()
	m.emptyContentTurns++
	m.emptyContentRepaired += uint64(out.Repaired)
	m.emptyContentMu.Unlock()
}

// emptyContentRepairSnapshot reads the WITNESSED empty-content-gate accumulators under their lock.
func (m *gatewayMetrics) emptyContentRepairSnapshot() (turns, repaired uint64) {
	if m == nil {
		return 0, 0
	}
	m.emptyContentMu.Lock()
	defer m.emptyContentMu.Unlock()
	return m.emptyContentTurns, m.emptyContentRepaired
}

// recordResetShadow folds one compacted turn's resetScore SHADOW verdict into the recommend-only
// accumulators. The reset policy NEVER acts in shadow mode (reset_shadow.go); this only counts what
// it WOULD recommend, bucketed by the closed ResetReason, so an operator can watch the cut-vs-reset
// pressure build before reset is ever enabled. The verdict is WITNESSED (fak's own policy); the
// inputs it scored are OBSERVED (provider cache counters). Lazily inits the reason map like
// observeInference, so a Server built without this family present still records correctly.
func (m *gatewayMetrics) recordResetShadow(d ResetDecision) {
	if m == nil {
		return
	}
	m.resetShadowMu.Lock()
	if m.resetShadowReasons == nil {
		m.resetShadowReasons = map[string]uint64{}
	}
	m.resetShadowReasons[string(d.Reason)]++
	if d.ShouldReset {
		m.resetShadowRecommend++
	}
	m.resetShadowLastScore = d.Score
	m.resetShadowMu.Unlock()
}

// recordAdjudicationOutcome folds one served turn's adjudication SHAPE into two
// separate turn-control signals:
//   - deny-all: every proposed tool call was hard-refused by the floor. This is the
//     bounded stop-policy signal; the guard Stop hook may eventually give up.
//   - tool-feedback: every proposed tool call was rejected as retryable/model-fixable
//     feedback (for example MALFORMED JSON). This should continue the turn but must not
//     be counted as a session-stop/give-up policy.
//
// A reset turn (at least one survivor/served call, or pure text) clears both consecutive
// runs. Called once per served Anthropic turn on both wire paths. A no-op for nil metrics.
//
// fingerprint is the deny-all turn's same-issue identity (denyAllFingerprint) and is consulted
// ONLY on a deny-all turn: it advances denyAllSameConsecutive when it matches the previous
// deny-all's fingerprint, and re-seeds it to 1 when the refused tool/reason changes — so a
// session re-proposing the identical refused action climbs while a varied one stays pinned at 1.
// An empty fingerprint fails open (re-seeds to 1) so an unidentifiable turn never accumulates
// toward a stop. It is ignored on tool-feedback/reset turns (they clear the same-issue run).
func (m *gatewayMetrics) recordAdjudicationOutcome(signal adjudicationOutcomeSignal, fingerprint string) {
	if m == nil {
		return
	}
	m.denyAllMu.Lock()
	switch signal {
	case adjudicationOutcomeDenyAll:
		m.denyAllStops++
		m.denyAllConsecutive++
		if fingerprint != "" && fingerprint == m.denyAllFingerprint {
			m.denyAllSameConsecutive++
		} else {
			m.denyAllSameConsecutive = 1
		}
		m.denyAllFingerprint = fingerprint
		m.toolFeedbackConsecutive = 0
	case adjudicationOutcomeToolFeedback:
		m.toolFeedbackTurns++
		m.toolFeedbackConsecutive++
		m.denyAllConsecutive = 0
		m.denyAllSameConsecutive = 0
		m.denyAllFingerprint = ""
	default:
		m.denyAllConsecutive = 0
		m.denyAllSameConsecutive = 0
		m.denyAllFingerprint = ""
		m.toolFeedbackConsecutive = 0
	}
	m.denyAllMu.Unlock()
}

// recordServedInline attributes n vDSO served-inline hits (a read-only call answered
// locally on a served turn, folded to assistant text, dropped before the client could
// re-run it) to the gateway seam. Atomic, off any lock — bumped once per served turn
// from every wire's adjudication handler. A no-op for a nil metrics.
func (m *gatewayMetrics) recordServedInline(n int) {
	if m == nil || n <= 0 {
		return
	}
	atomic.AddUint64(&m.servedInline, uint64(n))
}

// servedInlineSnapshot reads the cumulative served-inline count. Pure read.
func (m *gatewayMetrics) servedInlineSnapshot() uint64 {
	if m == nil {
		return 0
	}
	return atomic.LoadUint64(&m.servedInline)
}

// recordCacheBreak witnesses one cache-break event (#2916): a warm prompt prefix
// that broke mid-conversation under the closed cause vocabulary, and the cold-rebuild
// token cost that break caused (the prefix that must now be re-prefilled). This is the
// CONSUMER seam sibling #2915's prefix-mutation detector calls; the cause and cost are
// caller-supplied from cache accounting and normalized by internal/metrics (an
// out-of-vocabulary cause folds to unknown, a negative cost clamps to zero). A nil
// metrics is a no-op. Off the hot path — appended under its own lock, folded only at
// scrape / exit-summary time.
func (m *gatewayMetrics) recordCacheBreak(cause metrics.CacheBreakCause, costTokens int64) {
	if m == nil {
		return
	}
	m.cacheBreakMu.Lock()
	m.cacheBreakEvents = append(m.cacheBreakEvents, metrics.WitnessCacheBreak(cause, costTokens))
	m.cacheBreakMu.Unlock()
}

// cacheBreakReport folds the session's witnessed cache-break events into the
// per-session operator readout (#2916): total events, total cold-rebuild token cost,
// and the per-cause tally in canonical order. The guard exit summary and the Prometheus
// surface fold the SAME witnesses, so the two views can never disagree. An empty sink
// folds to a clean zero. Pure read — snapshots under the lock, folds outside it.
func (m *gatewayMetrics) cacheBreakReport() metrics.CacheBreakReport {
	if m == nil {
		return metrics.CacheBreakReport{}
	}
	m.cacheBreakMu.Lock()
	events := append([]metrics.CacheBreakEvent(nil), m.cacheBreakEvents...)
	m.cacheBreakMu.Unlock()
	return metrics.FoldCacheBreaks(events)
}

// denyAllSnapshot reads the deny-all accumulators under their lock. Pure read — the exit
// summary, the Prometheus surface, and the guard Stop-hook gauge all fold the same two
// numbers, so the views can never disagree.
func (m *gatewayMetrics) denyAllSnapshot() (stops, consecutive uint64) {
	if m == nil {
		return 0, 0
	}
	m.denyAllMu.Lock()
	defer m.denyAllMu.Unlock()
	return m.denyAllStops, m.denyAllConsecutive
}

// denyAllSameSnapshot reads the same-issue consecutive run under the deny-all lock: consecutive
// deny-all turns proposing the IDENTICAL refused action (same tool+reason). This is the signal
// the guard Stop hook keys its bounded give-up on — distinct from the blind denyAllConsecutive,
// which stays for observability. Pure read.
func (m *gatewayMetrics) denyAllSameSnapshot() uint64 {
	if m == nil {
		return 0
	}
	m.denyAllMu.Lock()
	defer m.denyAllMu.Unlock()
	return m.denyAllSameConsecutive
}

// toolFeedbackSnapshot reads the retryable tool-feedback accumulators under the same
// lock as deny-all. The two families are mutually exclusive on a given turn but rendered
// side by side so operators can tell "the model needs to fix JSON/args" apart from "the
// floor hard-refused every action."
func (m *gatewayMetrics) toolFeedbackSnapshot() (turns, consecutive uint64) {
	if m == nil {
		return 0, 0
	}
	m.denyAllMu.Lock()
	defer m.denyAllMu.Unlock()
	return m.toolFeedbackTurns, m.toolFeedbackConsecutive
}

// observeInference records one served model-generation turn: its disjoint uncached
// input, cached input, output, and cache-write accounting; why decode stopped; and the
// wall-clock the planner spent producing it. Negative/zero values are ignored so a
// planner that omits a count never corrupts the running totals. Provider-shaped Usage
// must enter through observeInferenceUsageServed, which normalizes providers such as
// Codex/OpenAI whose input_tokens already includes cached_input_tokens. Direct callers
// supply the already-disjoint axes used by tests and internal vcache folds. This is the
// signal that makes a busy gateway look busy: fak_kernel_*/fak_vdso_* stay 0 on a pure
// chat workload (no syscall, no fast-path lookup), so without this family every panel
// reads 0 while the box is in fact decoding tokens.
func (m *gatewayMetrics) observeInference(promptTok, complTok, cachedTok, cacheCreateTok int, finishReason string, dur time.Duration) {
	// A buffered turn cannot observe the first-token boundary, so prefill is "not
	// measured": ttft<=0 routes the whole duration into the decode-total accumulator
	// and leaves the prefill split untouched (it stays an honest 0, never a phantom).
	m.observeInferenceTimed(promptTok, complTok, cachedTok, cacheCreateTok, finishReason, dur, 0)
}

// observeInferenceUsageServed is the provider-Usage ingestion seam for the cumulative
// inference counters that back /debug/vars and `fak info`. Usage keeps the provider's
// wire semantics: Codex/OpenAI input_tokens includes cached_input_tokens, while
// Anthropic input_tokens is already the uncached remainder. Normalize exactly here so
// inferPromptTokens and inferCachedTokens remain disjoint without changing Usage,
// response forwarding, context-window accounting, or the internal vcache row contract.
func (m *gatewayMetrics) observeInferenceUsageServed(loc servingLocality, usage agent.Usage, finishReason string, dur time.Duration) {
	m.observeInferenceServed(loc,
		usage.UncachedPromptTokens(),
		usage.CompletionTokens,
		usage.CachedPromptTokens(),
		usage.CacheCreationInputTokens,
		finishReason,
		dur,
	)
}

// observeInferenceServed is observeInference plus WHO SERVED THE TURN — the
// self-hosted attribution the durable usage row needs. loc is what servedLocality
// resolved; localityUnknown counts the turn in the unsplit totals and in neither
// side, which is what makes the classified fraction a measurement rather than an
// assumption.
//
// It is a separate entry point rather than an extra parameter on observeInference so
// that every existing caller — including any the fleet adds while this is landing —
// keeps reporting an honest unknown instead of being silently defaulted into one
// side by a zero argument.
func (m *gatewayMetrics) observeInferenceServed(loc servingLocality, promptTok, complTok, cachedTok, cacheCreateTok int, finishReason string, dur time.Duration) {
	m.observeInferenceServedTimed(loc, promptTok, complTok, cachedTok, cacheCreateTok, finishReason, dur, 0)
}

// observeInferenceServedTimed is observeInferenceTimed with the serving side.
func (m *gatewayMetrics) observeInferenceServedTimed(loc servingLocality, promptTok, complTok, cachedTok, cacheCreateTok int, finishReason string, dur, ttft time.Duration) {
	m.observeInferenceTimed(promptTok, complTok, cachedTok, cacheCreateTok, finishReason, dur, ttft)
	m.attributeServedTurn(loc, promptTok, complTok)
}

// attributeServedTurn books one served turn's volume against the side that
// generated it. Clamps negatives out the same way the unsplit accumulators do, so a
// provider that reports a nonsense count cannot make a group exceed the total it is
// a subset of.
func (m *gatewayMetrics) attributeServedTurn(loc servingLocality, promptTok, complTok int) {
	if m == nil || loc == localityUnknown {
		return
	}
	if promptTok < 0 {
		promptTok = 0
	}
	if complTok < 0 {
		complTok = 0
	}
	m.inferenceMu.Lock()
	switch loc {
	case localitySelfHosted:
		m.inferSelfHostedTurns++
		m.inferSelfHostedPromptTokens += uint64(promptTok)
		m.inferSelfHostedComplTokens += uint64(complTok)
	case localityVendor:
		m.inferVendorTurns++
		m.inferVendorPromptTokens += uint64(promptTok)
		m.inferVendorComplTokens += uint64(complTok)
	}
	m.inferenceMu.Unlock()
}

// observeInferenceTimed is observeInference with an explicit time-to-first-token
// split. ttft is the wall-clock from the planner call starting to the FIRST content
// delta arriving (the prefill phase: prompt ingest + first token); dur is the whole
// turn. When ttft is in (0, dur] the turn's time is split — prefill = ttft, decode =
// dur-ttft — and the turn is counted toward the TTFT denominator so the prefill rate
// divides streamed prefill tokens by only the turns that measured them. When ttft<=0
// (a buffered turn, or a stream that produced no delta), the whole duration counts as
// decode total and the prefill split is left untouched. inferDecodeSecs stays the
// FULL inference wall-clock in both cases so the existing output_tokens_per_second and
// the fleet-value agent-seconds denominator are byte-identical to before.
func (m *gatewayMetrics) observeInferenceTimed(promptTok, complTok, cachedTok, cacheCreateTok int, finishReason string, dur, ttft time.Duration) {
	if m == nil {
		return
	}
	if finishReason == "" {
		finishReason = "unknown"
	}
	m.inferenceMu.Lock()
	if m.inferReqs == nil {
		m.inferReqs = map[string]uint64{}
	}
	if m.inferE2EHist == nil { // a directly-constructed metrics (no newGatewayMetrics)
		m.inferTTFTHist = newLatencyCounter()
		m.inferTPOTHist = newLatencyCounter()
		m.inferE2EHist = newLatencyCounter()
	}
	m.inferReqs[finishReason]++
	if promptTok > 0 {
		m.inferPromptTokens += uint64(promptTok)
	}
	if complTok > 0 {
		m.inferComplTokens += uint64(complTok)
	}
	if cachedTok > 0 {
		m.inferCachedTokens += uint64(cachedTok)
		m.inferCachedHits++ // this turn got a provider prompt-cache READ
	}
	if cacheCreateTok > 0 {
		m.inferCacheCreationTokens += uint64(cacheCreateTok) // this turn WROTE the provider cache
	}
	if dur > 0 {
		m.inferDecodeSecs += dur.Seconds()
		// e2e distribution: every served turn (buffered or streamed) lands here.
		m.inferE2EHist.observe(dur.Seconds())
	}
	// Split prefill from decode only when TTFT was actually observed and is sane
	// (positive and within the total). A clamp guards against a clock skew producing
	// ttft>dur, which would otherwise make decode-seconds negative.
	if ttft > 0 && dur > 0 {
		pre := ttft
		if pre > dur {
			pre = dur
		}
		m.inferPrefillSecs += pre.Seconds()
		m.inferTTFTTurns++
		// ttft distribution: only the streamed turns whose prefill boundary is observable.
		m.inferTTFTHist.observe(pre.Seconds())
		if promptTok > 0 {
			m.inferPrefillPromptTokens += uint64(promptTok)
		}
		decodeSecs := (dur - pre).Seconds()
		m.inferMeasuredDecodeSecs += decodeSecs
		if complTok > 0 {
			m.inferMeasuredComplTokens += uint64(complTok)
			// tpot (inter-token) distribution: mean per-output-token latency for this
			// turn = decode wall-clock / generated tokens.
			m.inferTPOTHist.observe(decodeSecs / float64(complTok))
		}
	}
	m.inferenceMu.Unlock()
}

// recordCacheCreationTierSplit attributes `cacheCreateTok` cache-creation tokens to
// the managed-cache 1h tier when `upgraded` is true (the session's TTL-upgrade rung
// was already active for this turn), else leaves them counted only in the unsplit
// inferCacheCreationTokens total — the same conservative "priced at 5m" convention
// MechanismSavings/ProviderCacheNetSavings apply to any unattributed write (#2179).
func cacheCreationSpanLabel(cacheCreateTok int, upgraded, messagePrefix bool) string {
	if cacheCreateTok <= 0 {
		return "none"
	}
	if !upgraded {
		return "head_5m"
	}
	if messagePrefix {
		return "message_prefix_1h"
	}
	return "head_1h"
}

func (m *gatewayMetrics) recordCacheCreationTierSplit(cacheCreateTok int, upgraded, messagePrefix bool) {
	if m == nil || cacheCreateTok <= 0 {
		return
	}
	m.inferenceMu.Lock()
	defer m.inferenceMu.Unlock()
	n := uint64(cacheCreateTok)
	m.inferCacheCreationTokensTierObserved += n
	if !upgraded {
		return
	}
	m.inferCacheCreationTokensUpgraded += n
	if messagePrefix {
		m.inferCacheCreationTokensMessagePrefix += n
	} else {
		m.inferCacheCreationTokensHeadOnly += n
	}
}

func (s *Server) observePlannerRequestMemory() {
	if s == nil || s.metrics == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.RequestMemoryReporter)
	if !ok {
		return
	}
	s.metrics.observeRequestMemory(reporter.RequestMemoryStats())
}

func (m *gatewayMetrics) observeRequestMemory(st agent.RequestMemoryStats) {
	if m == nil || !st.Observed {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	planRows := requestMemoryPlanByClassScopeDType(st.MemoryPlan)
	fitRows := requestMemoryFitRows(st.MemoryPlan, st.Capacities, st.HeadroomRatio)

	m.reqMemoryMu.Lock()
	if m.reqMemoryObserved == nil {
		m.reqMemoryObserved = map[string]uint64{}
	}
	if m.reqMemoryPlan == nil {
		m.reqMemoryPlan = map[requestMemoryMetricKey]*requestMemoryMetricStats{}
	}
	if m.reqMemoryTokens == nil {
		m.reqMemoryTokens = map[requestMemoryTokenKey]*requestMemoryTokenStats{}
	}
	if m.reqMemoryFit == nil {
		m.reqMemoryFit = map[requestMemoryFitKey]*requestMemoryFitStats{}
	}
	m.reqMemoryObserved[backend]++
	for _, row := range planRows {
		k := requestMemoryMetricKey{backend: backend, class: row.Class, scope: row.Scope, dtype: row.DType}
		st := m.reqMemoryPlan[k]
		if st == nil {
			st = &requestMemoryMetricStats{}
			m.reqMemoryPlan[k] = st
		}
		st.observations++
		st.totalBytes = addPositiveInt64ToUint64(st.totalBytes, row.Bytes)
		if row.Bytes > st.highWaterBytes {
			st.highWaterBytes = row.Bytes
		}
	}
	m.observeRequestMemoryTokenLocked(backend, "prompt", st.PromptTokens)
	m.observeRequestMemoryTokenLocked(backend, "max_new", st.MaxNewTokens)
	m.observeRequestMemoryTokenLocked(backend, "planned", st.PlannedTokens)
	for _, row := range fitRows {
		k := requestMemoryFitKey{backend: backend, scope: row.Scope}
		st := m.reqMemoryFit[k]
		if st == nil {
			st = &requestMemoryFitStats{}
			m.reqMemoryFit[k] = st
		}
		st.observations++
		if row.WantBytes > st.wantHighWater {
			st.wantHighWater = row.WantBytes
		}
		if row.CapacityKnown && (!st.marginKnown || row.MarginBytes < st.marginLowWater) {
			st.marginKnown = true
			st.marginLowWater = row.MarginBytes
		}
	}
	m.reqMemoryMu.Unlock()
}

func (m *gatewayMetrics) observeRequestMemoryTokenLocked(backend, kind string, value int) {
	if value < 0 {
		return
	}
	k := requestMemoryTokenKey{backend: backend, kind: kind}
	st := m.reqMemoryTokens[k]
	if st == nil {
		st = &requestMemoryTokenStats{}
		m.reqMemoryTokens[k] = st
	}
	st.observations++
	st.total = addPositiveIntToUint64(st.total, value)
	if value > st.highWater {
		st.highWater = value
	}
}

func (c *latencyCounter) observe(seconds float64) {
	c.count++
	c.sum += seconds
	for i, le := range gatewayLatencyBuckets {
		if seconds <= le {
			c.buckets[i]++
		}
	}
}

// observeToolFilter records one tools/list projection after it is actually served.
func (m *gatewayMetrics) observeToolFilter(status MCPToolFilterStatus) {
	if m == nil || status.Mode != "active" || status.SavedBytes <= 0 || status.ToolsBefore <= status.ToolsAfter {
		return
	}
	m.toolFilterMu.Lock()
	defer m.toolFilterMu.Unlock()
	m.toolFilterEvents++
	m.toolFilterTools += uint64(status.ToolsBefore - status.ToolsAfter)
	m.toolFilterBytes += uint64(status.SavedBytes)
	m.toolFilterTokens += uint64((status.SavedBytes + estBytesPerToken - 1) / estBytesPerToken)
}

func (m *gatewayMetrics) toolFilterSnapshot() (events, tools, bytes, tokens uint64) {
	if m == nil {
		return 0, 0, 0, 0
	}
	m.toolFilterMu.Lock()
	defer m.toolFilterMu.Unlock()
	return m.toolFilterEvents, m.toolFilterTools, m.toolFilterBytes, m.toolFilterTokens
}

// observeStaleElide records one eligible stale-read elision attempt using only
// bounded reason enums and aggregate counts. No path, content, or trace is kept.
func (m *gatewayMetrics) observeStaleElide(reason string, elided, shedBytes, shedTokens int) {
	if m == nil {
		return
	}
	m.staleElideMu.Lock()
	defer m.staleElideMu.Unlock()
	if reason == agent.StaleReasonNone && elided > 0 && shedBytes > 0 {
		m.staleElideTurns++
		m.staleElideReads += uint64(elided)
		m.staleElideBytes += uint64(shedBytes)
		m.staleElideTokens += uint64(max(shedTokens, 0))
		return
	}
	if reason == "" {
		reason = "identity"
	}
	if m.staleElideBails == nil {
		m.staleElideBails = make(map[string]uint64)
	}
	m.staleElideBails[reason]++
}

func (m *gatewayMetrics) staleElideSnapshot() (turns, reads, bytes, tokens uint64, bails map[string]uint64) {
	if m == nil {
		return 0, 0, 0, 0, nil
	}
	m.staleElideMu.Lock()
	defer m.staleElideMu.Unlock()
	bails = make(map[string]uint64, len(m.staleElideBails))
	for reason, count := range m.staleElideBails {
		bails[reason] = count
	}
	return m.staleElideTurns, m.staleElideReads, m.staleElideBytes, m.staleElideTokens, bails
}
