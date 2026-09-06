// Compact-report pressure folding: the behaviour/confusion aggregates and the
// recommendation lenses BuildCompactReport consults once the per-session rows are in
// hand. Split out of sessionaudit.go along that concern seam to keep the audit core
// under its god-file ceiling (internal/godfileceiling).
package sessionaudit

import (
	"fmt"

	"sort"

	"strings"
)

const (
	longContextPressureTokens  = int64(20_000_000)
	longContextPressureIORatio = 200.0
)

func compactRecommendations(rep CompactReport) []CompactRecommendation {
	var out []CompactRecommendation
	if rec, ok := compactOpusCostPressure(rep.Tiers); ok {
		out = append(out, rec)
	}
	if rec, ok := compactLongContextPressure(rep.TopLongContext); ok {
		out = append(out, rec)
	}
	if rec, ok := compactProcessIssuePressure(rep.Behavior); ok {
		out = append(out, rec)
	}
	if rec, ok := compactConfusionPressure(rep.Confusion); ok {
		out = append(out, rec)
	}
	if rec, ok := compactUnpricedHold(rep.Totals); ok {
		out = append(out, rec)
	}
	if rec, ok := compactShellFrictionPressure(rep.ShellChoice); ok {
		out = append(out, rec)
	}
	return out
}

// compactUnpricedHold raises the gate-able UNKNOWN-cost hold (#4635) whenever
// the window billed a model with no pricing provenance (cost UNKNOWN, held —
// never $0) or a Claude-family spelling that was neighbor-priced without a
// canonical identity. High severity: either way EstimatedCostUSD is not yet
// evidence-grade for the window, which is precisely what a cost gate must see.
func compactUnpricedHold(t CompactTotals) (CompactRecommendation, bool) {
	if len(t.UnpricedModels) == 0 && len(t.UnverifiedClaudeIDs) == 0 {
		return CompactRecommendation{}, false
	}
	return compactRecommendation(
		"unpriced_model_hold",
		"high",
		"pin each held model id in the canonical identity table (or its published rate card) before trusting EstimatedCostUSD for this window",
		"billed model ids without pricing provenance are HELD as UNKNOWN cost, never reported as $0 or a neighboring model's price",
		fmt.Sprintf("unpriced=%s unverified_claude=%s",
			strings.Join(t.UnpricedModels, ","), strings.Join(t.UnverifiedClaudeIDs, ",")),
	), true
}

func compactOpusCostPressure(tiers []CompactTier) (CompactRecommendation, bool) {
	opus, hasOpus := compactTierByName(tiers, "opus")
	fable, hasFable := compactTierByName(tiers, "fable")
	if !hasOpus || opus.EstimatedCostUSD <= 0 || opus.CostShare < 0.5 {
		return CompactRecommendation{}, false
	}
	if hasFable && fable.OutputShare >= opus.OutputShare {
		return compactRecommendation(
			"opus_cost_pressure",
			"high",
			"keep Fable as the default route and require an explicit Opus justification for cost-heavy long-context work",
			"Fable produced at least as much output as Opus, but Opus carried most estimated cost",
			fmt.Sprintf("opus_cost_share=%.1f%% fable_output_share=%.1f%% opus_output_share=%.1f%%",
				100*opus.CostShare, 100*fable.OutputShare, 100*opus.OutputShare),
		), true
	}
	return compactRecommendation(
		"opus_cost_pressure",
		"medium",
		"audit the top Opus-heavy sessions before launching more Opus turns",
		"Opus carried most estimated cost in the audited window",
		fmt.Sprintf("opus_cost_share=%.1f%% opus_output_share=%.1f%%", 100*opus.CostShare, 100*opus.OutputShare),
	), true
}

func compactLongContextPressure(rows []CompactLongContext) (CompactRecommendation, bool) {
	if len(rows) == 0 {
		return CompactRecommendation{}, false
	}
	top := rows[0]
	if top.TotalContextTokens < longContextPressureTokens && top.IORatio < longContextPressureIORatio {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if top.TotalContextTokens >= longContextPressureTokens || top.IORatio >= 2*longContextPressureIORatio {
		severity = "high"
	}
	return CompactRecommendation{
		Kind:     "long_context_pressure",
		Severity: severity,
		Action:   "checkpoint or reset the top long-context session before adding more high-cost turns; use ctxvalue/vcache context witnesses to prove shed-token value",
		Reason:   "the largest recent session is dominated by repeated context ingestion",
		Evidence: fmt.Sprintf("session=%s context_tokens=%d io_ratio=%.1f model=%s",
			top.Session, top.TotalContextTokens, top.IORatio, top.TopModel),
	}, true
}

const (
	// processIssueMinSessions: a stuck failure-loop recurring across this many
	// distinct sessions is systemic, not a one-off (1 is already the per-session signal).
	processIssueMinSessions = 2
	// processIssueMinTimeouts: this many shell timeout-kills across the window points
	// at a process/environment issue (a slow command, a wedged tool) rather than a route choice.
	processIssueMinTimeouts = 6
	// processIssueMinChurn: this much read-discipline churn (edit-before-read / stale-read
	// retries) across the window is worth surfacing as a work-hygiene issue.
	processIssueMinChurn = 12
)

// classEntry is one class's accumulated cross-session evidence. Sessions and namespaces are
// SETS, so a session that hits the same class ten times still counts as one session — which
// is what makes "seen in N distinct sessions" mean recurring rather than merely frequent.
type classEntry struct {
	sessions   map[string]bool
	namespaces map[string]bool
	example    string
	total      int64
}

// distinctSessions is the number of separate sessions that hit this class.
func (e *classEntry) distinctSessions() int64 { return int64(len(e.sessions)) }

// classRollup is the one cross-session join both recurring-* aggregations below share: fold
// many per-session rows into one entry per class, in first-sight order. Stating it once is
// what makes the two aggregations provably agree on what "recurring" means; the key type is
// free because only the counting is shared, not the vocabulary being counted.
type classRollup[K comparable] struct {
	entries map[K]*classEntry
	order   []K
}

func newClassRollup[K comparable]() *classRollup[K] {
	return &classRollup[K]{entries: map[K]*classEntry{}}
}

// observe folds one row into class k. First sight of a class fixes BOTH its example and its
// position in the output order, so the join is deterministic for a given input order. An
// empty namespace is not recorded: an unattributable path must not invent one.
func (r *classRollup[K]) observe(k K, session, namespace, example string, n int64) {
	e := r.entries[k]
	if e == nil {
		e = &classEntry{sessions: map[string]bool{}, namespaces: map[string]bool{}, example: example}
		r.entries[k] = e
		r.order = append(r.order, k)
	}
	e.sessions[session] = true
	if namespace != "" {
		e.namespaces[namespace] = true
	}
	e.total += n
}

// recurring walks, in first-sight order, every class seen in at least minSessions DISTINCT
// sessions and hands it to emit with its namespace set already sorted. Below the threshold a
// class is a one-off rather than a pattern, and is skipped.
func (r *classRollup[K]) recurring(minSessions int, emit func(k K, e *classEntry, namespaces []string)) {
	for _, k := range r.order {
		e := r.entries[k]
		if len(e.sessions) < minSessions {
			continue
		}
		names := make([]string, 0, len(e.namespaces))
		for n := range e.namespaces {
			names = append(names, n)
		}
		sort.Strings(names)
		emit(k, e, names)
	}
}

// aggregateCompactBehavior folds the per-session Behavior lens across the audited
// window: window totals (timeout-kills, sleep-polls, wasted read-discipline mutations,
// stuck sessions) plus the cross-session recurring-failure join. It returns nil when
// the window carries no behavioral signal at all, so a clean window omits the field.
func aggregateCompactBehavior(sessions []Session) *CompactBehavior {
	cb := &CompactBehavior{}
	type fkey struct{ tool, sig string }
	classes := newClassRollup[fkey]()
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		cb.Sessions++
		b := s.Behavior
		cb.TimeoutKills += b.TimeoutKills
		cb.SleepPolls += b.SleepPolls
		for _, v := range b.EditChurn {
			cb.WastedMutationCalls += v
		}
		if behaviorSessionStuck(b) {
			cb.StuckSessions++
		}
		ns := namespaceName(s.Path)
		for _, row := range b.FailureMass {
			// The example a failure class carries is the session it was first seen in.
			classes.observe(fkey{row.Tool, row.Sig}, s.Session, ns, s.Session, row.Count)
		}
	}
	classes.recurring(processIssueMinSessions, func(k fkey, e *classEntry, names []string) {
		cb.RecurringFailures = append(cb.RecurringFailures, RecurringFailureRow{
			Tool:           k.tool,
			Sig:            k.sig,
			Sessions:       e.distinctSessions(),
			Occurrences:    e.total,
			Namespaces:     names,
			ExampleSession: e.example,
		})
	})
	sort.SliceStable(cb.RecurringFailures, func(i, j int) bool {
		if cb.RecurringFailures[i].Sessions != cb.RecurringFailures[j].Sessions {
			return cb.RecurringFailures[i].Sessions > cb.RecurringFailures[j].Sessions
		}
		return cb.RecurringFailures[i].Occurrences > cb.RecurringFailures[j].Occurrences
	})
	if len(cb.RecurringFailures) > 10 {
		cb.RecurringFailures = cb.RecurringFailures[:10]
	}
	if cb.StuckSessions == 0 && cb.TimeoutKills == 0 && cb.SleepPolls == 0 &&
		cb.WastedMutationCalls == 0 && len(cb.RecurringFailures) == 0 {
		return nil
	}
	return cb
}

func behaviorSessionStuck(b Behavior) bool {
	return len(b.RepeatFailures) > 0 || len(b.FailureMass) > 0 || len(b.FileChurn) > 0 || len(b.SuccessLoops) > 0
}

// compactProcessIssuePressure raises a recommendation when the audited window shows a
// recurring process issue: the same stuck failure-loop across >= processIssueMinSessions
// distinct sessions, a shell timeout-kill storm, or heavy read-discipline churn. This is
// the behavioral counterpart to the cost/context pressure recommendations, and flows
// through the same actions gate.
func compactProcessIssuePressure(beh *CompactBehavior) (CompactRecommendation, bool) {
	if beh == nil {
		return CompactRecommendation{}, false
	}
	var top *RecurringFailureRow
	if len(beh.RecurringFailures) > 0 {
		top = &beh.RecurringFailures[0]
	}
	recurring := top != nil && top.Sessions >= processIssueMinSessions
	timeouts := beh.TimeoutKills >= processIssueMinTimeouts
	churn := beh.WastedMutationCalls >= processIssueMinChurn
	if !recurring && !timeouts && !churn {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if (top != nil && (top.Sessions >= 3 || len(top.Namespaces) >= 2)) || beh.TimeoutKills >= 2*processIssueMinTimeouts {
		severity = "high"
	}
	var reason, evidence string
	switch {
	case recurring:
		reason = "the same stuck failure-loop recurred across multiple sessions in the audited window"
		evidence = fmt.Sprintf("tool=%s sessions=%d occurrences=%d namespaces=%d timeout_kills=%d churn=%d sig=%q",
			top.Tool, top.Sessions, top.Occurrences, len(top.Namespaces), beh.TimeoutKills, beh.WastedMutationCalls, normHead(top.Sig, 80))
	case timeouts:
		reason = "the audited window is dominated by shell timeout-kills, a process/environment issue rather than a model-route choice"
		evidence = fmt.Sprintf("timeout_kills=%d stuck_sessions=%d churn=%d", beh.TimeoutKills, beh.StuckSessions, beh.WastedMutationCalls)
	default:
		reason = "read-discipline churn (edit-before-read / stale-read retries) is high across the audited window"
		evidence = fmt.Sprintf("churn=%d stuck_sessions=%d timeout_kills=%d", beh.WastedMutationCalls, beh.StuckSessions, beh.TimeoutKills)
	}
	return CompactRecommendation{
		Kind:     "process_issue_pressure",
		Severity: severity,
		Action:   "triage the recurring failure/churn signature and fix its root cause (env, tool, or prompt) before launching more sessions; deep-audit an example session to confirm the loop",
		Reason:   reason,
		Evidence: evidence,
	}, true
}

const (
	// confusedSessionMinMarkers: a session carrying this many prose confusion markers
	// (or the dead-end floor below) is a genuine outlier — the base rate is ~0.5% of
	// text turns, so 3 markers in one session is far above noise.
	confusedSessionMinMarkers = 3
	// confusedSessionMinDeadEnds: a session with this many dead-end turns (a repair that
	// visibly failed / the same failure recurring) is confused even with few other markers.
	confusedSessionMinDeadEnds = 2
	// confusionMinSessions: a confusion pattern recurring across this many distinct
	// sessions is systemic, not a one-off (mirrors processIssueMinSessions).
	confusionMinSessions = 2
	// confusionMinDeadEnds: this many dead-end turns across the window points at a
	// misleading signal the agent kept fighting rather than a single unlucky turn.
	confusionMinDeadEnds = 4
)

// aggregateCompactConfusion folds the per-session Confusion lens across the audited
// window: window turn totals, the count of sessions that crossed the confused threshold,
// the cross-session recurring-marker join, and the worst sessions to deep-audit first.
// It returns nil when the window carries no markers at all, so a clean window omits the
// field.
func aggregateCompactConfusion(sessions []Session) *CompactConfusion {
	cc := &CompactConfusion{}
	type mkey struct{ category, label string }
	classes := newClassRollup[mkey]()
	var confused []ConfusedSessionRow
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		cc.Sessions++
		conf := s.Confusion
		cc.TotalMarkers += conf.TotalMarkers
		cc.SelfCorrectionTurns += conf.SelfCorrectionTurns
		cc.DeadEndTurns += conf.DeadEndTurns
		cc.ConfusionTurns += conf.ConfusionTurns
		ns := namespaceName(s.Path)
		for _, row := range conf.Markers {
			// A marker class carries the marker TEXT it was first seen with as its example,
			// not a session id — the text is what an auditor needs to recognize the pattern.
			classes.observe(mkey{row.Category, row.Label}, s.Session, ns, row.Example, row.Count)
		}
		if confusedSession(conf) {
			cc.ConfusedSessions++
			// Silent = Behavior found no tool-loop signature; this is the slice the
			// Behavior lens is blind to and the one confusion_pressure gates on.
			silent := !behaviorSessionStuck(s.Behavior)
			if silent {
				cc.SilentConfusedSessions++
			}
			confused = append(confused, ConfusedSessionRow{
				Session:      s.Session,
				Namespace:    ns,
				Markers:      conf.TotalMarkers,
				DeadEndTurns: conf.DeadEndTurns,
				Score:        conf.Score,
				Silent:       silent,
			})
		}
	}
	classes.recurring(confusionMinSessions, func(k mkey, e *classEntry, names []string) {
		cc.RecurringMarkers = append(cc.RecurringMarkers, RecurringMarkerRow{
			Category:   k.category,
			Label:      k.label,
			Sessions:   e.distinctSessions(),
			Count:      e.total,
			Namespaces: names,
			Example:    e.example,
		})
	})
	sort.SliceStable(cc.RecurringMarkers, func(i, j int) bool {
		if cc.RecurringMarkers[i].Sessions != cc.RecurringMarkers[j].Sessions {
			return cc.RecurringMarkers[i].Sessions > cc.RecurringMarkers[j].Sessions
		}
		return cc.RecurringMarkers[i].Count > cc.RecurringMarkers[j].Count
	})
	if len(cc.RecurringMarkers) > 10 {
		cc.RecurringMarkers = cc.RecurringMarkers[:10]
	}
	sort.SliceStable(confused, func(i, j int) bool {
		// Behavior-invisible (silent) confused sessions rank first — they are the ones a
		// process-issue audit would miss, so they earn the operator's attention.
		if confused[i].Silent != confused[j].Silent {
			return confused[i].Silent
		}
		if confused[i].Markers != confused[j].Markers {
			return confused[i].Markers > confused[j].Markers
		}
		if confused[i].Score != confused[j].Score {
			return confused[i].Score > confused[j].Score
		}
		return confused[i].Session < confused[j].Session
	})
	if len(confused) > 10 {
		confused = confused[:10]
	}
	cc.TopSessions = confused
	if cc.TotalMarkers == 0 {
		return nil
	}
	return cc
}

// confusedSession reports whether one session crossed the confused threshold: enough
// total markers, or enough dead-end turns on their own.
func confusedSession(c Confusion) bool {
	return c.TotalMarkers >= confusedSessionMinMarkers || c.DeadEndTurns >= confusedSessionMinDeadEnds
}

// compactConfusionPressure raises a recommendation when the audited window shows a
// recurring reasoning-friction pattern the Behavior lens is blind to: the same confusion
// marker across multiple sessions, several Behavior-silent confused sessions, or a
// dead-end loop the agent kept fighting. Every trigger is gated on at least one
// Behavior-silent confused session (see SilentConfusedSessions) so it stays strictly
// complementary to compactProcessIssuePressure — it never restates a tool-loop finding
// Behavior already owns. It flows through the same actions gate as the other pressures.
func compactConfusionPressure(cc *CompactConfusion) (CompactRecommendation, bool) {
	if cc == nil {
		return CompactRecommendation{}, false
	}
	var top *RecurringMarkerRow
	if len(cc.RecurringMarkers) > 0 {
		top = &cc.RecurringMarkers[0]
	}
	// The lens only earns a recommendation where the tool-I/O Behavior lens is BLIND:
	// a session that thrashed in prose while every tool call succeeded. Gate on there
	// being at least one such Behavior-silent confused session, so confusion_pressure
	// never merely restates a process_issue_pressure finding Behavior already owns.
	if cc.SilentConfusedSessions == 0 {
		return CompactRecommendation{}, false
	}
	recurring := top != nil && top.Sessions >= confusionMinSessions
	confusedMany := cc.SilentConfusedSessions >= confusionMinSessions
	deadends := cc.DeadEndTurns >= confusionMinDeadEnds
	if !recurring && !confusedMany && !deadends {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if cc.SilentConfusedSessions >= 3 || cc.DeadEndTurns >= 2*confusionMinDeadEnds || (top != nil && len(top.Namespaces) >= 2) {
		severity = "high"
	}
	var reason, evidence string
	switch {
	case recurring:
		reason = "the same reasoning-confusion marker recurred across multiple sessions — a systemic pattern, not a one-off"
		evidence = fmt.Sprintf("category=%s label=%s sessions=%d count=%d namespaces=%d silent_confused=%d dead_end_turns=%d example=%q",
			top.Category, top.Label, top.Sessions, top.Count, len(top.Namespaces), cc.SilentConfusedSessions, cc.DeadEndTurns, normHead(top.Example, 80))
	case deadends:
		reason = "the audited window is heavy with dead-end turns — repairs that visibly failed or the same failure recurring, a sign the agent kept fighting a misleading signal"
		evidence = fmt.Sprintf("dead_end_turns=%d silent_confused=%d confused_sessions=%d total_markers=%d", cc.DeadEndTurns, cc.SilentConfusedSessions, cc.ConfusedSessions, cc.TotalMarkers)
	default:
		reason = "multiple Behavior-silent sessions crossed the confusion threshold — prose thrash the tool-I/O lens cannot see"
		evidence = fmt.Sprintf("silent_confused=%d confused_sessions=%d total_markers=%d self_correction_turns=%d dead_end_turns=%d", cc.SilentConfusedSessions, cc.ConfusedSessions, cc.TotalMarkers, cc.SelfCorrectionTurns, cc.DeadEndTurns)
	}
	return CompactRecommendation{
		Kind:     "confusion_pressure",
		Severity: severity,
		Action:   "deep-audit the most-confused session to find where the agent lost the thread, then reduce the friction at its source — clearer task framing / earlier reconnaissance for self-correction churn, or fix the misleading signal (flaky test, stale doc, wrong error) behind a dead-end loop",
		Reason:   reason,
		Evidence: evidence,
	}, true
}

func compactRecommendation(kind, severity, action, reason, evidence string) CompactRecommendation {
	return CompactRecommendation{
		Kind:     kind,
		Severity: severity,
		Action:   action,
		Reason:   reason,
		Evidence: evidence,
	}
}

func compactTierByName(tiers []CompactTier, name string) (CompactTier, bool) {
	for _, tier := range tiers {
		if tier.Tier == name {
			return tier, true
		}
	}
	return CompactTier{}, false
}

const (
	// shellFrictionMinCalls: a shell must reach this call volume before its error rate
	// is treated as signal rather than noise (#3227, #6523). A 3-call shell with 1 error
	// is 33% on sample noise alone.
	shellFrictionMinCalls = int64(10)
	// shellFrictionThreshold: error rate at or above which a shell is flagged for
	// friction. In #3227, PowerShell ran at 18.2% (6/33) vs Bash at 2.6% (5/194).
	shellFrictionThreshold = 0.10
)

// compactShellFrictionPressure raises a recommendation when a shell's error rate
// crosses shellFrictionThreshold over at least shellFrictionMinCalls (#3227, #6523).
// Stays silent for low-volume shells whose error rate is noise.
func compactShellFrictionPressure(sc CompactShellChoice) (CompactRecommendation, bool) {
	var worst *ShellStat
	for i := range sc.Shells {
		s := &sc.Shells[i]
		if s.Calls < shellFrictionMinCalls || s.ErrorRate == nil || *s.ErrorRate < shellFrictionThreshold {
			continue
		}
		if worst == nil || *s.ErrorRate > *worst.ErrorRate || (*s.ErrorRate == *worst.ErrorRate && s.Calls > worst.Calls) {
			worst = s
		}
	}
	if worst == nil {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if *worst.ErrorRate >= 0.15 || worst.Errors >= 5 {
		severity = "high"
	}
	preferred := sc.Preferred
	if preferred == "UNKNOWN" {
		preferred = ""
	}
	var action string
	if preferred != "" && preferred != worst.Tool {
		action = fmt.Sprintf("investigate %s command failures and consider routing commands through %s or fixing shell environment issues", worst.Tool, preferred)
	} else {
		action = fmt.Sprintf("investigate and resolve %s command failures and shell environment issues", worst.Tool)
	}
	reason := fmt.Sprintf("%s error rate (%.1f%% over %d calls) exceeds the shell friction threshold (%.1f%%)",
		worst.Tool, 100**worst.ErrorRate, worst.Calls, 100*shellFrictionThreshold)
	evidence := fmt.Sprintf("tool=%s calls=%d errors=%d error_rate=%.1f%% all_shell_calls=%d all_shell_errors=%d preferred=%s",
		worst.Tool, worst.Calls, worst.Errors, 100**worst.ErrorRate, sc.Calls, sc.Errors, sc.Preferred)

	return CompactRecommendation{
		Kind:     "shell_friction_pressure",
		Severity: severity,
		Action:   action,
		Reason:   reason,
		Evidence: evidence,
	}, true
}
