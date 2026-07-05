package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/callavoid"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// This file holds the `fak guard` exit-summary FORMATTERS — the pure
// string-rendering helpers split out of guard.go so the dispatch file stays under
// the steerability hard ceiling. They take already-folded summary structs and
// return display text; no I/O, no decision logic.

// formatJournalSummary is the exit-roll-up line for the durable trail: how many
// hash-chained rows this session appended, where, and the command to re-verify the
// chain. Empty when no journal ran, so a --no-audit session stays quiet.
func formatJournalSummary(j *journal.Journal, seq0 uint64) string {
	if j == nil {
		return ""
	}
	path := j.Path()
	if path == "" {
		return ""
	}
	if err := j.Flush(); err != nil {
		return fmt.Sprintf("fak guard: audit journal — flush error: %v\n", err)
	}
	seq, _, writeErr := j.Stats()
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard: audit journal — %d decision(s) appended this session; chain now holds %d hash-chained row(s) at %s",
		seq-seq0, seq, path)
	if writeErr > 0 {
		fmt.Fprintf(&b, " (%d write error(s))", writeErr)
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "  verify the tamper-evident chain: fak audit verify %s\n", path)
	return b.String()
}

// formatAuditSummary renders the exit roll-up of what the kernel decided while the
// agent ran. "kernel decision(s)" — not "tool calls" — because the tally folds BOTH
// proposed-call adjudications AND inbound tool-result admissions (a quarantined result
// is a kernel decision about a result the agent already ran, not a proposed call). It
// is one honest count: every number came from the same operation counters /metrics
// exposes, so the line can never overstate the protection.
func formatAuditSummary(sum gateway.AdjudicationSummary, kcOpt ...kernel.Counters) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard: %d kernel decision(s) — %d allowed, %d denied, %d repaired, %d quarantined",
		sum.Total, sum.Allowed, sum.Denied, sum.Transformed, sum.Quarantined)
	// Deferred (a non-blocking admit, e.g. a tool result let through) and escalated
	// (held pending a witness) are normal, non-error outcomes — show them only when
	// they happened so the common clean line stays short, and never under "errored".
	if sum.Deferred > 0 {
		fmt.Fprintf(&b, ", %d deferred", sum.Deferred)
	}
	if sum.Escalated > 0 {
		fmt.Fprintf(&b, ", %d escalated", sum.Escalated)
	}
	if sum.Errored > 0 {
		fmt.Fprintf(&b, ", %d errored", sum.Errored)
	}
	b.WriteByte('\n')
	cacheSavings := sum.MechanismSavings()
	if len(kcOpt) > 0 && kcOpt[0].VDSOHits > 0 {
		cacheSavings.FakVDSOAvoidedCalls = uint64(kcOpt[0].VDSOHits)
	}
	if line := formatCacheAttribution(cacheSavings); line != "" {
		b.WriteString(line)
	}
	if line := formatFakSliceDiagnostic(sum); line != "" {
		b.WriteString(line)
	}
	if sum.CompactionFired > 0 || sum.CompactionBailed > 0 || sum.CompactionOff > 0 {
		// WITNESSED half only: what fak attempted and removed. The OBSERVED post-fire cache_read
		// is a provider counter (it lives on /metrics) and is noise here — a low value with no
		// prefix_mismatch bail is a provider-side miss fak does not control, not a fak failure.
		// Lead with whether the lever is ENABLED so "0 fired" can't read as "disabled": budget>0
		// with all-under_budget bails is compaction ON and correctly idle (nothing sprawled past
		// the cut), the opposite of OFF.
		status := fmt.Sprintf("ENABLED, budget %d tok", sum.CompactionBudget)
		if sum.CompactionBudget <= 0 {
			status = "DISABLED (budget 0; body forwarded byte-for-byte)"
		} else if sum.CompactionFired == 0 && sum.CompactionShedTokens == 0 {
			status = fmt.Sprintf("ENABLED but idle, budget %d tok — nothing sprawled past the cut", sum.CompactionBudget)
			// An idle that is NOT a short session: the cache_control anchor protected a prefix
			// larger than the budget, so the lever could not fire no matter the session length.
			// This is the dormant-on-real-Claude-Code-traffic pathology (#1407), the opposite of
			// "nothing sprawled" — call it out so a tighter budget isn't misread as the fix.
			if sum.CompactionAnchorStarved > 0 {
				status = fmt.Sprintf("ENABLED but ANCHOR-STARVED, budget %d tok — the cache_control anchor protects MORE than the budget so it cannot fire (NOT a short session; --compact-anchor-head is default-on, so either it was disabled or the traffic carries no stable system/tools breakpoint to re-anchor on, #1407)", sum.CompactionBudget)
			}
		}
		fmt.Fprintf(&b, "fak guard: compaction [%s] — %d fired, %d bailed, %d off; shed %d token(s)\n",
			status,
			sum.CompactionFired,
			sum.CompactionBailed,
			sum.CompactionOff,
			sum.CompactionShedTokens)
		// Break the bailed lump out by reason (same shape as the deny "blocked:" loop below):
		// without this, N bailed conflates under_budget (benign, working-as-designed) with
		// no_breakpoint (can't fire) and prefix_mismatch (the ONLY fak-fault cache signal — call
		// it out explicitly when nonzero so a real regression can never hide in the lump).
		if len(sum.CompactionBailReasons) > 0 {
			reasons := make([]string, 0, len(sum.CompactionBailReasons))
			for r := range sum.CompactionBailReasons {
				reasons = append(reasons, r)
			}
			sort.Strings(reasons)
			for _, r := range reasons {
				note := ""
				if r == "prefix_mismatch" || r == "splice_failed" || r == "redecode_failed" {
					note = "  ⚠ fak-fault: a fired rewrite would have burst the cache — must stay 0"
				}
				fmt.Fprintf(&b, "  bailed: %-16s x%d%s\n", r, sum.CompactionBailReasons[r], note)
			}
		}
		// Anchor-starved is a SUBSET of the under_budget bails above, surfaced apart because it is
		// operationally opposite: a plain under_budget is a benign short session, an anchor-starved
		// one means the anchor swallowed the conversation so no budget tightening can ever make it
		// fire — only a re-anchor (#1407 / opt-in head-anchored firing #1408) can.
		if sum.CompactionAnchorStarved > 0 {
			fmt.Fprintf(&b, "  ⚠ anchor-starved x%d — protected prefix exceeds the %d-tok budget; compaction cannot fire on this traffic regardless of session length (a re-anchor is the fix, not a tighter budget: --compact-anchor-head is default-on, so either it was disabled or the traffic carries no stable system/tools breakpoint — #1407)\n",
				sum.CompactionAnchorStarved, sum.CompactionBudget)
		}
	}
	// Tool-floor prune (the INBOUND tools[] lever): how many unreachable tool DEFINITIONS fak
	// dropped from the advertised surface this session — a pure uncached-token saving that
	// never bursts the cache (the pruner only drops tools after the cache_control breakpoint and
	// re-proves the protected prefix is byte-identical). WITNESSED. Printed only when it actually
	// fired, so the common run — and the dominant Claude Code path, whose single breakpoint sits on
	// the LAST tool so nothing is droppable — stays quiet rather than printing a vacuous 0.
	if sum.ToolPruneCount > 0 {
		fmt.Fprintf(&b, "fak guard: tool-floor prune — dropped %d unreachable tool def(s) from tools[] across %d turn(s) (uncached-token saving; cache prefix byte-identical)\n",
			sum.ToolPruneCount, sum.ToolPruneTurns)
	}
	// Deny-all stops: turns the floor refused ENTIRELY, which the wire reports to the client as
	// end_turn so it does not hang hunting for a dropped tool_use block (the v0.15.0 contract).
	// That end_turn halts the agent though the model wanted to act — a STOP the agent did not
	// choose, and the false-stop this audit surfaces. Print it only when it happened, and name the
	// Stop-hook lever that auto-resumes the agent past it, so a session that hit the false stop
	// tells the operator both that it happened and how to keep the loop moving next time.
	if sum.DenyAllStops > 0 {
		fmt.Fprintf(&b, "fak guard: deny-all stops — %d turn(s) had every proposed tool call set aside, reported to the client as end_turn (a stop the agent did not choose; the model wanted to act and the floor set all of it aside). Keep the agent moving past these with --deny-all-continue=enforce (auto-resumes the agent with 'choose an allowed alternative', bounded).\n",
			sum.DenyAllStops)
	}
	// Tool-feedback turns: the RETRYABLE sibling of the deny-all stops above. Every proposed
	// tool call was refused as model-fixable feedback (e.g. MALFORMED args), so the turn is
	// live and the model may retry — this is a PER-TOOL refusal count, NOT a turn/session stop.
	// Surfacing it apart from (and NOT folded into) the deny-all stops is what keeps a strict
	// floor from reading like a dead session: a run of these is the floor working, not the
	// session ending (#2632). A session stop comes only from a declared stop policy, never from
	// accumulated tool refusals. Printed only when it happened so a clean run stays quiet.
	if sum.ToolFeedbackTurns > 0 {
		fmt.Fprintf(&b, "fak guard: tool-feedback turns — %d turn(s) had every proposed tool call returned as RETRYABLE feedback (per-tool, model-fixable; the turn was NOT stopped — the model can fix the arguments or tool choice and retry). This is a tool-refusal count, not a session stop: a stop comes only from a declared stop policy.\n",
			sum.ToolFeedbackTurns)
	}
	if len(sum.ByReason) > 0 {
		reasons := make([]string, 0, len(sum.ByReason))
		for r := range sum.ByReason {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		for _, r := range reasons {
			fmt.Fprintf(&b, "  blocked: %-16s x%d\n", r, sum.ByReason[r])
		}
	}
	// A DEFAULT_DENY is a tool the floor never enumerated (a harness verb, an MCP tool,
	// a new orchestration name) — not a genuine-danger refusal. The permissive-defaults
	// posture is that those are usually FINE to allow, so point the operator at the
	// out-of-band control that re-admits them for future sessions. Deliberately NOT shown
	// for POLICY_BLOCK / SECRET_EXFIL: those are the danger floor (an arg-rule like rm -rf
	// or an explicit deny), which the allow overlay cannot and should not lift.
	if sum.ByReason["DEFAULT_DENY"] > 0 {
		b.WriteString("  to always-allow a blocked tool for future sessions (operator, out-of-band from the agent):\n")
		b.WriteString("    fak guard allow --from-journal        # lists what was blocked + the exact allow command\n")
	}
	return b.String()
}

func formatCacheAttribution(s gateway.MechanismSavings) string {
	if !s.HasAnyTokenActivity() && s.FakVDSOAvoidedCalls == 0 {
		return ""
	}
	provider := s.ProviderTokenEquiv()
	fak := s.FakTokenEquiv()
	total := s.TotalTokenEquiv()
	var b strings.Builder
	if total > 0 {
		fmt.Fprintf(&b, "fak guard: avoided-spend attribution — provider ~%s (%.0f%%) + fak ~%s (%.0f%%) = ~%s token-equiv",
			gateway.HumanTokenEquiv(provider), provider/total*100,
			gateway.HumanTokenEquiv(fak), fak/total*100,
			gateway.HumanTokenEquiv(total))
	} else {
		fmt.Fprintf(&b, "fak guard: cache attribution — provider net ~%s + fak ~%s = ~%s token-equiv (not yet positive)",
			gateway.HumanTokenEquiv(provider),
			gateway.HumanTokenEquiv(fak),
			gateway.HumanTokenEquiv(total))
	}
	fmt.Fprintf(&b, " [provider read rebate %s, write premium %s; fak compaction %s, KV-prefix %s",
		gateway.HumanTokenEquiv(s.ProviderPromptCacheReadTokenEquiv),
		gateway.HumanTokenEquiv(s.ProviderPromptCacheWritePremiumTokenEquiv),
		gateway.HumanTokenEquiv(float64(s.FakCompactionShedTokens)),
		gateway.HumanTokenEquiv(float64(s.FakKVPrefixReusedTokens)))
	if s.FakVDSOAvoidedCalls > 0 {
		fmt.Fprintf(&b, "; vDSO %d avoided call(s)", s.FakVDSOAvoidedCalls)
	}
	b.WriteString("]. provider is OBSERVED/provider-relayed; fak is WITNESSED/fak-authored.\n")
	return b.String()
}

func formatFakSliceDiagnostic(sum gateway.AdjudicationSummary) string {
	savings := sum.MechanismSavings()
	if savings.FakTokenEquiv() > 0 || !fakSliceDiagnosticRelevant(sum) {
		return ""
	}
	reasons := make([]string, 0, 4)
	if sum.CompactionAnchorStarved > 0 {
		reasons = append(reasons, fmt.Sprintf("anchor-starved x%d (protected prefix exceeds the %d-tok compaction budget; needs a --compact-anchor-head re-anchor — default-on unless disabled)", sum.CompactionAnchorStarved, sum.CompactionBudget))
	}
	// burst_unprofitable is the OTHER reason a head-anchored shed does not fire, operationally
	// opposite to anchor-starved: there the anchor is fine and the middle IS shed-able, but the
	// one-time cost of re-writing the invalidated cache suffix has no repaying horizon — this
	// session carries no bounded turn budget AND has not idled past the message-cache TTL, so the
	// burst economics (CacheBurstPaysBack, #1408) conservatively keep the warm cache rather than
	// guess. This is the common WARM continuously-active long-session case (a headless worker that
	// never idles 5 min); the fix is a bounded horizon (a turn/context budget) or an idle gap, NOT
	// a tighter budget. Named apart so its "did not fire" reads as working-as-designed, not a bug.
	if n := sum.CompactionBailReasons["burst_unprofitable"]; n > 0 {
		reasons = append(reasons, fmt.Sprintf("burst-unprofitable x%d (shed-able middle, but no repaying turn-horizon and not idle past the message-cache TTL, so the burst economics keep the warm cache; fix = a bounded turn/context budget or an idle gap, not a tighter budget)", n))
	}
	switch {
	case sum.KVPrefixPromptTokens == 0:
		reasons = append(reasons, "no in-kernel KV-prefix multi-turn traffic observed")
	case sum.KVPrefixReusedTokens == 0:
		reasons = append(reasons, "no multi-turn KV-prefix reuse observed")
	}
	if sum.CompactionBudget <= 0 && sum.CompactionOff > 0 {
		reasons = append(reasons, "compaction disabled")
	}
	if (sum.CachedPromptTokens > 0 || sum.CacheCreationTokens > 0) && sum.CompactionAnchorStarved == 0 {
		reasons = append(reasons, "M2/default anchor gate did not produce a fak-authored saving on this provider-cache session")
	}
	if len(reasons) == 0 {
		reasons = append(reasons, "M2/default-on cache gates did not fire on this traffic")
	}
	// Point the operator at the one command that shows the accumulated cross-session
	// value and the provider-vs-fak owner split, so "F is ~0 this session" is legible as
	// "the provider is doing the caching; fak's own shed did not fire" rather than a dead
	// end. This is the fak-OWN cache-value path (outbound compaction-shed on the passthrough
	// #555) — see docs/notes/GUARD-OWN-CACHE-VALUE-PATH.md for how to make it fire.
	return fmt.Sprintf("fak guard: fak-slice diagnostic — F is ~0 because %s.\n"+
		"  see: fak cachevalue report   (provider-vs-fak owner split over all sessions; docs/notes/GUARD-OWN-CACHE-VALUE-PATH.md)\n",
		strings.Join(reasons, "; "))
}

func fakSliceDiagnosticRelevant(sum gateway.AdjudicationSummary) bool {
	return sum.CachedPromptTokens > 0 ||
		sum.CacheCreationTokens > 0 ||
		sum.CompactionFired > 0 ||
		sum.CompactionBailed > 0 ||
		sum.CompactionOff > 0 ||
		sum.CompactionAnchorStarved > 0 ||
		sum.KVPrefixPromptTokens > 0
}

// formatAmplification renders the avoided-call amplification headline for the guard
// exit summary — the realized answer to "how much further did the agent get per unit
// of real work?" It folds the session's kernel call-path counters (engine dispatches,
// vDSO hits, in-syscall repairs, fast-reject denies) through internal/callavoid.Account,
// the SAME pure economics the `fak callavoid account` CLI computes, so the line can
// never disagree with that tool. This closes the callavoid leaf's "Next milestone (not
// yet wired)": the tier-4 caller that reads a live guard session's kernel.Counters into
// Account for the exit summary.
//
// It returns the empty string when there was no avoidance to report — a session whose
// vDSO never hit and whose kernel repaired nothing has nothing to amplify (Execute-only
// work is 1:1), so the common clean run stays quiet rather than printing a vacuous 1.0×.
//
// kc is the in-kernel call-path axis (vDSO memo hits + in-syscall repairs), which only moves
// on the Submit/Reap path `fak serve` drives. On the flagship `fak guard -- claude` PROXY the
// kernel adjudicates with Decide, which increments none of those counters — so kc is empty
// every guard session and the kernel-axis line would never fire there. sum carries the
// Decide-path verdicts that DO move on the proxy (grammar repairs = Transformed, fast-reject
// denies = Denied); when kc is empty but the proxy repaired/denied real calls, we print a
// proxy-honest line about what the floor DID, framed as "repairs/denies applied" rather than
// "calls avoided" (a Decide-only proxy avoids no calls — the client still executes each tool).
func formatAmplification(kc kernel.Counters, sum gateway.AdjudicationSummary) string {
	// Map the live kernel counters onto the tier-1 callavoid mirror (a total, behaviour-
	// free field copy — the field names mirror kernel.Counters on purpose) and fold.
	rep := callavoid.Account(callavoid.TallyFromCounters(callavoid.Counters{
		EngineCalls: int(kc.EngineCalls),
		VDSOHits:    int(kc.VDSOHits),
		Transforms:  int(kc.Transforms),
		Denies:      int(kc.Denies),
	}))
	// Nothing was avoided on the kernel axis. Before staying silent, check the PROXY axis:
	// on `fak guard -- claude` the kernel counters are structurally 0 (Decide increments none),
	// but the floor may have repaired or denied real proposed calls — work the agent would
	// otherwise have paid a failed round-trip for. Surface THAT so the dominant path is not
	// silently blank when the floor was actually doing its job.
	if rep.MemoHits == 0 && rep.Repairs == 0 {
		if sum.Transformed > 0 || sum.Denied > 0 {
			return fmt.Sprintf("fak guard: floor effect — %d call(s) repaired in-flight, %d denied before a wasted round-trip (proxy path: the kernel adjudicates with Decide, so the in-kernel vDSO/amplification axis does not apply)\n",
				sum.Transformed, sum.Denied)
		}
		return ""
	}
	var b strings.Builder
	// Lead with the realized amplification ratio and the turns it spared, then the
	// breakdown of WHERE the avoidance came from (vDSO cache hits + in-syscall repairs).
	// A memo hit always pays callavoid.ValidateFloor (never free), so a pure-cache window
	// is capped at 1/ValidateFloor (=100×), not +Inf — Amplification is always finite on
	// this path. The only +Inf case is zero executed work, which means zero memo hits and
	// zero repairs, which the guard above has already returned the empty string for.
	fmt.Fprintf(&b, "fak guard: avoided-call amplification — %.2f× (%s); spared ~%.0f naive round-trip(s) of %d proposed",
		rep.Amplification, rep.Status, rep.AvoidedTurns, rep.RawTurns)
	parts := make([]string, 0, 2)
	if rep.MemoHits > 0 {
		parts = append(parts, fmt.Sprintf("%d served from the vDSO cache", rep.MemoHits))
	}
	if rep.Repairs > 0 {
		parts = append(parts, fmt.Sprintf("%d repaired in-syscall", rep.Repairs))
	}
	if len(parts) > 0 {
		fmt.Fprintf(&b, " — %s", strings.Join(parts, ", "))
	}
	b.WriteByte('\n')
	return b.String()
}

// formatTurnsTimeSaved renders the dedicated "turns / time saved" headline for the guard
// exit summary — the human-legible dual of the amplification/floor lines above. The turns
// count reconciles with them BY CONSTRUCTION: it folds the same kernel counters through the
// same callavoid.Account, and splits on the same either/or path boundary, so the two
// surfaces can never disagree and never double-count.
//
//   - Kernel/`fak serve` axis (vDSO hits + in-syscall repairs move the counters): turns
//     saved is Account's realized AvoidedTurns — the round-trips a naive 1:1 agent would
//     have spent to reach the same state.
//   - Flagship `fak guard -- claude` PROXY axis (kc is structurally 0 — Decide increments
//     none): the witnessed turn-saver is the in-flight grammar repair (a repaired call
//     spares the failed round-trip + retry a naive agent would pay). Denies are NOT counted
//     — the callavoid model treats a hard deny as symmetric (both agents propose-and-deny
//     once), so "K spared" equals the floor-effect line's "K repaired"; denies stay shown
//     separately by formatAmplification, never folded in here.
//
// Time saved = turns × the session's OWN observed mean E2E turn latency (WITNESSED/OBSERVED,
// never a fabricated tokens/sec). When no turn was timed this session
// (MeanTurnLatencySeconds == 0), the wall-clock dual is omitted rather than estimated — the
// honest read is "we spared K round-trips; we cannot price them without a measured latency".
// Returns "" when nothing was witnessed to report, matching formatAmplification's clean-run
// silence.
func formatTurnsTimeSaved(kc kernel.Counters, sum gateway.AdjudicationSummary) string {
	rep := callavoid.Account(callavoid.TallyFromCounters(callavoid.Counters{
		EngineCalls: int(kc.EngineCalls),
		VDSOHits:    int(kc.VDSOHits),
		Transforms:  int(kc.Transforms),
		Denies:      int(kc.Denies),
	}))
	turns := rep.AvoidedTurns
	basis := "vDSO cache hits + in-syscall repairs"
	if rep.MemoHits == 0 && rep.Repairs == 0 {
		// Proxy path: the kernel amplification axis is empty, so the witnessed turn-saver is
		// the Decide-path grammar repair (Transformed). Denied is deliberately excluded.
		turns = float64(sum.Transformed)
		basis = "in-flight grammar repairs"
	}
	if turns < 0.5 { // nothing witnessed — stay quiet, same rule the amplification line uses.
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard: turns / time saved — ~%.0f naive round-trip(s) spared", turns)
	if mean := sum.MeanTurnLatencySeconds(); mean > 0 {
		fmt.Fprintf(&b, " ≈ ~%.1fs of wall-clock (this session's observed ~%.1fs/turn over %d timed turn(s))",
			turns*mean, mean, sum.E2ELatencyCount)
	} else {
		b.WriteString("; wall-clock omitted — no turn latency observed this session")
	}
	fmt.Fprintf(&b, ". WITNESSED: %s; time is observed, not modeled\n", basis)
	return b.String()
}
