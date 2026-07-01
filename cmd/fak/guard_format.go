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
				status = fmt.Sprintf("ENABLED but ANCHOR-STARVED, budget %d tok — the cache_control anchor protects MORE than the budget so it cannot fire (NOT a short session; pass --compact-anchor-head to re-anchor, #1407)", sum.CompactionBudget)
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
			fmt.Fprintf(&b, "  ⚠ anchor-starved x%d — protected prefix exceeds the %d-tok budget; compaction cannot fire on this traffic regardless of session length (pass --compact-anchor-head to re-anchor, not a tighter budget — #1407)\n",
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
		fmt.Fprintf(&b, "fak guard: deny-all stops — %d turn(s) had EVERY proposed tool call refused, reported to the client as end_turn (a stop the agent did not choose; the model wanted to act, the floor blocked all of it). Keep the agent moving past these with --deny-all-continue=enforce (auto-resumes the agent with 'choose an allowed alternative', bounded).\n",
			sum.DenyAllStops)
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
	reasons := make([]string, 0, 3)
	if sum.CompactionAnchorStarved > 0 {
		reasons = append(reasons, fmt.Sprintf("anchor-starved x%d (protected prefix exceeds the %d-tok compaction budget; pass --compact-anchor-head to re-anchor)", sum.CompactionAnchorStarved, sum.CompactionBudget))
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
	return fmt.Sprintf("fak guard: fak-slice diagnostic — F is ~0 because %s.\n", strings.Join(reasons, "; "))
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
