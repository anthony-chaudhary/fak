package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// Live cache ABLATION for the `fak info` Cache tab. The offline `fak ablate` / `fak console
// ablate` pane answers "how does each caching CONCEPT pay off" by REPLAYING a frozen reference
// trace under N feature configs — which needs a trace file on disk, so it cannot run inside the
// live overlay (an installed binary run anywhere has no testdata trace to replay). This is its
// LIVE twin: instead of replaying a trace, it reads the session's OWN witnessed cache-attribution
// counters (/debug/vars CacheAttribution — the same numbers the guard-exit banner prints) and
// shows, per mechanism, the token-equivalent saving you would LOSE by ablating it. A per-mechanism
// savings bar sourced entirely from what THIS session already witnessed: no trace replay, no
// sweep, no gateway read beyond the snapshot the cache tab already holds — a pure projection.
//
// It renders BYTE-CLEAN rows (no SGR); color is layered afterward by colorizeGuardInfoBlock, which
// paints each mechanism row in its owner's hue — provider prompt-cache cyan, fak-authored green —
// matching the offline ablate pane's provider/fak palette. Keeping the rows monochrome here lets
// every width/layout test keep asserting on plain text, exactly as the rest of the pane does.

// cacheAblationMech is one ablatable cache mechanism: its label, and its token-equivalent saving
// this session (what turning it off would cost). detail is an optional trailing clause (the
// provider read/write split) shown only when the gateway reported the sub-fields.
type cacheAblationMech struct {
	label  string
	tokEq  float64
	detail string
}

// cacheAblationMechs is the per-mechanism token-equivalent decomposition of the SAME owner split
// the rollup line (guardInfoCacheAttributionText) prints — so the bars and the rollup can never
// disagree. Every mechanism is priced in the one input-token currency the rollup uses:
//
//   - provider prompt-cache is the provider NET (read rebate minus write premium), ProviderTokenEquiv.
//   - fak compaction shed is the WARM/COLD-blended shed value, NOT the raw shed count: it is
//     FakTokenEquiv minus the prefix slice, i.e. exactly cacheprice.ShedTokenEquiv(shed, warm).
//     Booking the raw shed here (the prior bug) over-stated fak's compaction on a warm session —
//     the raw count can dwarf the provider net while the rollup's fak slice is a fraction of it,
//     so the bars and the split line told two different stories. Deriving it from FakTokenEquiv
//     keeps the bar reconciled with the rollup by CONSTRUCTION, with no warm-witness field needed.
//   - fak KV-prefix reuse is valued at its full local marginal, so raw tokens == token-equiv.
//
// By construction the three mechanisms sum to TotalTokenEquiv, so the bars decompose the exact
// total the split line shows, and each bar's share equals its share of that split.
func cacheAblationMechs(ca *guardInfoCacheAttribution) []cacheAblationMech {
	shedTokEq := ca.FakTokenEquiv - float64(ca.FakKVPrefixReusedTokens)
	if shedTokEq < 0 { // a partial/older producer with an inconsistent pair never draws a negative shed
		shedTokEq = 0
	}
	return []cacheAblationMech{
		{label: "provider prompt-cache", tokEq: ca.ProviderTokenEquiv, detail: providerAblationDetail(ca)},
		{label: compactionShedMechLabel, tokEq: shedTokEq, detail: shedAblationDetail(ca, shedTokEq)},
		{label: "fak KV-prefix reuse", tokEq: float64(ca.FakKVPrefixReusedTokens)},
	}
}

// renderInfoCacheAblationRows projects the live CacheAttribution block into the Cache tab's
// ablation section: a section rule, a one-line framing, then one savings-bar row per token
// mechanism (bar length ∝ that mechanism's share of the largest mechanism's save), plus the vDSO
// avoided-CALLS line (calls, not tokens, so it never reads as a token saving on the bars above) and
// the cold-tool-defer block (#3647) — the third shed mechanism, kept off the bars for the same
// reason as vDSO: it is a counted event, not a priced token saving.
// When no attribution is reported, or every mechanism is zero (a cold / plain-passthrough
// session), it renders a single honest line so the tab still explains itself rather than showing
// an empty rule.
func renderInfoCacheAblationRows(ctx guardInfoPanelCtx) []string {
	width := ctx.width
	rows := []string{guardInfoRuleTUI("cache ablation", width)}
	ca := ctx.v.CacheAttribution
	if ca == nil {
		return append(rows, " ablation  no attribution yet — each cache mechanism's save appears here once witnessed")
	}
	mechs := cacheAblationMechs(ca)
	// The largest POSITIVE mechanism scales the bars; a mechanism that currently costs more than
	// it saves (an unrepaid provider write premium → net negative) draws an empty bar, never a
	// phantom-full one. A session with only vDSO avoided calls still gets the section (its own row
	// below), so a passthrough that memo-served turns but shed no tokens is not silently blank.
	maxEq := 0.0
	for _, m := range mechs {
		if m.tokEq > maxEq {
			maxEq = m.tokEq
		}
	}
	// The cold-tool-defer shed counts toward "this session has something to show" even though it
	// prices no tokens, mirroring the producer's own gate (gateway.cacheAttributionVars folds
	// sum.DeferColdCount into the same emit decision). Without this term a defer-ON session with no
	// token slice decoded a populated block and then threw it away behind "nothing to ablate on a
	// cold/passthrough session" — the pane calling a session cold while holding the witness that
	// fak had deferred N tools on it. That is precisely the defer-on session #3647 asks to surface.
	if maxEq <= 0 && ca.FakVDSOAvoidedCalls == 0 && !hasColdToolDeferShed(ca) {
		// A zero shed has TWO readings the bars cannot tell apart (#5430): a genuinely cold session
		// that had nothing to shed, and a session where the compaction lever ran EVERY turn and
		// bailed — a silent, stable misconfiguration an operator can sit on for hours while this
		// pane shows a calm zero. When the gateway witnessed compaction attempts, the session was
		// not cold, so say what the gate actually did instead of calling it a passthrough.
		if gate := compactionGateRows(ctx.v.Adjudication); len(gate) > 0 {
			return append(append(rows,
				" ablation  no cache savings witnessed yet — compaction ran and shed nothing; its gate says why:"), gate...)
		}
		return append(rows, " ablation  no cache savings witnessed yet — nothing to ablate on a cold/passthrough session")
	}
	rows = append(rows, " ablation  turn a mechanism off → tokens you'd lose (click a row for how it's priced):")
	barW := clampIntTUI(width-48, 8, 24)
	for i, m := range mechs {
		frac := 0.0
		if maxEq > 0 {
			frac = m.tokEq / maxEq
		}
		// A 3-cell marker slot leads every row so an expanded mechanism is set off by a » gutter
		// without shifting the bars — unselected rows keep the original 3-space indent byte-for-byte.
		mark := "   "
		if ctx.cacheMech == i+1 {
			mark = " » "
		}
		row := fmt.Sprintf("%s%s %s %s tok", mark, padRightTUI(m.label, 22), gaugeBarTUI(frac, barW), ablatePadLeft(ablateSignedTokEq(m.tokEq), 8))
		if m.detail != "" {
			row += "  " + m.detail
		}
		rows = append(rows, row)
		if ctx.cacheMech == i+1 { // expanded: spell out where this mechanism's token-equiv came from
			rows = append(rows, cacheMechDetailLines(i, ca, m.tokEq)...)
		}
		// The shed bar prices WHAT was shed; the gate row below it says whether the lever engaged at
		// all and, if not, why (#5430). It rides directly under the mechanism it explains so the two
		// read as one story instead of a number and an unrelated diagnostic. Bound by LABEL, not by
		// slot: see compactionShedMechLabel.
		if m.label == compactionShedMechLabel {
			rows = append(rows, compactionGateRows(ctx.v.Adjudication)...)
		}
	}
	if ca.FakVDSOAvoidedCalls > 0 {
		rows = append(rows, fmt.Sprintf("   %s ·  %s engine calls avoided (not tokens)",
			padRightTUI("fak vDSO memo", 22), guardInfoShortCount(int(ca.FakVDSOAvoidedCalls))))
	}
	return append(rows, deferColdAblationRows(ca)...)
}

// compactionShedMechLabel is the cacheAblationMechs LABEL of the "fak compaction shed" bar — the
// mechanism the compaction gate row explains. The gate row is placed by matching this label, the
// same identity applyInfoCacheMechClick hit-tests a clicked row with, rather than by a slot number:
// an index would have moved the gate row silently under a different mechanism the first time the
// cacheAblationMechs order changed, and no compiler or test can check a bare integer. The label
// match makes that drift impossible by construction.
//
// cacheMechDetailLines still expands the shed provenance under `case 1`, which is a slot number; it
// is a switch on the CALLER's mechIdx and cannot key off a label, so
// TestCompactionGateRowBindsToTheShedBarByLabel pins slot 1 to this label to keep that binding
// honest too.
const compactionShedMechLabel = "fak compaction shed"

// compactionGateLabel labels the live compaction-gate posture row. Like deferColdMechLabel it is
// deliberately NOT a substring of any cacheAblationMechs label ("fak compaction shed"), so
// applyInfoCacheMechClick's Contains hit-test can never mistake this diagnostic for a priced bar
// and expand a mechanism nobody clicked.
const compactionGateLabel = "fak compaction gate"

// compactionGateRows says WHY the compaction shed bar reads what it does: the budget the gate is
// ACTUALLY running at and the dominant bail reason. Both already ride /debug/vars under
// `adjudication.compaction_*` (gateway.AdjudicationSummary) and are already decoded by this pane —
// they were simply never rendered in the plain-words view. Issue #5430: a permanently disengaged
// lever draws a ZERO shed bar indistinguishable from a session that had nothing to shed, so the
// failure is silent AND stable. The guard EXIT summary has printed this since #1407/#1408
// (guard_format.go's "compaction" section); this reads the same counters in the LIVE view, where
// someone would notice in time to fix it. Not byte-for-byte the same rendering, and deliberately
// not the same reason SET: the exit summary lists every bail reason it saw, this row ranks ONE
// dominant reason over compaction CANDIDATES only (agent.CompactBailPreEligible), and its
// malformed_body gloss is its own (guard_format's fault note covers only the three cache-burst
// reasons). Where the two do overlap the wording is kept identical on purpose.
//
// Showing the EFFECTIVE budget is half the point: a `fak guard` launch that did not pass
// --compact-history-budget does not run at the flag's printed default — resolveGuardCompactBudget
// substitutes gateway.HeadlessCompactHistoryBudget — so the number here, read off the running
// gateway, is the only one an operator can trust.
//
// Nil when the gateway attempted no compaction at all (an older gateway, a non-Anthropic route, or
// a session that never reached the transform), so a pane with no compaction witness stays
// byte-identical rather than fabricating a posture.
func compactionGateRows(adj *gateway.AdjudicationSummary) []string {
	if adj == nil || adj.CompactionFired+adj.CompactionBailed+adj.CompactionOff == 0 {
		return nil
	}
	state := fmt.Sprintf("budget %s tok · %d fired / %d bailed",
		guardInfoShortCount(adj.CompactionBudget), adj.CompactionFired, adj.CompactionBailed)
	if adj.CompactionBudget <= 0 {
		state = "OFF — budget 0, body forwarded byte-for-byte"
	}
	rows := []string{fmt.Sprintf("   %s ·  %s", padRightTUI(compactionGateLabel, 22), state)}
	reason, n := dominantCompactionBail(adj.CompactionBailReasons)
	// The pre-eligible lump is held OUT of the ranking, so the fired/bailed count on the posture row
	// above and the "x%d" here are counted over different populations. Disclosing the held-out total
	// is what keeps that from reading as an arithmetic error ("190 bailed … dominant x3").
	held := preEligibleBailTotal(adj.CompactionBailReasons)
	// Indented past the 22-cell label column and its " · " gutter so the reason hangs under the
	// posture and the pair reads as one block — the same shape deferColdAblationRows uses.
	if reason == "" {
		if held == 0 { // nothing bailed at all: no reason line rather than an invented one
			return rows
		}
		// Every bail was decided BEFORE a compactible span existed, so there is no compaction-health
		// answer to give. Saying that is the honest degrade; falling silent here would have left the
		// operator staring at "N bailed" with no line at all, and naming the biggest pre-eligible
		// reason would have pointed them at the one bucket with nothing to do.
		return append(rows, fmt.Sprintf("   %s    no candidate bails: all %d were pre-eligible (decided before any compactible span existed) — says nothing about compaction health",
			padRightTUI("", 22), held))
	}
	line := fmt.Sprintf("   %s    dominant bail: %s x%d", padRightTUI("", 22), reason, n)
	if held > 0 {
		line += fmt.Sprintf(" (of %d candidate bails; %d pre-eligible held out)", candidateBailTotal(adj.CompactionBailReasons), held)
	}
	if gloss := compactionBailGloss(reason, adj); gloss != "" {
		line += " — " + gloss
	}
	return append(rows, line)
}

// preEligibleBailTotal and candidateBailTotal split the bail lump the way
// agent.CompactBailPreEligible does: requests the compactor declined BEFORE any compactible span
// existed (non_json, no_messages_key, decode_failed, too_few_msgs) versus real candidates it
// declined or aborted afterwards. The split exists so the gate row can SHOW its own denominator —
// it ranks over candidates, while the posture row's "N bailed" is the raw gateway counter.
func preEligibleBailTotal(reasons map[string]uint64) uint64 {
	total := uint64(0)
	for r, n := range reasons {
		if agent.CompactBailPreEligible(r) {
			total += n
		}
	}
	return total
}

func candidateBailTotal(reasons map[string]uint64) uint64 {
	total := uint64(0)
	for r, n := range reasons {
		if !agent.CompactBailPreEligible(r) {
			total += n
		}
	}
	return total
}

// dominantCompactionBail picks the reason that explains most of the bailed lump — the one field
// #5430 asks for, since "51 bailed" alone is uninterpretable. Ties break on the reason NAME
// (sortedMapKeys) so the row is stable turn-over-turn instead of flickering between two
// equally-common reasons as Go's map iteration reshuffles.
//
// It ranks over compaction CANDIDATES only: agent.CompactBailPreEligible names the identity-returns
// the compactor decided before any compactible span existed (non_json, no_messages_key,
// decode_failed, too_few_msgs), and those are both benign and by far the highest-volume — the
// compactor is attempted on every Anthropic passthrough, so a session's auxiliary pings pile into
// too_few_msgs. Ranking the raw map hands the row to that bucket on ordinary mixed traffic and
// points the operator at the one group with nothing to do, while the actionable under_budget /
// burst_unprofitable bail that IS the #5430 failure hides underneath it. HEAD already restricts its
// offline twin the same way (gatewayusageledger's TopBailReason is computed over candBailReasons),
// and internal/gatewayusageledger/compaction_noncandidate_test.go pins the scenario: too_few_msgs
// x190 must not beat under_budget x3.
//
// An UNREGISTERED reason ranks as a candidate (CompactBailPreEligible reports false), so a future
// CompactReason* can never be silently dropped from the row — the same conservative direction the
// vocabulary owner chose.
//
// Empty name when nothing bailed AND when every bail was pre-eligible; the caller distinguishes the
// two with preEligibleBailTotal and says so, rather than printing a false "none".
func dominantCompactionBail(reasons map[string]uint64) (string, uint64) {
	best, bestN := "", uint64(0)
	for _, r := range sortedMapKeys(reasons) {
		if agent.CompactBailPreEligible(r) {
			continue
		}
		if reasons[r] > bestN {
			best, bestN = r, reasons[r]
		}
	}
	return best, bestN
}

// compactionBailGloss turns a closed agent.CompactReason* string into the clause that tells an
// operator what to DO about it. The count alone cannot: "under_budget x51" reads as a benign short
// session, yet on a long one it is exactly the #5430 failure. The budget is applied to the
// messages[] array ALONE — CompactAnthropicHistoryWithOptions sums estimateElementTokens over
// message elements and never counts the system+tools block — so an operator sizing it from "my
// context is N tokens" sets it too HIGH in the safe-looking direction and the cut never engages.
// Naming that unit right on the bail row is the cheapest place to stop the misreading.
//
// An unrecognised reason returns no clause rather than a guess, so a future CompactReason* prints
// its raw name and count instead of an invented explanation.
func compactionBailGloss(reason string, adj *gateway.AdjudicationSummary) string {
	switch reason {
	case "under_budget":
		// ANCHOR-STARVED is a SUBSET of under_budget and operationally its opposite: the protected
		// prefix already exceeds the budget, so no budget value can make the cut fire (#1407). When
		// it is present the plain "lower the budget" advice would send the operator the wrong way.
		if adj.CompactionAnchorStarved > 0 {
			return fmt.Sprintf("ANCHOR-STARVED x%d — the protected prefix already exceeds the budget, so NO budget value makes it fire; needs a re-anchor (#1407)", adj.CompactionAnchorStarved)
		}
		return "the messages[] suffix never exceeded the budget (the system+tools block is NOT counted) — a LOWER --compact-history-budget is what engages the cut"
	case "burst_unprofitable":
		return "shed-able middle, but no repaying turn horizon — needs a bounded turn/context budget, an idle gap, or --compact-solvency-floor; never a tighter budget"
	case "no_breakpoint":
		return "no cache_control anchor to protect a prefix behind — the cut cannot fire on this traffic"
	case "too_few_msgs":
		// Pre-eligible: dominantCompactionBail no longer ranks it, so this clause is unreachable from
		// the gate row today. Kept because the gloss is a reason→clause map for any caller, and a
		// half-populated one would read as "fak has no explanation for this reason".
		return "too few messages to drop one safely (benign)"
	case "window_no_drop":
		return "the kept window swallowed the whole suffix — nothing left to drop"
	case "cached_span":
		return "the candidate drop would have deleted cache_control-marked history"
	case "prefix_mismatch", "splice_failed", "redecode_failed":
		// Verbatim the guard exit summary's note for the same three reasons (guard_format.go), so the
		// live row and the exit banner cannot tell an operator two different stories about a fault.
		return "⚠ fak-fault: a fired rewrite would have burst the cache — must stay 0"
	case "malformed_body":
		// NOT a cache burst, and folding it into the sentence above said it was. This abort fires when
		// the splice re-decodes for fak but leaves an Anthropic-invalid message (empty text/content),
		// which the API answers with a 400 (internal/agent/anthropic_compact.go's
		// CompactReasonMalformedBody) — the protected prefix is intact, so the cache was never at
		// risk. Same severity, different failure: still fak's own bug, still must stay 0.
		return "⚠ fak-fault: the spliced body was Anthropic-invalid (empty text/content) and would have 400'd — the cache was not at risk, but it must stay 0"
	}
	return ""
}

// deferColdMechLabel labels the cold-tool-defer block's count row. It is deliberately NOT a
// substring of any cacheAblationMechs label, so applyInfoCacheMechClick's Contains hit-test can
// never mistake this informational row for a priced bar and expand a mechanism nobody clicked.
const deferColdMechLabel = "fak cold-tool defer"

// deferColdMaxNamed caps how many deferred tool names the block spells out before collapsing the
// tail into "+N more". The cold tail on an MCP-heavy session runs to dozens of tools, and a row
// naming every one would dwarf the section it belongs to. The producer sorts the names
// (gateway.toolDeferNamesSnapshot), so the prefix shown is stable turn-over-turn rather than a
// reshuffling sample.
const deferColdMaxNamed = 6

// hasColdToolDeferShed reports whether the session witnessed the cold-tool-defer lever firing. Any
// one of the three counters is sufficient: a producer that reported names but lost its count (or
// vice versa) still describes a session where the lever fired, and suppressing the block on a
// partial witness would hide the very thing #3647 exists to surface.
func hasColdToolDeferShed(ca *guardInfoCacheAttribution) bool {
	return ca.FakDeferColdCount > 0 || ca.FakDeferColdTurns > 0 || len(ca.FakDeferColdToolNames) > 0
}

// deferColdAblationRows renders the cold-tool-DEFER shed (#3647, the --defer-cold-tools lever) as
// its OWN block below the priced bars, never as a savings bar. This is the THIRD shed mechanism and
// the one most easily conflated with the compaction shed above it: both "drop" something, but defer
// shrinks NO request bytes and buys NO token-equiv — every def still ships on the wire, and the
// reduction is provider-side (only the hot core loads into context; a cold schema faults in on
// demand), so it is OBSERVED in the usage relay rather than priced here. Giving it the vDSO line's
// " · " gutter instead of a bar is what keeps that distinction legible at a glance: in this section
// a bar means priced tokens, a · means a counted event. The second row names WHICH tools went cold
// — the question the raw _tool_defer_* counters could not answer, and the reason an operator can
// now tell "the lever fired" from "the lever fired on the tools I actually needed".
//
// Nil when the lever never fired, so a defer-off session's pane stays byte-identical to before.
func deferColdAblationRows(ca *guardInfoCacheAttribution) []string {
	if !hasColdToolDeferShed(ca) {
		return nil
	}
	count, turns := int(ca.FakDeferColdCount), int(ca.FakDeferColdTurns)
	rows := []string{fmt.Sprintf("   %s ·  %s cold tool %s deferred over %s %s (provider-side, not tokens)",
		padRightTUI(deferColdMechLabel, 22),
		guardInfoShortCount(count), pluralWord(count, "def", "defs"),
		guardInfoShortCount(turns), pluralWord(turns, "turn", "turns"))}
	names := ca.FakDeferColdToolNames
	if len(names) == 0 {
		return rows
	}
	shown, extra := names, ""
	if len(shown) > deferColdMaxNamed {
		shown = shown[:deferColdMaxNamed]
		extra = fmt.Sprintf(" (+%d more)", len(names)-deferColdMaxNamed)
	}
	// Indented past the 22-cell label column and its " · " gutter so the names hang under the count
	// clause and the two rows read as one block instead of two unrelated lines.
	return append(rows, fmt.Sprintf("   %s    deferred: %s%s",
		padRightTUI("", 22), strings.Join(shown, ", "), extra))
}

// providerAblationDetail spells the provider prompt-cache NET as its read rebate and its write
// premium, so an operator sees WHY the net is what it is: a large rebate mostly eaten by cold
// writes reads very differently from a clean win, and a still-negative net is the "writes not yet
// repaid" story. The write field is stored signed (negative until reads repay writes), so it
// prints its own sign. Empty when the gateway emitted only the net (an older producer), so the
// bar row degrades to just its token-equiv rather than a half-blank clause.
func providerAblationDetail(ca *guardInfoCacheAttribution) string {
	read := ca.ProviderPromptCacheReadTokenEquiv
	write := ca.ProviderPromptCacheWritePremiumTokenEquiv
	if read == 0 && write == 0 {
		return ""
	}
	return fmt.Sprintf("reads %s · writes %s", gateway.HumanTokenEquiv(read), gateway.HumanTokenEquiv(write))
}

// shedAblationDetail explains why the compaction-shed bar is priced BELOW the raw shed count on a
// warm session: the warm portion — tokens the provider was already serving as cache_reads — is
// worth only the read marginal when dropped, not fresh input, so the honest saving is the blended
// value (cacheprice.ShedTokenEquiv), not the raw token count. This is the clause that reconciles a
// small shed BAR against a large raw shed the operator may have seen elsewhere. Empty when the raw
// shed already prices at full (a cold session, no discount) or nothing was shed, so the row stays
// clean unless there is a discount to explain.
func shedAblationDetail(ca *guardInfoCacheAttribution, shedTokEq float64) string {
	raw := float64(ca.FakCompactionShedTokens)
	if raw <= 0 || shedTokEq >= raw {
		return ""
	}
	// Name the DELTA the operator is squinting at — the raw shed count they may have seen elsewhere
	// vs the honest bar value — right on the row, so the gap explains itself at a glance instead of
	// only after a click. "worth Nx less (already cached)" turns the unexplained ~10× into its own
	// one-line reason: a warm token was already a 0.1× cache read, so dropping it can only save that
	// 0.1×. The full causal story is one click away in cacheMechDetailLines.
	return fmt.Sprintf("shed %s tok · worth %s× less (already cached)",
		gateway.HumanTokenEquiv(raw), shedDeltaFactor(raw, shedTokEq))
}

// shedDeltaFactor formats how many times smaller the honest shed VALUE is than its raw token COUNT
// — the user-facing name for the warm discount, and the plain answer to "why is the bar ~10× below
// the tokens I saw shed?". A fully-warm shed prices at the 0.1× cache-read marginal, i.e. ~10× less;
// a half-warm shed lands in between. One decimal below 10 keeps a mixed 5.3× honest; at/above 10 the
// fraction is noise. Guards a zero value (all-warm rounding) to the 10× fully-cached ceiling so the
// clause never divides by zero or prints an infinity.
func shedDeltaFactor(raw, tokEq float64) string {
	if tokEq <= 0 {
		return "10"
	}
	f := raw / tokEq
	if f < 10 {
		return fmt.Sprintf("%.1f", f)
	}
	return fmt.Sprintf("%.0f", f)
}

// cacheMechDetailLines is the expanded provenance for one ablation mechanism (mechIdx 0-based),
// shown indented under its bar when the operator clicks the row. Each mechanism answers "why is
// this token-equiv what it is" in the SAME input-token currency the bar and the rollup use, so the
// drill-down deepens the number rather than restating it. tokEq is the mechanism's already-priced
// bar value (passed in so the panel and the detail can never round differently).
func cacheMechDetailLines(mechIdx int, ca *guardInfoCacheAttribution, tokEq float64) []string {
	switch mechIdx {
	case 0: // provider prompt-cache: the net is the read rebate net of the (signed) write premium
		return []string{
			fmt.Sprintf("     └ read rebate %s, write premium %s → %s tok-equiv net",
				gateway.HumanTokenEquiv(ca.ProviderPromptCacheReadTokenEquiv),
				gateway.HumanTokenEquiv(ca.ProviderPromptCacheWritePremiumTokenEquiv),
				gateway.HumanTokenEquiv(tokEq)),
			"       provider-owned; same currency as offline `fak ablate` provider_tokeq",
		}
	case 1: // fak compaction shed: the warm/cold blend that makes the value ≤ the raw shed count
		shed := ca.FakCompactionShedTokens
		warm := ca.FakCompactionCacheReadTokens
		if warm > shed {
			warm = shed
		}
		cold := shed - warm
		return []string{
			fmt.Sprintf("     └ %s shed: %s warm @0.1× (already cached) + %s cold @1×",
				gateway.HumanTokenEquiv(float64(shed)), gateway.HumanTokenEquiv(float64(warm)), gateway.HumanTokenEquiv(float64(cold))),
			fmt.Sprintf("       → %s tok-equiv — the bar shows this value, not the raw shed count", gateway.HumanTokenEquiv(tokEq)),
			// The intuition the numbers alone don't give — the plain-English answer to "why the ~10×
			// gap?". A cached token is already cheap to KEEP (0.1×), so it is only that cheap to DROP;
			// the gap the operator sees is not fak under-valuing its work, it is value the provider's
			// cache already banked, which fak cannot save a second time.
			"       already-cached tokens are cheap to keep (0.1×), so cheap to drop:",
			fmt.Sprintf("       shedding it saves only 0.1× — the %s× gap was already banked by the cache",
				shedDeltaFactor(float64(shed), tokEq)),
		}
	case 2: // fak KV-prefix reuse: reused prefix, valued at the full local marginal
		return []string{
			fmt.Sprintf("     └ %s tokens reused from the in-kernel KV prefix,", gateway.HumanTokenEquiv(float64(ca.FakKVPrefixReusedTokens))),
			"       valued at the full local marginal (1×) — no provider round-trip avoided",
		}
	}
	return nil
}

// applyInfoCacheMechClick folds a Cache-tab body click into the expanded-mechanism state. It
// resolves the click against the ACTUALLY-RENDERED rows (re-rendering the exact block the operator
// clicked, pre-color) and matches the clicked row to a mechanism by its label — so the hit-test
// rides the same scroll window / indicator rows / width caps the renderer produced and can never
// drift from a separately-derived coordinate. A click on a mechanism's bar toggles its detail
// (re-clicking the open one collapses it); a click on any other row (rule, framing, detail, vDSO,
// or off the Cache tab entirely) is inert and returns the state unchanged. blockY is block-relative
// (1 = tab bar), matching blockRelativeRow's output.
func applyInfoCacheMechClick(s infoViewState, v guardInfoVars, tr *guardInfoTrend, width, height, blockY int) infoViewState {
	if s.active != viewCache || s.glossaryOpen || blockY < 2 || v.CacheAttribution == nil {
		return s
	}
	rows := strings.Split(renderGuardInfoInteractiveBlock(s, v, tr, width, height), "\n")
	if blockY-1 >= len(rows) {
		return s
	}
	line := rows[blockY-1]
	for i, m := range cacheAblationMechs(v.CacheAttribution) {
		// Only a mechanism BAR row carries the exact label; the framing/detail/vDSO rows do not, so
		// Contains uniquely identifies the clicked bar even through the » marker gutter.
		if strings.Contains(line, m.label) {
			if s.cacheMech == i+1 {
				s.cacheMech = 0
			} else {
				s.cacheMech = i + 1
			}
			return s
		}
	}
	return s
}
